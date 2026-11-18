"""Position sizing and portfolio construction (Spec §8.6; Ref 3.10).

Kelly bound per position:  f* = mu / sigma^2, of which a fraction (1/4 to 1/2) is used as a cap.
Example (Spec §8.6 note): mu = 4%, sigma = 32% -> f* = 39%, so full Kelly is far too aggressive.

Transaction cost of a trade Δw_i (fractions of AUM):
    TC_i = |Δw_i| (spread_i / 2) + Y sigma_i |Δw_i| sqrt(|Δw_i| AUM / ADV$_i) + b_i (h/252) max(-w_i, 0)
The last term is the borrow fee, charged only on short positions for a holding period of h days.

Optimisation:
    max_w  mu^T w - lambda w^T Sigma w - TC(w - w_prev)
    s.t.   |B_sector^T w| <= eps,  |beta^T w| <= eps,  |w_i| <= w_max,
           sqrt(w^T Sigma w) <= sigma_target,  sum |w_i| <= gross_max,  |sum w_i| <= net_max,
           w_i >= 0 for stocks that cannot be borrowed (utilisation >= 80% or fee >= 5%).
cvxpy (OSQP / Clarabel) is used when installed. Otherwise the same problem is solved with scipy's
SLSQP, with the weights split into long and short parts (w = p - n, p, n >= 0) so that the absolute
values in the gross-exposure and cost terms become smooth.

Drawdown rule: gross exposure is cut by 50% when drawdown exceeds 8% and restored at 4%.
Capacity: AUM* is the AUM at which net alpha falls to half of gross alpha (found by bisection).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from atlas._optional import optional_import


def kelly_cap(mu, sigma, fraction: float = 0.25):
    return fraction * np.asarray(mu) / np.asarray(sigma) ** 2


def transaction_cost(dw, spread, sigma_daily, adv_dollars, aum: float, Y: float = 0.8,
                     w_new=None, borrow_fee=None, holding_days: int = 21) -> np.ndarray:
    dw = np.abs(np.asarray(dw, float))
    tc = dw * np.asarray(spread) / 2 + Y * np.asarray(sigma_daily) * dw * np.sqrt(dw * aum / np.asarray(adv_dollars))
    if w_new is not None and borrow_fee is not None:
        tc = tc + np.asarray(borrow_fee) * holding_days / 252 * np.maximum(-np.asarray(w_new), 0)
    return tc


@dataclass
class OptimiserInputs:
    mu: pd.Series                 # expected return over the holding period
    Sigma: pd.DataFrame           # covariance over the same period
    sector_exposure: pd.DataFrame  # rows stocks, columns sector dummies
    beta: pd.Series
    w_prev: pd.Series | None = None
    spread: pd.Series | None = None
    sigma_daily: pd.Series | None = None
    adv_dollars: pd.Series | None = None
    borrow_fee: pd.Series | None = None
    utilisation: pd.Series | None = None


def shortable(borrow_fee: pd.Series | None, utilisation: pd.Series | None, index, max_fee=0.05, max_util=0.80) -> np.ndarray:
    ok = np.ones(len(index), bool)
    if borrow_fee is not None:
        ok &= (borrow_fee.reindex(index).fillna(0.0) < max_fee).values
    if utilisation is not None:
        ok &= (utilisation.reindex(index).fillna(0.0) < max_util).values
    return ok


def optimise(x: OptimiserInputs, risk_aversion: float = 5.0, w_max: float = 0.03, gross_max: float = 3.0,
             net_max: float = 0.05, sigma_target: float | None = None, exposure_eps: float = 0.01,
             aum: float = 1e8, Y: float = 0.8, holding_days: int = 21, cost_amortisation: float = 1.0,
             use_cvxpy: bool = True) -> pd.Series:
    """Solve the optimisation of Spec §8.6.

    cost_amortisation spreads the one-off trading cost over the expected number of holding periods of
    a position (for a signal that persists for about three months and monthly rebalancing, 3), so that
    a one-month alpha is not charged the full cost of entering and leaving a position every month.
    Borrow fees are recurring and are not amortised.
    """
    idx = x.mu.index
    n = len(idx)
    mu = x.mu.values
    S = x.Sigma.reindex(index=idx, columns=idx).values
    S = (S + S.T) / 2
    Bs = x.sector_exposure.reindex(idx).fillna(0.0).values
    beta = x.beta.reindex(idx).fillna(1.0).values
    w0 = (x.w_prev.reindex(idx).fillna(0.0) if x.w_prev is not None else pd.Series(0.0, index=idx)).values
    spread = (x.spread.reindex(idx).fillna(0.001) if x.spread is not None else pd.Series(0.001, index=idx)).values
    sd = (x.sigma_daily.reindex(idx).fillna(0.02) if x.sigma_daily is not None else pd.Series(0.02, index=idx)).values
    adv = (x.adv_dollars.reindex(idx).fillna(1e7) if x.adv_dollars is not None else pd.Series(1e7, index=idx)).values
    fee = (x.borrow_fee.reindex(idx).fillna(0.0) if x.borrow_fee is not None else pd.Series(0.0, index=idx)).values
    can_short = shortable(x.borrow_fee, x.utilisation, idx)

    cp = optional_import("cvxpy") if use_cvxpy else None
    if cp is not None:
        w = cp.Variable(n)
        dw = w - w0
        # Impact term |dw|^{1.5} is convex: Y sigma sqrt(AUM/ADV) |dw|^{3/2}.
        cost = ((cp.sum(cp.multiply(spread / 2, cp.abs(dw)))
                 + cp.sum(cp.multiply(Y * sd * np.sqrt(aum / adv), cp.power(cp.abs(dw), 1.5)))) / cost_amortisation
                + cp.sum(cp.multiply(fee * holding_days / 252, cp.neg(w))))
        obj = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(S)) - cost)
        cons = [cp.abs(Bs.T @ w) <= exposure_eps, cp.abs(beta @ w) <= exposure_eps, cp.abs(w) <= w_max,
                cp.norm1(w) <= gross_max, cp.abs(cp.sum(w)) <= net_max]
        if (~can_short).any():
            cons.append(w[np.nonzero(~can_short)[0]] >= 0)
        if sigma_target is not None:
            cons.append(cp.quad_form(w, cp.psd_wrap(S)) <= sigma_target ** 2)
        cp.Problem(obj, cons).solve()
        if w.value is not None:
            return pd.Series(np.asarray(w.value).ravel(), index=idx, name="weight")

    # ---- scipy fallback: w = p - q with p, q >= 0 ---------------------------------------------
    def split(z):
        return z[:n], z[n:]

    def objective(z):
        p, q = split(z)
        w = p - q
        dw = np.abs(w - w0) + 1e-12
        cost = (((spread / 2 * dw).sum() + (Y * sd * np.sqrt(aum / adv) * dw ** 1.5).sum()) / cost_amortisation
                + (fee * holding_days / 252 * q).sum())
        return -(mu @ w - risk_aversion * w @ S @ w - cost)

    def grad(z):
        p, q = split(z)
        w = p - q
        d = w - w0
        sgn = np.sign(d)
        # Gradient of the (positive) objective with respect to w, then chained to p (+1) and q (-1).
        gw = mu - 2 * risk_aversion * S @ w - (spread / 2 * sgn + 1.5 * Y * sd * np.sqrt(aum / adv) * np.sqrt(np.abs(d)) * sgn) / cost_amortisation
        g_p = gw
        g_q = -gw - fee * holding_days / 252   # the borrow fee is charged on the short part q
        return -np.concatenate([g_p, g_q])

    cons = []
    for j in range(Bs.shape[1]):
        b = Bs[:, j]
        cons += [{"type": "ineq", "fun": lambda z, b=b: exposure_eps - b @ (z[:n] - z[n:])},
                 {"type": "ineq", "fun": lambda z, b=b: exposure_eps + b @ (z[:n] - z[n:])}]
    cons += [{"type": "ineq", "fun": lambda z: exposure_eps - beta @ (z[:n] - z[n:])},
             {"type": "ineq", "fun": lambda z: exposure_eps + beta @ (z[:n] - z[n:])},
             {"type": "ineq", "fun": lambda z: gross_max - z.sum()},
             {"type": "ineq", "fun": lambda z: net_max - (z[:n] - z[n:]).sum()},
             {"type": "ineq", "fun": lambda z: net_max + (z[:n] - z[n:]).sum()}]
    if sigma_target is not None:
        cons.append({"type": "ineq", "fun": lambda z: sigma_target ** 2 - (z[:n] - z[n:]) @ S @ (z[:n] - z[n:])})
    bounds = [(0, w_max)] * n + [(0, w_max if can_short[i] else 0.0) for i in range(n)]
    z0 = np.concatenate([np.maximum(w0, 0), np.maximum(-w0, 0)])
    res = minimize(objective, z0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-10})
    p, q = split(res.x)
    return pd.Series(p - q, index=idx, name="weight")


def optimise_to_target_vol(x: OptimiserInputs, sigma_target: float, risk_aversion: float = 2.0, n_iter: int = 4,
                           **kw) -> tuple[pd.Series, float]:
    """Choose the risk aversion lambda so the portfolio's predicted volatility approaches the target.

    Without costs and binding limits the optimal weights scale with 1/lambda, so lambda is updated as
    lambda <- lambda x vol(w) / target and the problem re-solved a few times. Position, gross and net
    limits stay in force, so the target is an upper bound that weak forecasts may not reach.
    """
    lam = risk_aversion
    S = x.Sigma.reindex(index=x.mu.index, columns=x.mu.index).values
    w = None
    for _ in range(n_iter):
        w = optimise(x, risk_aversion=lam, sigma_target=sigma_target, **kw)
        vol = float(np.sqrt(max(w.values @ S @ w.values, 1e-16)))
        if abs(vol / sigma_target - 1) < 0.05:
            break
        lam = lam * vol / sigma_target
    return w, lam


def drawdown_scale(equity: pd.Series, cut: float = 0.08, restore: float = 0.04, factor: float = 0.5) -> pd.Series:
    """Gross-exposure multiplier implied by the drawdown rule (with hysteresis between 8% and 4%)."""
    peak = equity.cummax()
    dd = 1 - equity / peak
    scale, state = [], 1.0
    for v in dd.values:
        if state == 1.0 and v > cut:
            state = factor
        elif state == factor and v < restore:
            state = 1.0
        scale.append(state)
    return pd.Series(scale, index=equity.index, name="gross_scale")


def capacity(gross_alpha: float, cost_at_aum, lo: float = 1e6, hi: float = 1e11, tol: float = 0.01) -> float:
    """AUM* where net alpha = gross_alpha - cost(AUM) equals gross_alpha / 2.

    cost_at_aum(aum) returns annual costs as a fraction of capital, typically from rerunning the costed
    backtest; it grows like sqrt(AUM) through the impact term, so the solution is found by bisection
    on log AUM.
    """
    f = lambda a: gross_alpha - cost_at_aum(a) - gross_alpha / 2
    if f(lo) < 0:
        return lo
    if f(hi) > 0:
        return hi
    a, b = np.log(lo), np.log(hi)
    while b - a > tol:
        m = (a + b) / 2
        if f(np.exp(m)) > 0:
            a = m
        else:
            b = m
    return float(np.exp((a + b) / 2))
