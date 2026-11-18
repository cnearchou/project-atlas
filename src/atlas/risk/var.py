"""Value at Risk and Expected Shortfall (Spec §8.2; Ref 3.10).

Losses are positive numbers (L = -return).

Filtered historical simulation (FHS):
    z_t = eps_t / sigma_t over the last 500 days;  simulated r = sigma_{T+1} z_t
    VaR_95 = 5th percentile of simulated losses;  ES_95 = mean of losses beyond VaR.
Monte Carlo: the joint simulation of Spec §7.8 applied to portfolio weights.
Extreme Value Theory (peaks over threshold):
    u = 95th percentile of losses, y = L - u for L > u, y ~ GPD(xi, beta) by MLE,
    VaR_q = u + (beta/xi) [ ((N/N_u)(1 - q))^{-xi} - 1 ]
    ES_q  = (VaR_q + beta - xi u) / (1 - xi)
Parametric normal reference (Spec §8.2 note): VaR_95 = 1.645 sigma, ES_95 = 2.063 sigma.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def empirical_var_es(losses, level: float = 0.95) -> tuple[float, float]:
    L = np.asarray(losses, float)
    L = L[np.isfinite(L)]
    var = float(np.quantile(L, level))
    tail = L[L >= var]
    return var, float(tail.mean()) if len(tail) else var


def fhs_var_es(std_resid, sigma_next: float, level: float = 0.95, window: int = 500, mu: float = 0.0) -> tuple[float, float]:
    z = np.asarray(std_resid, float)[-window:]
    sim = mu + sigma_next * z
    return empirical_var_es(-sim, level)


def normal_var_es(sigma: float, level: float = 0.95) -> tuple[float, float]:
    q = stats.norm.ppf(level)
    return float(q * sigma), float(stats.norm.pdf(q) / (1 - level) * sigma)


def evt_var_es(losses, q: float = 0.99, threshold_q: float = 0.95) -> dict:
    """Peaks-over-threshold with a GPD fitted to exceedances (location fixed at 0)."""
    L = np.asarray(losses, float)
    L = L[np.isfinite(L)]
    u = float(np.quantile(L, threshold_q))
    y = L[L > u] - u
    N, Nu = len(L), len(y)
    xi, _, beta = stats.genpareto.fit(y, floc=0)
    if abs(xi) < 1e-6:
        var = u + beta * np.log(Nu / (N * (1 - q)))
    else:
        var = u + beta / xi * ((N / Nu * (1 - q)) ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi) if xi < 1 else np.inf
    return {"var": float(var), "es": float(es), "xi": float(xi), "beta": float(beta), "u": u, "n_exceed": Nu}


def mc_portfolio_var_es(sim_returns: np.ndarray, weights, level: float = 0.95) -> tuple[float, float]:
    pnl = np.asarray(sim_returns) @ np.asarray(weights)
    return empirical_var_es(-pnl, level)
