"""DCF Monte Carlo (Spec §1.5; Ref 3.3).

Steps, numbered as in the specification:

1. Correlated annual shocks: Sigma_shocks = L L^T, z ~ N(0, I), [eps_t, eta_t, omega_t] = L z.
   A new set of shocks is drawn for every year.
2. Growth and margin are updated with mean-reverting processes:
       g_t = g_bar + phi (g_{t-1} - g_bar) + eps_t
       m_t = m_sector + psi (m_{t-1} - m_sector) + eta_t
   The WACC shock is drawn once per simulation (from the first-year draw) and held constant.
3. A DCF value is calculated for each simulation, applying the three consistency rules
   (reinvestment linked to growth, simulated value-driver terminal value, residual income for
   financial companies; see atlas.fundamentals.dcf).
4. Simulation noise is reduced with antithetic variates (z and -z).
5. Precision is checked with SE = sigma_V / sqrt(n).
6. Parameter uncertainty: phi, psi and g_bar are drawn from their sampling distributions.
7. Conversion into a 12-month return distribution (with the corrections recommended in the Technical
   Reference, Part 1.6):
       R_12,n = beta^T E[f_12] + kappa (V_n - P_0) / P_0 + sigma_res * e_n,   e_n ~ scaled Student-t
   Dividends are not added separately because E[f_12] is a total-return expectation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atlas.fundamentals import dcf


@dataclass
class DCFInputs:
    """Inputs for one company. Rates are decimals (0.06 = 6%)."""

    revenue0: float
    margin0: float              # current operating (EBIT) margin
    g0: float                   # latest annual revenue growth
    g_bar: float = 0.06         # long-run growth (Spec §1.4)
    phi: float = 0.7            # growth persistence
    m_sector: float = 0.18      # sector margin the company reverts toward
    psi: float = 0.8            # margin persistence
    wacc: float = 0.0975
    tax: float = 0.21
    roic_new: float = 0.15      # return on new invested capital (shrunk historical ROIC)
    roic_new_sd: float = 0.03
    net_debt: float = 0.0
    shares: float = 1.0
    g_inf_mean: float = 0.025   # long-run growth, capped at nominal GDP growth
    g_inf_sd: float = 0.005
    g_inf_cap: float = 0.04
    roic_inf_spread: float = 0.02  # ROIC_inf fades toward WACC + sector spread
    roic_inf_spread_sd: float = 0.01
    se_phi: float = 0.05        # standard errors for parameter uncertainty (step 6)
    se_psi: float = 0.05
    se_g_bar: float = 0.005
    shock_cov: np.ndarray = field(default_factory=lambda: np.diag([0.06, 0.03, 0.01]) ** 2)
    financial: bool = False     # True -> residual income model (banks, insurers, REITs)
    book0: float | None = None  # book equity for the residual income model
    roe0: float | None = None
    cost_of_equity: float | None = None


@dataclass
class MCResult:
    values: np.ndarray          # value per share for every simulation
    mean: float
    sd: float
    se: float                   # Monte Carlo standard error of the mean (step 5)
    paths_growth: np.ndarray
    paths_margin: np.ndarray


def run_dcf_monte_carlo(x: DCFInputs, n: int = 10_000, years: int = 5, seed: int | None = 0,
                        antithetic: bool = True, parameter_uncertainty: bool = True) -> MCResult:
    """Simulate n DCF valuations (steps 1–6)."""
    rng = np.random.default_rng(seed)
    half = n // 2 if antithetic else n
    L = np.linalg.cholesky(x.shock_cov)                       # step 1: Sigma = L L^T
    z = rng.standard_normal((half, years, 3))
    if antithetic:                                            # step 4: pair each z with -z
        z = np.concatenate([z, -z], axis=0)
    m = z.shape[0]
    shocks = z @ L.T                                          # [eps, eta, omega] = L z
    eps, eta = shocks[:, :, 0], shocks[:, :, 1]
    omega = shocks[:, 0, 2]                                   # WACC shock: first-year draw, held constant

    # Step 6: parameter uncertainty. Persistence parameters are kept inside (0, 0.99).
    if parameter_uncertainty:
        phi = np.clip(rng.normal(x.phi, x.se_phi, m), 0.0, 0.99)
        psi = np.clip(rng.normal(x.psi, x.se_psi, m), 0.0, 0.99)
        g_bar = rng.normal(x.g_bar, x.se_g_bar, m)
    else:
        phi, psi, g_bar = np.full(m, x.phi), np.full(m, x.psi), np.full(m, x.g_bar)

    wacc_n = np.maximum(x.wacc + omega, 0.03)                 # WACC_n = WACC + omega (floored for safety)

    # Step 2: mean-reverting growth and margin paths.
    g = np.empty((m, years))
    mg = np.empty((m, years))
    g_prev, m_prev = np.full(m, x.g0), np.full(m, x.margin0)
    for t in range(years):
        g[:, t] = g_bar + phi * (g_prev - g_bar) + eps[:, t]
        mg[:, t] = np.clip(x.m_sector + psi * (m_prev - x.m_sector) + eta[:, t], -0.5, 0.9)
        g_prev, m_prev = g[:, t], mg[:, t]

    # Long-run assumptions are simulated too (second consistency rule).
    g_inf = np.minimum(rng.normal(x.g_inf_mean, x.g_inf_sd, m), x.g_inf_cap)
    roic_inf = wacc_n + rng.normal(x.roic_inf_spread, x.roic_inf_spread_sd, m)
    roic_inf = np.maximum(roic_inf, g_inf + 1e-3)
    roic_new = np.maximum(rng.normal(x.roic_new, x.roic_new_sd, m), 0.02)

    if x.financial:
        # Third consistency rule: residual income model on book equity and ROE.
        if x.book0 is None or x.roe0 is None:
            raise ValueError("Financial companies need book0 and roe0 for the residual income model.")
        roe = x.roe0 * (1 + mg - x.margin0)  # ROE moves with the simulated profitability path
        re = x.cost_of_equity or x.wacc
        value = dcf.residual_income_value(x.book0, roe, re, g_inf=float(np.mean(g_inf)))
        per_share = value / x.shares
    else:
        rev = x.revenue0 * np.cumprod(1 + g, axis=1)
        nopat = rev * mg * (1 - x.tax)
        g_next = np.concatenate([g[:, 1:], g_inf[:, None]], axis=1)
        reinvest = g_next / roic_new[:, None] * nopat          # first consistency rule
        fcf = nopat - reinvest
        nopat_next = nopat[:, -1] * (1 + g_inf)
        tv = dcf.value_driver_terminal(nopat_next, g_inf, roic_inf, wacc_n)
        ev = dcf.discounted_value(fcf, tv, wacc_n)
        per_share = (ev - x.net_debt) / x.shares

    sd = float(np.std(per_share, ddof=1))
    return MCResult(per_share, float(np.mean(per_share)), sd, sd / np.sqrt(len(per_share)), g, mg)


def convert_to_12m_returns(values: np.ndarray, price: float, kappa: float, factor_return_12m: float,
                           resid_vol_12m: float, nu: float = 5.0, seed: int | None = 1) -> np.ndarray:
    """Step 7: convert simulated values into 12-month total returns.

    The gap-closing term kappa (V_n - P_0) / P_0 is the part of the valuation gap expected to close
    within a year. The factor term is the expected 12-month return from market and sector exposure
    (Spec §7.4). The Student-t noise term with the stock's residual volatility adds ordinary price
    variation; without it the distribution is far too narrow (Technical Reference, Part 1.6).
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_t(nu, len(values)) / np.sqrt(nu / (nu - 2))  # unit-variance Student-t
    return factor_return_12m + kappa * (values - price) / price + resid_vol_12m * e


def scenario_probabilities(returns: np.ndarray, threshold: float = 0.15) -> dict[str, float]:
    """Bull/base/bear probabilities counted on simulated returns (Spec §A.3 definition)."""
    r = np.asarray(returns)
    bull = float(np.mean(r > threshold))
    bear = float(np.mean(r < -threshold))
    return {"p_bull": bull, "p_base": 1 - bull - bear, "p_bear": bear}


def estimate_kappa(valuation_gap: np.ndarray, residual_return_12m: np.ndarray) -> tuple[float, float]:
    """Gap-closing speed kappa from history (Spec §1.5, step 7):

        R_{i,t->t+12} - beta^T f_{t->t+12} = a + kappa (V - P)/P + e

    Inputs are pooled past point-in-time valuation gaps and the following residual 12-month returns.
    Returns (kappa, standard error). Overlapping 12-month windows make the naive standard error too
    small, so it should be interpreted with the effective sample size of Spec §7.7.
    """
    x = np.asarray(valuation_gap, float)
    y = np.asarray(residual_return_12m, float)
    ok = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ beta
    s2 = resid @ resid / max(ok.sum() - 2, 1)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(beta[1]), float(se)
