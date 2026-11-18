"""Risk-neutral distribution from option prices (Spec §1.6; Ref 3.3).

Procedure:
1. Quotes are filtered: positive bid, bid–ask spread below 20% of mid, open interest above 50.
2. Out-of-the-money options are used and implied volatility is calculated (Black–Scholes).
3. A smooth SVI smile is fitted in log-moneyness: w(k) = a + b[rho(k - m) + sqrt((k - m)^2 + s^2)].
4. The fitted smile is converted into call prices on a fine strike grid.
5. Density: f(K) ~= e^{rT} [C(K+h) - 2C(K) + C(K-h)] / h^2   (Breeden–Litzenberger).
6. Tail probability: P(S_T > K) = -e^{rT} dC/dK.
7. Checks: f(K) >= 0 and the density integrates to about one.

These are risk-neutral probabilities: they overweight bad outcomes because investors pay for crash
protection. They are used as features and compared with model probabilities through the adjusted
bear gap (Spec §1.7), never as real-world probabilities directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from atlas._compat import trapezoid as _trapz


def bs_price(S, K, T, r, sigma, call=True):
    """Black–Scholes price of a European option (no dividends)."""
    S, K, sigma = np.asarray(S, float), np.asarray(K, float), np.asarray(sigma, float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, r, sigma):
    """Gamma = phi(d1) / (S sigma sqrt(T)), used for dealer gamma exposure (Spec §4.2)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def implied_vol(price, S, K, T, r, call=True) -> float:
    """Implied volatility by Brent root-finding; NaN when the price violates no-arbitrage bounds."""
    f = lambda s: bs_price(S, K, T, r, s, call) - price  # noqa: E731
    try:
        return float(brentq(f, 1e-4, 5.0))
    except ValueError:
        return float("nan")


def filter_quotes(q: pd.DataFrame, max_rel_spread: float = 0.20, min_oi: int = 50) -> pd.DataFrame:
    """Step 1 filters. Expects columns bid, ask, open_interest."""
    mid = (q["bid"] + q["ask"]) / 2
    ok = (q["bid"] > 0) & ((q["ask"] - q["bid"]) / mid < max_rel_spread) & (q["open_interest"] > min_oi)
    return q[ok].assign(mid=mid[ok])


def svi_total_variance(k, a, b, rho, m, s):
    """SVI raw parameterisation of total implied variance w(k) = sigma_imp^2 T."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit_svi(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Least-squares fit of the five SVI parameters with simple no-arbitrage bounds."""
    x0 = [max(w.min() * 0.5, 1e-4), 0.1, -0.3, 0.0, 0.1]
    lb = [-1.0, 1e-6, -0.999, -2.0, 1e-4]
    ub = [5.0, 5.0, 0.999, 2.0, 5.0]
    res = least_squares(lambda p: svi_total_variance(k, *p) - w, x0, bounds=(lb, ub))
    return res.x


@dataclass
class RiskNeutralDensity:
    strikes: np.ndarray
    density: np.ndarray
    calls: np.ndarray
    integral: float
    nonnegative: bool

    def prob_above(self, K: float) -> float:
        """P(S_T > K) by integrating the density above K."""
        mask = self.strikes >= K
        return float(_trapz(self.density[mask], self.strikes[mask])) if mask.sum() > 1 else 0.0

    def prob_below(self, K: float) -> float:
        mask = self.strikes <= K
        return float(_trapz(self.density[mask], self.strikes[mask])) if mask.sum() > 1 else 0.0


def risk_neutral_density(S: float, T: float, r: float, strikes: np.ndarray, ivs: np.ndarray,
                         grid_points: int = 400) -> RiskNeutralDensity:
    """Steps 3–7 from a set of (strike, implied vol) pairs at one expiry."""
    F = S * np.exp(r * T)
    k = np.log(strikes / F)
    params = fit_svi(k, ivs ** 2 * T)
    grid = np.linspace(strikes.min() * 0.5, strikes.max() * 1.5, grid_points)
    w = np.maximum(svi_total_variance(np.log(grid / F), *params), 1e-8)
    calls = bs_price(S, grid, T, r, np.sqrt(w / T), call=True)
    h = grid[1] - grid[0]
    dens = np.zeros_like(grid)
    dens[1:-1] = np.exp(r * T) * (calls[2:] - 2 * calls[1:-1] + calls[:-2]) / h ** 2
    integral = float(_trapz(dens, grid))
    return RiskNeutralDensity(grid, dens, calls, integral, bool((dens >= -1e-8).all()))
