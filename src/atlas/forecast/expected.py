"""From score to expected return (Spec §7.4; Ref 3.9).

Grinold's formula:
    alpha_i = IC_h x sigma_i,h x z_i
IC_h is the out-of-sample IC at the *same* horizon h as the forecast and sigma_i,h is the residual
volatility over that horizon. A 1-month IC is never applied to a 12-month forecast.

Expected factor returns are regime-weighted:
    E[f_k] = sum_s P(s_t = s) mu~_{k,s},
    mu~_{k,s} = w_s mu_{k,s} + (1 - w_s) f_bar_k,   w_s = n_s / (n_s + n_0)
where mu_{k,s} is the historical mean of factor k in regime s, f_bar_k its long-run mean and n_0 a
prior strength (in months). The market term uses r_f + ERP (Spec §2.7).

Total expected return:  mu_i = beta_i^T E[f] + alpha_i.
Relative scenarios (Section A.3) use alpha_i only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def grinold_alpha(score: pd.Series, ic: float, resid_vol_h: pd.Series) -> pd.Series:
    """alpha = IC x sigma x z.  Example (Spec §7.4 note): 0.04 x 25% x (-1.2) = -1.2%."""
    return (ic * resid_vol_h * score).rename("alpha")


def regime_factor_means(factor_returns: pd.DataFrame, regime: pd.Series, n0: float = 24.0) -> pd.DataFrame:
    """Shrunk regime means mu~_{k,s} (rows regime, columns factor), from periodic factor returns."""
    d = factor_returns.join(regime.rename("regime"), how="inner").dropna(subset=["regime"])
    fbar = d.drop(columns="regime").mean()
    out = {}
    for s, g in d.groupby("regime"):
        n = len(g)
        w = n / (n + n0)
        out[s] = w * g.drop(columns="regime").mean() + (1 - w) * fbar
    return pd.DataFrame(out).T


def expected_factor_returns(regime_means: pd.DataFrame, regime_probs: pd.Series | np.ndarray,
                            periods: float = 1.0, market_override: float | None = None) -> pd.Series:
    """E[f] = sum_s P(s) mu~_s, scaled to the forecast horizon in periods.

    market_override replaces the market factor with r_f + ERP over the horizon (Spec §2.7), which is
    better anchored than the historical regime mean of the market return.
    """
    p = pd.Series(np.asarray(regime_probs, float), index=regime_means.index)
    ef = (regime_means.mul(p, axis=0)).sum() * periods
    if market_override is not None and "market" in ef:
        ef["market"] = market_override
    return ef


def expected_factor_returns_path(regime_means: pd.DataFrame, probs_now: np.ndarray, transition: np.ndarray,
                                 periods: int, market_override: float | None = None) -> pd.Series:
    """E[f over the next `periods`] = sum_{k=1..periods} (pi_t T^k) mu~.

    The regime is not assumed to persist: the probabilities are propagated through the HMM transition
    matrix month by month, so a 12-month expectation drifts toward the long-run regime mix.
    """
    p = np.asarray(probs_now, float)
    total = pd.Series(0.0, index=regime_means.columns)
    for _ in range(periods):
        p = p @ np.asarray(transition)
        total += (regime_means.mul(pd.Series(p, index=regime_means.index), axis=0)).sum()
    if market_override is not None and "market" in total:
        total["market"] = market_override
    return total


def total_expected_return(exposures: pd.DataFrame, ef: pd.Series, alpha: pd.Series) -> pd.Series:
    """mu_i = beta_i^T E[f] + alpha_i (exposures: rows stocks, columns factors)."""
    common = [c for c in exposures.columns if c in ef.index]
    return (exposures[common] @ ef[common] + alpha.reindex(exposures.index).fillna(0.0)).rename("mu")
