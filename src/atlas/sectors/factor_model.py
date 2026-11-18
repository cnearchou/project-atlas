"""Factor model (Spec §6.3; Ref 3.8).

Exposures X_t (recomputed daily): market = 1, sector dummies, and standardised styles (size =
z(ln mcap), value = z(B/P), momentum = z(MOM_res), volatility = z(252-day vol), quality, growth).

Daily factor returns by weighted least squares:
    r_t = X_t f_t + u_t,   f_hat_t = (X^T W X)^{-1} X^T W r_t,   W = diag(sqrt(mcap))
with the identifying constraint sum_s (sector cap weight_s) f_sector,s = 0, because the market and the
full set of sector dummies are collinear. The constraint is imposed by reparameterisation: the last
sector's return is written as minus the cap-weighted sum of the others.

Factor covariance by EWMA with a 90-day half-life:
    F_t = lambda F_{t-1} + (1 - lambda) f_t f_t^T,   lambda = 0.5^(1/90)
optionally scaled so that recent realised factor volatility matches predicted volatility.

Specific risk: d^2_{i,t} = EWMA of u^2_{i,t}, shrunk toward the average of the stock's size bucket.
Total covariance: Sigma = B F B^T + D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas.data.standardise import rank_gauss

STYLE_COLS = ["size", "value", "momentum", "volatility", "quality", "growth"]


def build_exposures(cs: pd.DataFrame, sectors: list[str], styles: list[str] = STYLE_COLS) -> pd.DataFrame:
    """Exposure matrix for one date. `cs` holds one row per stock with sector and raw style columns.

    Style exposures are rank-Gaussianised across stocks (Spec §0.7), so each style has mean about 0
    and unit spread on every date.
    """
    X = pd.DataFrame(index=cs.index)
    X["market"] = 1.0
    for s in sectors:
        X[f"sector_{s}"] = (cs["sector"] == s).astype(float)
    for st in styles:
        if st in cs:
            X[st] = rank_gauss(cs[st]).fillna(0.0)
    return X


def constrained_wls(r: np.ndarray, X: pd.DataFrame, weights: np.ndarray, sector_cols: list[str],
                    sector_caps: np.ndarray) -> tuple[pd.Series, np.ndarray]:
    """WLS with the cap-weighted zero-sum sector constraint, solved by reparameterisation.

    With sector cap weights c_s (summing to 1), f_last = -sum_{s<last} (c_s / c_last) f_s. Substituting
    this removes one sector column and leaves an ordinary WLS problem.
    """
    cols = list(X.columns)
    last = sector_cols[-1]
    c = sector_caps / sector_caps.sum()
    X2 = X.copy()
    for s, cs_ in zip(sector_cols[:-1], c[:-1]):
        X2[s] = X[s] - (cs_ / c[-1]) * X[last]
    X2 = X2.drop(columns=last)
    sw = np.sqrt(weights)
    A = X2.values * sw[:, None]
    b = r * sw
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    f = pd.Series(coef, index=X2.columns)
    f[last] = -sum((cs_ / c[-1]) * f[s] for s, cs_ in zip(sector_cols[:-1], c[:-1]))
    f = f[cols]
    resid = r - X.values @ f.values
    return f, resid


@dataclass
class FactorModelResult:
    factor_returns: pd.DataFrame     # date x factor
    residuals: pd.DataFrame          # date x entity (specific returns u_{i,t})
    exposures: dict                  # date -> exposure DataFrame


def estimate_factor_returns(panel: pd.DataFrame, sectors: list[str], styles: list[str] = STYLE_COLS,
                            ret_col: str = "ret") -> FactorModelResult:
    """Run the daily cross-sectional regression over a panel.

    `panel` columns: date, entity_id, ret, sector, market_cap and the raw style columns. Exposures
    must be measured before the return they explain (the caller lags them by one day).
    """
    f_rows, resid_rows, expos = {}, {}, {}
    for d, g in panel.groupby("date"):
        g = g.dropna(subset=[ret_col, "market_cap"]).set_index("entity_id")
        if len(g) < len(sectors) + len(styles) + 10:
            continue
        X = build_exposures(g, sectors, styles)
        sector_cols = [c for c in X.columns if c.startswith("sector_") and X[c].sum() > 0]
        X = X[["market", *sector_cols, *[s for s in styles if s in X]]]
        caps = np.array([g.loc[X[s] == 1, "market_cap"].sum() for s in sector_cols])
        f, u = constrained_wls(g[ret_col].values, X, np.sqrt(g["market_cap"].values), sector_cols, caps)
        f_rows[d] = f
        resid_rows[d] = pd.Series(u, index=g.index)
        expos[d] = X
    return FactorModelResult(pd.DataFrame(f_rows).T.sort_index(), pd.DataFrame(resid_rows).T.sort_index(), expos)


def ewma_covariance(f: pd.DataFrame, half_life: float = 90.0, newey_west_lags: int = 2,
                    vol_regime_window: int = 21) -> np.ndarray:
    """EWMA factor covariance with a Newey–West correction and a volatility-regime scale.

    * EWMA with lambda = 0.5^(1/half_life).
    * Newey–West: lagged cross-covariances are added so autocorrelated daily factor returns are not
      under-estimated when scaled to longer horizons.
    * Volatility regime: the matrix is rescaled by the ratio of recent realised to predicted
      volatility of the factor returns (averaged across factors).
    """
    X = f.fillna(0.0).values
    T, k = X.shape
    lam = 0.5 ** (1 / half_life)
    w = lam ** np.arange(T - 1, -1, -1)
    w = w / w.sum()
    F = (X * w[:, None]).T @ X
    for l in range(1, newey_west_lags + 1):
        G = (X[l:] * w[l:, None]).T @ X[:-l]
        F += (1 - l / (newey_west_lags + 1)) * (G + G.T)
    if T > vol_regime_window:
        recent = X[-vol_regime_window:]
        pred_sd = np.sqrt(np.clip(np.diag(F), 1e-12, None))
        ratio = np.nanmean(recent.std(axis=0) / pred_sd)
        if np.isfinite(ratio) and ratio > 0:
            F *= ratio ** 2
    # Symmetrise and make positive semi-definite.
    F = (F + F.T) / 2
    ev, V = np.linalg.eigh(F)
    return V @ np.diag(np.clip(ev, 1e-12, None)) @ V.T


def specific_variance(resid: pd.DataFrame, half_life: float = 90.0, size_bucket: pd.Series | None = None,
                      shrink: float = 0.3) -> pd.Series:
    """d^2_i: EWMA of squared residuals, shrunk toward the size-bucket average (weight `shrink`)."""
    lam = 0.5 ** (1 / half_life)
    R = resid.fillna(0.0) ** 2
    w = lam ** np.arange(len(R) - 1, -1, -1)
    d2 = pd.Series((R.values * (w / w.sum())[:, None]).sum(axis=0), index=resid.columns)
    if size_bucket is not None:
        bucket_mean = d2.groupby(size_bucket.reindex(d2.index)).transform("mean")
        d2 = (1 - shrink) * d2 + shrink * bucket_mean.fillna(d2)
    return d2


def total_covariance(B: pd.DataFrame, F: np.ndarray, d2: pd.Series) -> pd.DataFrame:
    """Sigma = B F B^T + D for the stocks in B (rows) using factor covariance F (columns of B)."""
    S = B.values @ F @ B.values.T + np.diag(d2.reindex(B.index).fillna(d2.mean()).values)
    return pd.DataFrame(S, index=B.index, columns=B.index)


def ledoit_wolf(F_sample: np.ndarray, returns: np.ndarray | None = None) -> np.ndarray:
    """Ledoit–Wolf shrinkage toward a scaled identity (scikit-learn), when returns are supplied."""
    if returns is None:
        return F_sample
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(returns).covariance_
