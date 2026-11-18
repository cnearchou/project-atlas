"""Standardisation and neutralisation (Spec §0.7; Ref 3.2).

Cross-sectional rank-to-Gaussian transform:

    z_i = Phi^{-1}((rank_i - 0.5) / N)

makes every signal robust, bounded and comparable regardless of its original units.

Neutralisation removes unwanted sector and size effects from a signal s by keeping the OLS residual:

    s_perp = s - X (X^T X)^{-1} X^T s,   X = [sector dummies, ln(market cap)]

Every signal in Sections 1–5 passes through Spec §0.6 then §0.7 before entering the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def rank_gauss(x: pd.Series) -> pd.Series:
    """Rank-to-Gaussian transform of one cross-section. Missing values stay missing.

    Worked example (Spec §0.7 note): five values ranked 1..5 map to percentiles 0.1, 0.3, 0.5, 0.7,
    0.9 and then to z-scores -1.28, -0.52, 0, +0.52, +1.28.
    """
    r = x.rank(method="average")
    n = x.notna().sum()
    if n == 0:
        return x * np.nan
    return pd.Series(norm.ppf((r - 0.5) / n), index=x.index)


def neutralise(s: pd.Series, sectors: pd.Series, log_mcap: pd.Series | None = None) -> pd.Series:
    """Residual of an OLS regression of the signal on sector dummies and ln(market cap).

    The projection X (X^T X)^{-1} X^T s is the part of the signal explained by sector and size; the
    residual s_perp is the stock-specific part. `numpy.linalg.lstsq` is used instead of an explicit
    inverse for numerical stability.
    """
    ok = s.notna() & sectors.notna()
    if log_mcap is not None:
        ok &= log_mcap.notna()
    X = pd.get_dummies(sectors[ok], dtype=float)
    if log_mcap is not None:
        X["log_mcap"] = log_mcap[ok]
    y = s[ok].values
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    resid = pd.Series(np.nan, index=s.index)
    resid[ok] = y - X.values @ beta
    return resid


def standardise_panel(df: pd.DataFrame, cols: list[str], date_col: str = "date", sector_col: str | None = "sector",
                      log_mcap_col: str | None = "log_mcap") -> pd.DataFrame:
    """Rank-Gaussianise and then neutralise every signal column, date by date."""
    out = df.copy()
    for c in cols:
        out[c] = out.groupby(date_col)[c].transform(rank_gauss)
        if sector_col and sector_col in out:
            parts = []
            for _, g in out.groupby(date_col):
                lm = g[log_mcap_col] if log_mcap_col and log_mcap_col in g else None
                parts.append(neutralise(g[c], g[sector_col], lm))
            out[c] = pd.concat(parts).reindex(out.index)
    return out
