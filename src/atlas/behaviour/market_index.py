"""Market sentiment index (Spec §5.3; Ref 3.7).

Proxies: VIX, put/call ratio, AAII bull–bear spread, margin-debt growth, IPO volume and average
first-day IPO return.

1. Each proxy is regressed on the growth and inflation factors and the residual is kept, which removes
   the rational, business-cycle part and leaves mood:
       proxy_{k,t} = a_k + b_{k,g} growth_t + b_{k,pi} inflation_t + u_{k,t}
2. The residuals are standardised with data available at the time.
3. PCA is refitted each month on past data only (expanding window): C v = lambda v.
4. Index_t = sum_k v_{1,k} z_{k,t}, oriented so that higher means more bullish.
5. Historical percentile -> regime: fear (< 10th), neutral, euphoria (> 90th).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Proxies that rise with fear; their loadings are expected to be negative in a bullish index.
FEAR_PROXIES = {"vix", "put_call"}


def orthogonalise(proxies: pd.DataFrame, macro: pd.DataFrame, min_obs: int = 24) -> pd.DataFrame:
    """Expanding-window residuals of each proxy on the macro factors (no look-ahead)."""
    out = pd.DataFrame(index=proxies.index, columns=proxies.columns, dtype=float)
    M = macro.reindex(proxies.index)
    for t in range(min_obs, len(proxies)):
        X = np.column_stack([np.ones(t + 1), M.iloc[: t + 1].values])
        ok = np.isfinite(X).all(axis=1)
        for c in proxies:
            y = proxies[c].iloc[: t + 1].values
            m = ok & np.isfinite(y)
            if m.sum() < min_obs:
                continue
            b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            out.iloc[t, out.columns.get_loc(c)] = y[-1] - X[-1] @ b if np.isfinite(y[-1]) else np.nan
    return out


def sentiment_index(adjusted: pd.DataFrame, min_obs: int = 36) -> pd.DataFrame:
    """Expanding-window PCA index with percentile and regime labels."""
    rows = []
    for t in range(min_obs, len(adjusted)):
        hist = adjusted.iloc[: t + 1].dropna()
        if len(hist) < min_obs:
            continue
        z = (hist - hist.mean()) / hist.std()
        C = np.corrcoef(z.values.T)
        evals, evecs = np.linalg.eigh(C)
        v1 = evecs[:, -1]
        # Orient so that fear proxies load negatively (higher index = more bullish).
        fear_load = sum(v1[i] for i, c in enumerate(z.columns) if c in FEAR_PROXIES)
        greed_load = sum(v1[i] for i, c in enumerate(z.columns) if c not in FEAR_PROXIES)
        if greed_load - fear_load < 0:
            v1 = -v1
        idx_series = z.values @ v1
        today = idx_series[-1]
        pct = float((idx_series <= today).mean())
        regime = "fear" if pct < 0.10 else ("euphoria" if pct > 0.90 else "neutral")
        rows.append({"date": adjusted.index[t], "sentiment_index": float(today), "percentile": pct,
                     "regime": regime, "pc1_variance_explained": float(evals[-1] / evals.sum())})
    return pd.DataFrame(rows)
