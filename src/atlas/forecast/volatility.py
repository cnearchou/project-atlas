"""Volatility forecast (Spec §7.5; Ref 3.9).

    sigma_blend = w sigma_implied + (1 - w) sigma_GARCH
w is estimated by regressing realised volatility on the two forecasts (constrained to [0, 1]).
Horizon scaling with the square-root-of-time rule:
    sigma_h = sigma_daily x sqrt(h)        (h in trading days; 252 a year)
When implied volatility is unavailable (no OptionMetrics licence), the GARCH forecast is used alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def scale_vol(sigma_daily, h: int):
    return np.asarray(sigma_daily) * np.sqrt(h)


def fit_blend_weight(realised, sigma_implied, sigma_garch) -> float:
    """w minimising the squared error of realised vol against the blend (closed-form, then clipped)."""
    r, a, b = (np.asarray(x, float) for x in (realised, sigma_implied, sigma_garch))
    ok = np.isfinite(r) & np.isfinite(a) & np.isfinite(b)
    r, a, b = r[ok], a[ok], b[ok]
    if len(r) < 10:
        return 0.5
    res = minimize_scalar(lambda w: np.mean((r - (w * a + (1 - w) * b)) ** 2), bounds=(0, 1), method="bounded")
    return float(res.x)


def blend(sigma_implied: pd.Series | None, sigma_garch: pd.Series, w: float) -> pd.Series:
    if sigma_implied is None:
        return sigma_garch.rename("sigma")
    s = w * sigma_implied.reindex(sigma_garch.index) + (1 - w) * sigma_garch
    return s.fillna(sigma_garch).rename("sigma")


def ewma_vol(returns: pd.DataFrame, half_life: float = 30) -> pd.Series:
    """Simple EWMA daily volatility per stock; used as a fast fallback when GARCH fails to converge."""
    return np.sqrt((returns ** 2).ewm(halflife=half_life).mean().iloc[-1])
