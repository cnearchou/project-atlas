"""Live monitoring (Spec §9.5-§9.6).

Rolling 12-month IC for every signal.
CUSUM on the IC (one-sided, detecting deterioration below the expected IC mu_0):
    S_t = max(0, S_{t-1} + (mu_0 - IC_t) - k),  alarm when S_t > h
with tolerance k (typically half the shift to detect) and threshold h (in IC units).
Calibration drift: ECE (Spec §7.7) recomputed monthly.
Population Stability Index for feature drift:
    PSI = sum_b (A_b - E_b) ln(A_b / E_b)
where E_b are training-period bucket shares (deciles of the training distribution) and A_b the recent
shares. PSI > 0.25 indicates a material change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ic(ic: pd.Series, window: int = 12) -> pd.Series:
    return ic.rolling(window, min_periods=max(3, window // 2)).mean()


def cusum(ic: pd.Series, mu0: float, k: float = 0.01, h: float = 0.10) -> pd.DataFrame:
    S, out = 0.0, []
    for v in ic.values:
        S = max(0.0, S + (mu0 - v) - k) if np.isfinite(v) else S
        out.append(S)
    s = pd.Series(out, index=ic.index, name="cusum")
    return pd.DataFrame({"cusum": s, "alarm": s > h})


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-4) -> float:
    e = np.asarray(expected, float); a = np.asarray(actual, float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    E = np.histogram(e, edges)[0] / len(e)
    A = np.histogram(a, edges)[0] / len(a)
    E, A = np.clip(E, eps, None), np.clip(A, eps, None)
    return float(np.sum((A - E) * np.log(A / E)))


def status(ic_12m: float, alarm: bool, psi_max: float, ece_now: float, psi_alarm: float = 0.25,
           ece_alarm: float = 0.05) -> str:
    if alarm and ic_12m <= 0:
        return "retired"
    if alarm or psi_max > psi_alarm or ece_now > ece_alarm:
        return "on watch"
    return "live"


def monitoring_row(signal: str, date, ic: pd.Series, mu0: float, train_feature: dict, recent_feature: dict,
                   ece_now: float = float("nan")) -> dict:
    r = rolling_ic(ic).iloc[-1]
    cs = cusum(ic, mu0).iloc[-1]
    psi_max = max(psi(train_feature[f], recent_feature[f]) for f in train_feature) if train_feature else 0.0
    return {"signal": signal, "date": pd.Timestamp(date), "ic_12m": float(r), "cusum": float(cs["cusum"]),
            "alarm": bool(cs["alarm"]), "ece": ece_now, "psi_max": psi_max,
            "status": status(float(r), bool(cs["alarm"]), psi_max, ece_now if np.isfinite(ece_now) else 0.0)}
