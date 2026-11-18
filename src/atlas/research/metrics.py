"""Metrics calculated per signal (Spec §9.2; Ref 3.11).

    IC_t         = Spearman correlation across stocks between the signal at t and returns over t -> t+h
    ICIR         = mean(IC) / sd(IC)
    IR           ~= IC x sqrt(breadth)                      (fundamental law of active management)
    Sharpe       = mean(r_LS) / sd(r_LS) x sqrt(12)          (monthly long–short returns)
    Turnover     = (1/2) sum_i |w_{i,t} - w_{i,t-1}|
    Max drawdown = max_t (1 - V_t / max_{s<=t} V_s)
    Hit rate     = share of months with r_LS > 0
    Decay        = IC at lags 1, 2, 3, 6 and 12 months

Example (Spec §9.2 note): IC = 0.03, breadth = 1,000 -> IR ~= 0.95.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ic_series(signal: pd.DataFrame, fwd_returns: pd.DataFrame, min_names: int = 20) -> pd.Series:
    """Cross-sectional Spearman IC per date (both inputs: date x entity)."""
    s, r = signal.align(fwd_returns, join="inner")
    out = {}
    for d in s.index:
        a, b = s.loc[d], r.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= min_names:
            out[d] = a[ok].rank().corr(b[ok].rank())
    return pd.Series(out, name="ic").sort_index()


def icir(ic: pd.Series) -> float:
    return float(ic.mean() / ic.std())


def fundamental_law(ic: float, breadth: float) -> float:
    return float(ic * np.sqrt(breadth))


def sharpe(r: pd.Series, periods_per_year: int = 12) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    v = (1 + r.fillna(0)).cumprod()
    return float((1 - v / v.cummax()).max())


def hit_rate(r: pd.Series) -> float:
    return float((r.dropna() > 0).mean())


def turnover(weights: pd.DataFrame) -> pd.Series:
    """weights: date x entity. One-way turnover per rebalance."""
    return 0.5 * weights.fillna(0).diff().abs().sum(axis=1).iloc[1:]


def ic_decay(signal: pd.DataFrame, monthly_returns: pd.DataFrame, lags=(1, 2, 3, 6, 12)) -> pd.Series:
    """Mean IC of the signal at t against the single-month return in month t + lag."""
    return pd.Series({lag: ic_series(signal, monthly_returns.shift(-lag)).mean() for lag in lags}, name="ic_decay")


def summary(ic: pd.Series, ls: pd.Series, periods_per_year: int = 12) -> dict:
    from atlas.research.inference import nw_tstat_mean
    m, se, t = nw_tstat_mean(ic.dropna().values)
    return {"ic_mean": float(ic.mean()), "ic_sd": float(ic.std()), "icir": icir(ic), "t_nw": t,
            "sharpe": sharpe(ls, periods_per_year), "max_drawdown": max_drawdown(ls), "hit_rate": hit_rate(ls),
            "n_periods": int(ic.notna().sum())}
