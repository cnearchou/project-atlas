"""Stock-level behavioural signals (Spec §5.2; Ref 3.7).

Momentum (three versions compared in testing):
    Raw          MOM_t    = sum_{k=2..12} ln(1 + r_{t-k})       (skips the latest month)
    Vol-scaled   MOM_vs   = MOM / sigma_12m
    Residual     MOM_res  = sum_{k=2..12} eps_{t-k}               (factor-model residuals, Spec §6.3)
Short-term reversal:
    REV_t = - sum_{d=1..5} eps_{t-d} / sigma_daily
Analyst revisions (three months):
    Revision magnitude = sum ΔConsensus EPS / Price
    Revision breadth   = (n_up - n_down) / n_estimates
Social attention:
    a_t = ln(1 + count_t),  attention_z = (a_t - rolling 60-day mean) / rolling 60-day std
52-week-high ratio: price / highest price of the past 252 trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas._compat import MONTH_END


def monthly_returns(daily: pd.DataFrame, ret_col: str = "ret") -> pd.DataFrame:
    """Compound daily returns into calendar-month returns (wide: month x entity)."""
    wide = daily.pivot_table(index="date", columns="entity_id", values=ret_col)
    return (1 + wide).resample(MONTH_END).prod(min_count=1) - 1


def momentum_raw(monthly: pd.DataFrame) -> pd.DataFrame:
    """Momentum known at the end of month t and used to forecast month t+1.

    In the specification's notation the forecast month is t+1, so MOM = sum_{k=2..12} ln(1 + r_{t+1-k})
    uses months t-11 to t-1 and skips the most recent month t (where short-term reversal operates).
    """
    lr = np.log1p(monthly)
    return lr.shift(1).rolling(11, min_periods=9).sum()


def momentum_vol_scaled(monthly: pd.DataFrame) -> pd.DataFrame:
    sigma = monthly.rolling(12, min_periods=9).std()
    return momentum_raw(monthly) / sigma


def momentum_residual(monthly_resid: pd.DataFrame) -> pd.DataFrame:
    """Same window applied to monthly factor-model residual returns."""
    return monthly_resid.shift(1).rolling(11, min_periods=9).sum()


def short_term_reversal(daily_resid: pd.DataFrame, window: int = 5, vol_window: int = 63) -> pd.DataFrame:
    """Negative sum of the last five daily residuals, scaled by residual volatility (wide frames)."""
    s = daily_resid.rolling(window).sum()
    vol = daily_resid.rolling(vol_window, min_periods=20).std()
    return -s / vol


def high_52w_ratio(prices_wide: pd.DataFrame) -> pd.DataFrame:
    return prices_wide / prices_wide.rolling(252, min_periods=126).max()


def revision_signals(est: pd.DataFrame, price: pd.Series | None = None) -> pd.DataFrame:
    """From monthly consensus rows (entity_id, date, meanest, numup, numdown, numest).

    Magnitude sums the last three monthly changes in consensus EPS, scaled by price; breadth uses the
    up/down revision counts over the same window.
    """
    e = est.sort_values(["entity_id", "date"]).copy()
    g = e.groupby("entity_id")
    d_cons = g["meanest"].diff()
    e["revision_magnitude"] = d_cons.groupby(e["entity_id"]).transform(lambda s: s.rolling(3, min_periods=1).sum())
    if price is not None:
        e["revision_magnitude"] = e["revision_magnitude"] / price.reindex(e.index)
    up = g["numup"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    down = g["numdown"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    e["revision_breadth"] = (up - down) / e["numest"].where(e["numest"] > 0)
    return e[["entity_id", "date", "revision_magnitude", "revision_breadth"]]


def attention_z(counts: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Daily mention counts (date x ticker, after bot filtering) -> standardised attention.

    The rolling statistics use the previous `window` days only (shifted by one), so today's spike is
    compared with a baseline that does not include it.
    """
    a = np.log1p(counts)
    mu = a.shift(1).rolling(window, min_periods=20).mean()
    sd = a.shift(1).rolling(window, min_periods=20).std()
    return (a - mu) / sd


def weighted_social_sentiment(posts: pd.DataFrame) -> pd.DataFrame:
    """Daily average post sentiment per ticker, each author weighted by ln(1 + followers)."""
    p = posts.assign(w=np.log1p(posts["followers"]))
    p["ws"] = p["w"] * p["sentiment"]
    g = p.groupby(["ticker", "date"])
    return (g["ws"].sum() / g["w"].sum()).rename("social_sentiment").reset_index()
