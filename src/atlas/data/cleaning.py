"""Cleaning (Spec §0.6; Ref 3.2).

All cleaning is cross-sectional (across companies on the same date) unless stated otherwise:

| Problem           | Method                               | Equation                                   |
|-------------------|--------------------------------------|--------------------------------------------|
| Outliers          | Robust z-score, clipped               | z = 0.6745 (x - median) / MAD, clip to ±3  |
| Outliers (alt.)   | Winsorise                             | cap at the 1st and 99th percentiles        |
| Missing values    | Sector-median impute + missing flag  | x_missing = median_sector, flag = 1        |
| Stale values      | Forward-fill with a maximum age      | <= 1 quarter fundamentals, <= 5 days prices |
| Mixed frequencies | Backward as-of join                  | see atlas.data.point_in_time.asof_join     |
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_zscore(x: pd.Series, clip: float = 3.0) -> pd.Series:
    """Modified z-score z = 0.6745 (x - median) / MAD, clipped to ±clip.

    The constant 0.6745 makes the score comparable with an ordinary z-score for normally distributed
    data. The median and MAD are not dragged by the outlier itself, unlike the mean and standard
    deviation. When MAD is zero (many identical values) the score is set to zero.
    """
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(0.0, index=x.index).where(x.notna())
    return (0.6745 * (x - med) / mad).clip(-clip, clip)


def winsorise(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Replace values beyond the lower/upper cross-sectional percentiles with those percentiles."""
    lo, hi = x.quantile(lower), x.quantile(upper)
    return x.clip(lo, hi)


def impute_sector_median(df: pd.DataFrame, col: str, sector_col: str = "sector") -> pd.DataFrame:
    """Fill missing values with the sector median and add a `<col>_missing` flag (= 1 if filled).

    The flag preserves the information that the value was missing, which can itself be informative
    (for example, troubled companies often report late).
    """
    out = df.copy()
    flag = out[col].isna().astype(int)
    out[col] = out[col].fillna(out.groupby(sector_col)[col].transform("median"))
    out[f"{col}_missing"] = flag
    return out


def forward_fill_limited(df: pd.DataFrame, cols: list[str], max_days: int, id_col: str = "entity_id",
                         date_col: str = "date") -> pd.DataFrame:
    """Forward-fill each company's values, but only for up to `max_days` calendar days.

    Beyond the maximum age the value is treated as missing, so a three-year-old debt figure is never
    treated as current (Spec §0.6 'Stale values').
    """
    out = df.sort_values([id_col, date_col]).copy()
    for c in cols:
        last_date = out[date_col].where(out[c].notna())
        last_date = last_date.groupby(out[id_col]).ffill()
        filled = out.groupby(id_col)[c].ffill()
        age = (out[date_col] - last_date).dt.days
        out[c] = filled.where(age <= max_days)
    return out.sort_index()


def clean_cross_section(df: pd.DataFrame, cols: list[str], date_col: str = "date", method: str = "robust",
                        clip: float = 3.0) -> pd.DataFrame:
    """Apply outlier treatment to each column, separately on each date.

    method="robust" uses the clipped robust z-score; method="winsor" winsorises at 1%/99%.
    """
    out = df.copy()
    for c in cols:
        if method == "robust":
            out[c] = out.groupby(date_col)[c].transform(lambda s: robust_zscore(s, clip))
        else:
            out[c] = out.groupby(date_col)[c].transform(winsorise)
    return out
