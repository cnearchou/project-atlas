"""Sector aggregate indexes (Spec §6.5).

Sector-level indicators are computed from company-level estimates:
* fundamentals: market-cap-weighted averages and medians for growth, margins and revisions;
* revision breadth: percentage of companies with upward EPS revisions;
* flows: cap-weighted constituent flow pressure;
* sentiment: cap-weighted social sentiment combined with the sector news Z-score.

Missing company values are excluded and the cap weights are re-normalised over the companies that
have data, as the specification requires this choice to be stated explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cap_weighted(df: pd.DataFrame, col: str, by=("date", "sector"), cap_col: str = "market_cap") -> pd.Series:
    """Cap-weighted mean with weights re-normalised over non-missing values."""
    d = df.dropna(subset=[col, cap_col])
    num = (d[col] * d[cap_col]).groupby([d[b] for b in by]).sum()
    den = d[cap_col].groupby([d[b] for b in by]).sum()
    return num / den.replace(0, np.nan)


def sector_indexes(df: pd.DataFrame, news_z: pd.Series | None = None) -> pd.DataFrame:
    """Build the sector_indexes table. Available input columns are used; others are skipped."""
    by = ["date", "sector"]
    out = {}
    for c in ("growth", "margin", "revision_magnitude"):
        if c in df:
            out[f"{c}_capw"] = cap_weighted(df, c)
            out[f"{c}_median"] = df.groupby(by)[c].median()
    if "revision_breadth" in df:
        out["breadth"] = df.assign(up=(df["revision_breadth"] > 0).astype(float)).groupby(by)["up"].mean()
    if "flow_pressure" in df:
        out["flows"] = cap_weighted(df, "flow_pressure")
    if "social_sentiment" in df:
        out["sentiment"] = cap_weighted(df, "social_sentiment")
    res = pd.DataFrame(out)
    if news_z is not None:
        res["news_z"] = news_z.reindex(res.index)
    return res.reset_index()
