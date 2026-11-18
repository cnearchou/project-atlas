"""Training panel (Spec §7.1; Ref 3.9).

One row per (date, entity_id), one column per feature. The steps follow the specification:

1. Every feature is cleaned (Spec §0.6) and rank-Gaussianised across stocks on each date (Spec §0.7).
2. Missing values are set to 0, the neutral value after the rank-Gaussian transform, and a binary
   missing-indicator column `<feature>_missing` is added so the model knows the value was imputed.
3. The target is the forward residual return over horizon h,
       y_{i,t} = sum_{d=t+1}^{t+h} u_{i,d},
   where u is the factor-model residual of Spec §6.3. It starts the day *after* the feature date, so a
   feature known at the close of day t never overlaps its own target. The target is then
   rank-Gaussianised across stocks on each date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.data.standardise import rank_gauss


def forward_residual_target(residuals: pd.DataFrame, h: int) -> pd.DataFrame:
    """date x entity matrix of y_{i,t} = sum of residuals on days t+1 ... t+h.

    The rolling sum ending at t+h is shifted back by h days. Rows near the end of the sample, whose
    window is not complete yet, are NaN and therefore excluded from training.
    """
    return residuals.rolling(h, min_periods=h).sum().shift(-h)


def build_panel(features: pd.DataFrame, target_wide: pd.DataFrame | None, feature_cols: list[str],
                add_missing_indicators: bool = True) -> pd.DataFrame:
    """Assemble the modelling panel.

    features: long table with date, entity_id and raw feature columns (already point-in-time).
    target_wide: date x entity forward residual returns, or None at prediction time.
    """
    df = features[["date", "entity_id", *feature_cols]].copy()
    for c in feature_cols:
        if add_missing_indicators:
            df[f"{c}_missing"] = df[c].isna().astype(float)
        df[c] = df.groupby("date")[c].transform(rank_gauss).fillna(0.0)
    if target_wide is not None:
        tgt = target_wide.stack().rename("target_raw").reset_index()
        tgt.columns = ["date", "entity_id", "target_raw"]
        df = df.merge(tgt, on=["date", "entity_id"], how="left")
        df["target"] = df.groupby("date")["target_raw"].transform(rank_gauss)
    return df.sort_values(["date", "entity_id"]).reset_index(drop=True)


def model_columns(panel: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Features plus any missing-indicator columns that carry information (non-constant)."""
    extra = [f"{c}_missing" for c in feature_cols if f"{c}_missing" in panel and panel[f"{c}_missing"].std() > 0]
    return [*feature_cols, *extra]


def sample_dates(panel: pd.DataFrame, every: int = 21) -> pd.DataFrame:
    """Keep one date in every `every` trading days (e.g. month ends) to reduce overlap and size."""
    dates = np.sort(panel["date"].unique())
    keep = set(dates[::every])
    return panel[panel["date"].isin(keep)].reset_index(drop=True)
