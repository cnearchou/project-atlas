"""The [forecasts] output table (Spec §7.9).

Columns: id, date, horizon, score, alpha, mu, sigma, q10, q50, q90, p_bull, p_base, p_bear,
p_bull_cal, p_base_cal, p_bear_cal, p_bull_rel, p_base_rel, p_bear_rel, top_5_shap_drivers.
Columns that a run cannot produce (for example calibrated probabilities before enough forecasts have
resolved) are left as NaN, so the schema is always the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST_COLUMNS = ["id", "date", "horizon", "score", "alpha", "mu", "sigma", "q10", "q50", "q90",
                    "p_bull", "p_base", "p_bear", "p_bull_cal", "p_base_cal", "p_bear_cal",
                    "p_bull_rel", "p_base_rel", "p_bear_rel", "top_5_shap_drivers"]


def assemble_forecasts(parts: list[pd.DataFrame], date, horizon: str) -> pd.DataFrame:
    """Join per-stock pieces (all indexed by entity id) into the fixed output schema."""
    df = pd.concat(parts, axis=1)
    df.insert(0, "id", df.index)
    df["date"] = pd.Timestamp(date)
    df["horizon"] = horizon
    for c in FORECAST_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[FORECAST_COLUMNS].reset_index(drop=True)
