"""The [risk] output table (Spec §8.8).

Columns that need unavailable data (implied volatility, borrow, earnings-implied moves) stay NaN so
the schema is fixed. `max_position` combines every limit: the mandate cap (3%), the fractional Kelly
bound and the liquidity cap of at most `max_days` days to exit at the participation rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_COLUMNS = ["id", "date", "sigma_garch", "sigma_implied", "var95_fhs", "es95_fhs", "var99_evt", "es99_evt",
                "factor_risk_pct", "idio_risk_pct", "days_to_liquidate", "lvar", "earnings_implied_move",
                "altman_z", "distance_to_default", "squeeze_score", "crowding",
                "stress_2008", "stress_2020", "stress_2022", "max_position"]


def max_position(kelly: pd.Series, adv_dollars: pd.Series, aum: float, mandate_cap: float = 0.03,
                 participation: float = 0.20, max_days: float = 3.0) -> pd.Series:
    liq = participation * adv_dollars * max_days / aum
    return pd.concat([kelly.abs(), liq.reindex(kelly.index), pd.Series(mandate_cap, index=kelly.index)], axis=1).min(axis=1)


def assemble_risk(parts: list[pd.DataFrame], date) -> pd.DataFrame:
    df = pd.concat(parts, axis=1)
    df.insert(0, "id", df.index)
    df["date"] = pd.Timestamp(date)
    for c in RISK_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[RISK_COLUMNS].reset_index(drop=True)
