"""Corporate-action adjustment (Spec §0.3; Ref 3.2).

Splits and dividends distort raw prices. This module builds total-return series and backward-
adjusted prices:

    Total return:     r_t = (P_t + D_t) / P_{t-1} - 1
    Split factor:     a_split = 1 / split ratio        (2-for-1 split -> 0.5)
    Dividend factor:  a_div = 1 - D_t / P_{t-1}
    Adjusted price:   P_adj,t = P_t * prod_{s > t} a_s

Adjustments are applied backward because later corporate actions change how earlier prices should be
represented for comparison. Every model in the system is fed adjusted prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_returns(prices: pd.DataFrame, price_col: str = "price", div_col: str = "dividend",
                  split_col: str = "split_ratio", id_col: str = "entity_id") -> pd.Series:
    """Total return per row, grouped by company and sorted by date.

    The previous price is restated for any split on day t (a 2-for-1 split halves the previous
    price), so a split does not appear as a fake -50% return.
    """
    df = prices.sort_values([id_col, "date"])
    prev = df.groupby(id_col)[price_col].shift(1)
    split = df[split_col].fillna(1.0) if split_col in df else 1.0
    div = df[div_col].fillna(0.0) if div_col in df else 0.0
    # On a split day the previous price is divided by the split ratio before computing the return.
    prev_adj = prev / split
    r = (df[price_col] + div) / prev_adj - 1.0
    return r.reindex(prices.index)


def adjustment_factors(df: pd.DataFrame, price_col: str = "price", div_col: str = "dividend",
                       split_col: str = "split_ratio") -> pd.Series:
    """Per-row adjustment factor a_t for one company (rows sorted by date).

    The factor on day t applies to all prices strictly before t:
        a_t = (1 / split ratio on t) * (1 - D_t / P_{t-1})
    """
    prev_price = df[price_col].shift(1)
    split = df[split_col].fillna(1.0) if split_col in df else pd.Series(1.0, index=df.index)
    div = df[div_col].fillna(0.0) if div_col in df else pd.Series(0.0, index=df.index)
    a_split = 1.0 / split
    # The dividend factor uses the previous close restated for any same-day split.
    a_div = 1.0 - div / (prev_price / split)
    return (a_split * a_div.fillna(1.0)).fillna(1.0)


def backward_adjusted_prices(prices: pd.DataFrame, price_col: str = "price", id_col: str = "entity_id",
                             **kwargs) -> pd.Series:
    """Backward-adjusted price P_adj,t = P_t * prod_{s>t} a_s for every company.

    The product over later events is a reverse cumulative product of the factors, shifted by one day
    so that the factor on day s only affects days before s.
    """
    out = pd.Series(np.nan, index=prices.index, dtype=float)
    for _, g in prices.sort_values([id_col, "date"]).groupby(id_col):
        a = adjustment_factors(g, price_col=price_col, **kwargs)
        later = a[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        out.loc[g.index] = g[price_col].values * later.values
    return out
