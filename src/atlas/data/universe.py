"""Universe construction without survivorship bias (Spec §0.5).

On each date a stock is eligible if price > $5, market cap > $500M and 20-day average daily dollar
volume (ADV) > $5M, all evaluated on that date using only past data. Delisted companies stay in the
history with their delisting return (often -30% to -100%); dropping them would make every strategy
look better than it is.
"""

from __future__ import annotations

import pandas as pd


def eligible_universe(panel: pd.DataFrame, min_price: float = 5.0, min_market_cap: float = 5e8,
                      min_adv_dollars: float = 5e6, adv_window: int = 20) -> pd.Series:
    """Boolean eligibility flag for every (entity_id, date) row.

    The panel needs columns: entity_id, date, price, shares_outstanding, volume. ADV is the mean of
    price x volume over the previous `adv_window` days, lagged one day so the filter uses only
    information available before the trading decision.
    """
    df = panel.sort_values(["entity_id", "date"]).copy()
    df["market_cap"] = df["price"].abs() * df["shares_outstanding"]
    dollar_volume = df["price"].abs() * df["volume"]
    df["adv_dollars"] = (dollar_volume.groupby(df["entity_id"])
                         .transform(lambda s: s.rolling(adv_window, min_periods=adv_window).mean().shift(1)))
    ok = (df["price"].abs() > min_price) & (df["market_cap"] > min_market_cap) & (df["adv_dollars"] > min_adv_dollars)
    return ok.reindex(panel.index).fillna(False)


def apply_delisting_returns(returns: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    """Merge delisting returns into the return panel (Spec §0.5).

    `delistings` has columns entity_id, date, delisting_return. On the delisting date the total
    return is compounded with the delisting return: (1 + r) * (1 + r_delist) - 1. When a delisting
    return is missing (common for bankruptcies in CRSP), -30% is used, a standard conservative
    assumption in the literature.
    """
    dl = delistings[["entity_id", "date", "delisting_return"]].copy()
    dl["delisting_return"] = dl["delisting_return"].fillna(-0.30)  # missing -> conservative -30%
    out = returns.merge(dl, on=["entity_id", "date"], how="left")
    out["ret"] = (1 + out["ret"].fillna(0.0)) * (1 + out["delisting_return"].fillna(0.0)) - 1
    return out.drop(columns="delisting_return")
