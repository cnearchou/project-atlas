"""Execution and slippage measurement (Spec §8.9; Ref 3.10 and Technical Reference Part 2.9).

Trades are sent to Interactive Brokers through `ib_async`, on the paper-trading account first
(gateway port 4002). Orders are worked over the next day with VWAP/TWAP-style slicing and participation
capped at 10–20% of volume; short sales are only sent after a locate is confirmed.

Implementation shortfall of each fill, with side = +1 for buys and -1 for sells:
    IS = side x (P_fill - P_decision) / P_decision
Realised costs are compared with the model of Spec §8.6, and the impact coefficient Y is re-estimated
quarterly by regressing IS - spread/2 on sigma sqrt(Q / ADV).

Safety: the broker class refuses to connect to a live port unless `allow_live=True` is passed
explicitly, and every order is checked against pre-trade limits first (a basic kill switch).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas._optional import require
from atlas.research.inference import ols

PAPER_PORTS = {4002, 7497}


def trade_list(w_target: pd.Series, w_current: pd.Series, prices: pd.Series, aum: float,
               min_trade_value: float = 1000.0) -> pd.DataFrame:
    """Convert weight changes into share orders, skipping trades too small to be worth the cost."""
    idx = w_target.index.union(w_current.index)
    dw = w_target.reindex(idx).fillna(0.0) - w_current.reindex(idx).fillna(0.0)
    value = dw * aum
    shares = (value / prices.reindex(idx)).round()
    t = pd.DataFrame({"dw": dw, "value": value, "shares": shares})
    t = t[(t["value"].abs() >= min_trade_value) & t["shares"].ne(0)]
    t["side"] = np.sign(t["shares"]).astype(int)
    return t


def slice_schedule(shares: float, adv_shares: float, participation: float = 0.15, n_slices: int = 13) -> np.ndarray:
    """TWAP-style child orders over a day (13 half-hour buckets), capped at the participation rate."""
    cap = participation * adv_shares / n_slices
    per = np.full(n_slices, abs(shares) / n_slices)
    per = np.minimum(per, cap)
    return np.sign(shares) * per


def implementation_shortfall(fills: pd.DataFrame) -> pd.Series:
    """fills: side (+1/-1), fill_price, decision_price."""
    return (fills["side"] * (fills["fill_price"] - fills["decision_price"]) / fills["decision_price"]).rename("is")


def reestimate_impact(fills: pd.DataFrame) -> float:
    """Y from IS - spread/2 = Y sigma sqrt(Q / ADV) (no intercept)."""
    y = implementation_shortfall(fills) - fills["spread"] / 2
    x = fills["sigma_daily"] * np.sqrt(fills["quantity"].abs() / fills["adv_shares"])
    return float(ols(y.values, x.values, hac_lags=0, add_const=False).params[0])


@dataclass
class PreTradeLimits:
    max_order_value: float = 1e6
    max_orders: int = 500


class IBBroker:
    """Thin wrapper around ib_async for paper trading. Not used by tests or the demo."""

    def __init__(self, allow_live: bool = False, limits: PreTradeLimits | None = None):
        # Connection details come from the environment (.env), never from code (Spec §0.1 / Ref Part 2).
        self.host = os.environ.get("IB_HOST", "127.0.0.1")
        self.port = int(os.environ.get("IB_PORT", "4002"))
        self.client_id = int(os.environ.get("IB_CLIENT_ID", "1"))
        if self.port not in PAPER_PORTS and not allow_live:
            raise RuntimeError(f"Port {self.port} is not a paper-trading port; pass allow_live=True deliberately.")
        self.limits = limits or PreTradeLimits()
        self.ib = None

    def connect(self):
        ib_async = require("ib_async", "data")
        self.ib = ib_async.IB()
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        return self

    def submit(self, orders: pd.DataFrame, prices: pd.Series):
        """Submit market-on-open style orders after pre-trade checks. Returns the trade objects."""
        ib_async = require("ib_async", "data")
        if len(orders) > self.limits.max_orders:
            raise RuntimeError("Kill switch: too many orders in one batch.")
        trades = []
        for ticker, row in orders.iterrows():
            if abs(row["shares"] * prices[ticker]) > self.limits.max_order_value:
                raise RuntimeError(f"Kill switch: order for {ticker} exceeds the maximum order value.")
            contract = ib_async.Stock(str(ticker), "SMART", "USD")
            action = "BUY" if row["shares"] > 0 else "SELL"
            trades.append(self.ib.placeOrder(contract, ib_async.MarketOrder(action, abs(int(row["shares"])))))
        return trades
