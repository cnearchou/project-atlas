"""Statement processing (Spec §1.2; Ref 3.3).

Step 1  Tag mapping (see atlas.ingest.edgar.TAG_MAP).
Step 2  Q4 = Annual - (Q1 + Q2 + Q3), because 10-Ks report full years only.
Step 3  Trailing twelve months: flow items are summed over the last four quarters; stock items use
        the latest value or a five-quarter average for ratio denominators.
Step 4  Growth rates with care for negative and small bases:
            YoY growth  g = (X_t - X_{t-4}) / |X_{t-4}|
            Log growth  g = ln(X_t / X_{t-4})   when both values are positive
            |X_{t-4}| < 1% of revenue -> missing (unstable)
Step 5  Ratios: NOPAT, invested capital, ROIC, gross profitability, accruals, EV, FCF yield.

All functions work on a quarterly panel with one row per (entity_id, event_date) and the
knowledge_date of the latest field used, so the output stays point-in-time (Spec §0.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLOW_FIELDS = ["revenue", "ebit", "net_income", "operating_cash_flow", "gross_profit", "capex", "depreciation", "sga"]
STOCK_FIELDS = ["total_assets", "total_debt", "equity", "cash", "receivables", "current_assets",
                "current_liabilities", "retained_earnings", "total_liabilities", "shares"]


def pit_to_quarterly(pit: pd.DataFrame) -> pd.DataFrame:
    """Pivot point-in-time rows into a quarterly panel.

    For every (entity, quarter) the latest version of each field is kept, and the panel's
    knowledge_date is the latest filing date of the fields used, so the row only becomes usable once
    all of its inputs are public.
    """
    latest = pit.sort_values("knowledge_date").groupby(["entity_id", "event_date", "field"], as_index=False).last()
    wide = latest.pivot_table(index=["entity_id", "event_date"], columns="field", values="value")
    known = latest.groupby(["entity_id", "event_date"])["knowledge_date"].max()
    wide["knowledge_date"] = known
    return wide.reset_index().sort_values(["entity_id", "event_date"])


def derive_q4(quarters: pd.DataFrame, annual: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Derive Q4 values from annual totals: Q4 = Annual - (Q1 + Q2 + Q3) (Spec §1.2, Step 2).

    `quarters` holds Q1–Q3 rows with a `fiscal_year` column; `annual` holds one row per fiscal
    year. The derived Q4 row inherits the annual report's knowledge date.
    """
    q_sum = quarters.groupby(["entity_id", "fiscal_year"])[fields].sum(min_count=3)
    a = annual.set_index(["entity_id", "fiscal_year"])
    q4 = a[fields] - q_sum.reindex(a.index)
    q4["event_date"] = a["event_date"]
    q4["knowledge_date"] = a["knowledge_date"]
    return q4.reset_index()


def ttm(panel: pd.DataFrame, fields: list[str] | None = None) -> pd.DataFrame:
    """Trailing-twelve-month sums for flow items and five-quarter averages for stock items."""
    out = panel.sort_values(["entity_id", "event_date"]).copy()
    g = out.groupby("entity_id")
    for f in fields or [c for c in FLOW_FIELDS if c in out]:
        out[f"{f}_ttm"] = g[f].transform(lambda s: s.rolling(4, min_periods=4).sum())
    for f in [c for c in STOCK_FIELDS if c in out]:
        out[f"{f}_avg5"] = g[f].transform(lambda s: s.rolling(5, min_periods=2).mean())
    return out


def yoy_growth(panel: pd.DataFrame, field: str, revenue_field: str = "revenue") -> pd.Series:
    """Year-on-year growth with an absolute-value denominator and a small-base filter.

    Dividing by |X_{t-4}| keeps the sign meaningful when last year's value was negative. Values whose
    base is below 1% of revenue are set to missing, because growth from a tiny base is meaningless.
    """
    s = panel.groupby("entity_id")[field].shift(4)
    g = (panel[field] - s) / s.abs()
    base_small = s.abs() < 0.01 * panel[revenue_field].abs()
    return g.mask(base_small)


def log_growth(panel: pd.DataFrame, field: str) -> pd.Series:
    """ln(X_t / X_{t-4}), defined only when both values are positive."""
    s = panel.groupby("entity_id")[field].shift(4)
    ok = (panel[field] > 0) & (s > 0)
    return pd.Series(np.where(ok, np.log(panel[field].where(ok) / s.where(ok)), np.nan), index=panel.index)


def ratios(panel: pd.DataFrame, market_cap: pd.Series | None = None, tax_rate: float = 0.21) -> pd.DataFrame:
    """Profitability, quality and value ratios of Spec §1.2, Step 5.

        NOPAT               = EBIT_TTM x (1 - tax rate)
        Invested capital    = Equity + Debt - Cash
        ROIC                = NOPAT / average invested capital (5 quarters)
        Gross profitability = Gross profit (TTM) / Total assets
        Accruals            = (Net income - Operating cash flow) / average total assets
        EV                  = Market cap + Debt - Cash
        FCF yield           = FCF_TTM / EV
    """
    p = panel.copy()
    out = pd.DataFrame(index=p.index)
    out["nopat"] = p["ebit_ttm"] * (1 - tax_rate)
    ic = p["equity"] + p["total_debt"] - p["cash"]
    ic_avg = ic.groupby(p["entity_id"]).transform(lambda s: s.rolling(5, min_periods=2).mean())
    out["roic"] = out["nopat"] / ic_avg.where(ic_avg > 0)
    out["gross_profitability"] = p["gross_profit_ttm"] / p["total_assets"]
    out["accruals"] = (p["net_income_ttm"] - p["operating_cash_flow_ttm"]) / p["total_assets_avg5"]
    out["leverage"] = p["total_debt"] / p["total_assets"]
    out["roe"] = p["net_income_ttm"] / p["equity"].where(p["equity"] > 0)
    capex = p["capex_ttm"] if "capex_ttm" in p else 0.0
    out["fcf_ttm"] = p["operating_cash_flow_ttm"] - capex
    if market_cap is not None:
        ev = market_cap + p["total_debt"] - p["cash"]
        out["ev"] = ev
        out["fcf_yield"] = out["fcf_ttm"] / ev.where(ev > 0)
        out["earnings_yield"] = p["net_income_ttm"] / market_cap
        out["book_to_price"] = p["equity"] / market_cap
        ebitda = p["ebit_ttm"] + (p["depreciation_ttm"] if "depreciation_ttm" in p else 0.0)
        out["ebitda_ev"] = ebitda / ev.where(ev > 0)
    return out
