"""SEC EDGAR client (Technical Reference, Part 2.2; Spec §1.1–§1.2, §3, §4, §6).

EDGAR is free and needs no key, but every request must carry a User-Agent with a name and email
(set SEC_USER_AGENT in .env), and requests are limited to about 10 per second.

The `companyfacts` endpoint returns every XBRL fact a company has filed. Each fact carries the
filing date (`filed`), which is used as the knowledge date for point-in-time storage (Spec §0.4).
"""

from __future__ import annotations

import pandas as pd

from atlas.config import Settings
from atlas.ingest.http import RateLimitedSession

BASE = "https://data.sec.gov"

# Priority mapping of XBRL tags to standard fields (Spec §1.2, Step 1). The first tag present for a
# company is used, and the chosen tag is recorded so unusual values can be investigated later.
TAG_MAP: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "ebit": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "gross_profit": ["GrossProfit"],
    "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "shares": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
}


class EdgarClient:
    def __init__(self, user_agent: str | None = None):
        ua = user_agent or Settings.secret("SEC_USER_AGENT", required=True)
        self.http = RateLimitedSession(min_interval=0.12, headers={"User-Agent": ua})

    def ticker_map(self) -> pd.DataFrame:
        """Current ticker -> CIK map published by the SEC (input to the security master, Spec §0.2)."""
        data = self.http.get("https://www.sec.gov/files/company_tickers.json").json()
        df = pd.DataFrame(data.values())
        df["cik"] = df["cik_str"].astype(str).str.zfill(10)
        return df[["cik", "ticker", "title"]]

    def company_facts(self, cik: str) -> dict:
        return self.http.get(f"{BASE}/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json").json()

    def facts_point_in_time(self, cik: str) -> pd.DataFrame:
        """Convert company facts into point-in-time rows using the tag priority map.

        Output columns: entity_id, field, event_date (period end), knowledge_date (filing date),
        value, form, tag, quarterly flag. Duration facts covering about three months are marked as
        quarterly; annual 10-K values are kept so Q4 can be derived (Spec §1.2, Step 2).
        """
        facts = self.company_facts(cik).get("facts", {}).get("us-gaap", {})
        rows = []
        for field, tags in TAG_MAP.items():
            for tag in tags:
                if tag not in facts:
                    continue
                for unit, items in facts[tag]["units"].items():
                    for it in items:
                        start = pd.to_datetime(it.get("start")) if it.get("start") else None
                        end = pd.to_datetime(it["end"])
                        days = (end - start).days if start is not None else 0
                        rows.append({"entity_id": str(cik).zfill(10), "field": field, "event_date": end,
                                     "knowledge_date": pd.to_datetime(it["filed"]), "value": float(it["val"]),
                                     "form": it.get("form"), "tag": tag, "unit": unit,
                                     "period_days": days})
                break  # first tag present wins (priority order)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["is_quarter"] = df["period_days"].between(80, 100)
        df["is_annual"] = df["period_days"].between(350, 380)
        return df

    def submissions(self, cik: str) -> pd.DataFrame:
        """List of filings (forms 10-K, 10-Q, 8-K, 4, 13F-HR ...) with dates and document names."""
        recent = self.http.get(f"{BASE}/submissions/CIK{str(cik).zfill(10)}.json").json()["filings"]["recent"]
        return pd.DataFrame(recent)

    def filing_text(self, cik: str, accession: str, document: str) -> str:
        """Raw text of one filing document (used for 10-K Item 1, customer disclosures, 8-Ks)."""
        acc = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{document}"
        return self.http.get(url).text
