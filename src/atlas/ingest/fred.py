"""FRED and ALFRED client (Technical Reference, Part 2.4; Spec §2.1–§2.2).

FRED provides current macro series; ALFRED provides every historical vintage, so a backtest dated
30 July uses the GDP figure known on 30 July rather than a later revision (Spec §2.2, Step 1).
A free API key is required (FRED_API_KEY in .env). The REST API is called directly with requests,
so the fredapi package is optional.
"""

from __future__ import annotations

import pandas as pd

from atlas.config import Settings
from atlas.ingest.http import RateLimitedSession

BASE = "https://api.stlouisfed.org/fred"

# Series used in Spec §2 (FRED identifiers).
SERIES = {
    "cpi": "CPIAUCSL", "core_pce": "PCEPILFE", "gdp": "GDPC1", "payrolls": "PAYEMS",
    "unemployment": "UNRATE", "fed_funds": "FEDFUNDS", "y2": "DGS2", "y10": "DGS10",
    "baa_spread": "BAA10Y", "oil": "DCOILWTICO", "usd": "DTWEXBGS", "vix": "VIXCLS",
}


class FredClient:
    def __init__(self, api_key: str | None = None):
        self.key = api_key or Settings.secret("FRED_API_KEY", required=True)
        self.http = RateLimitedSession(min_interval=0.6)  # FRED allows about 120 requests per minute

    def _obs(self, series_id: str, **params) -> pd.DataFrame:
        p = {"series_id": series_id, "api_key": self.key, "file_type": "json", **params}
        data = self.http.get(f"{BASE}/series/observations", params=p).json()["observations"]
        df = pd.DataFrame(data)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        for c in ("date", "realtime_start", "realtime_end"):
            df[c] = pd.to_datetime(df[c])
        return df

    def latest(self, series_id: str) -> pd.Series:
        """Current (latest-revised) values. Suitable for live use, not for backtests."""
        df = self._obs(series_id)
        return df.set_index("date")["value"].rename(series_id)

    def vintages(self, series_id: str) -> pd.DataFrame:
        """All vintages in point-in-time form (Spec §0.4): event_date = observation date,
        knowledge_date = realtime_start (the date that value became known)."""
        df = self._obs(series_id, realtime_start="1776-07-04", realtime_end="9999-12-31")
        return df.rename(columns={"date": "event_date", "realtime_start": "knowledge_date"})[
            ["event_date", "knowledge_date", "value"]].assign(field=series_id)

    def first_release(self, series_id: str) -> pd.Series:
        """First-release value of every observation (the advance estimate)."""
        v = self.vintages(series_id).sort_values("knowledge_date")
        first = v.groupby("event_date").first()
        return first["value"].rename(series_id)
