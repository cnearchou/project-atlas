"""Other public sources (Technical Reference, Parts 2.5–2.7).

* GDELT DOC API: news article lists for a query (Spec §3.1). No key.
* congress.gov API: bills, stages and cosponsors (Spec §3.7). Free key from api.data.gov.
* Polymarket and Kalshi: prediction-market prices for elections (Spec §3.6). Free to read.
* FINRA: short-interest and daily short-volume files (Spec §4.1). Public downloads.

Endpoints change from time to time; each function keeps the URL in one place so it is easy to update.
"""

from __future__ import annotations

import io

import pandas as pd

from atlas.config import Settings
from atlas.ingest.http import RateLimitedSession

_http = RateLimitedSession(min_interval=0.5, headers={"User-Agent": "project-atlas research"})


# ---- GDELT (Spec §3.1) --------------------------------------------------------------------------
def gdelt_articles(query: str, start: str, end: str, max_records: int = 250) -> pd.DataFrame:
    """Articles matching a query from the GDELT DOC 2.0 API.

    Dates use the format YYYYMMDDHHMMSS. The result includes url, title, seendate (UTC), domain and
    language, which feed the processing pipeline in Spec §3.2.
    """
    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": max_records,
              "startdatetime": start, "enddatetime": end}
    data = _http.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params).json()
    df = pd.DataFrame(data.get("articles", []))
    if not df.empty:
        df["published_utc"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", utc=True)
    return df


# ---- congress.gov (Spec §3.7) -------------------------------------------------------------------
def congress_bills(congress: int, limit: int = 250, offset: int = 0) -> pd.DataFrame:
    key = Settings.secret("CONGRESS_API_KEY", required=True)
    url = f"https://api.congress.gov/v3/bill/{congress}"
    data = _http.get(url, params={"api_key": key, "limit": limit, "offset": offset, "format": "json"}).json()
    return pd.json_normalize(data.get("bills", []))


# ---- Prediction markets (Spec §3.6) -----------------------------------------------------------
def polymarket_markets(search: str | None = None, limit: int = 100) -> pd.DataFrame:
    """Market list from Polymarket's public Gamma API (outcomes and current prices)."""
    params = {"limit": limit, "active": "true"}
    data = _http.get("https://gamma-api.polymarket.com/markets", params=params).json()
    df = pd.json_normalize(data)
    if search and not df.empty and "question" in df:
        df = df[df["question"].str.contains(search, case=False, na=False)]
    return df


def kalshi_markets(event_ticker: str | None = None, limit: int = 200) -> pd.DataFrame:
    """Markets from Kalshi's public trade API (read-only market data needs no key)."""
    params = {"limit": limit}
    if event_ticker:
        params["event_ticker"] = event_ticker
    data = _http.get("https://api.elections.kalshi.com/trade-api/v2/markets", params=params).json()
    return pd.json_normalize(data.get("markets", []))


# ---- FINRA (Spec §4.1) -------------------------------------------------------------------------
def finra_daily_short_volume(date: str) -> pd.DataFrame:
    """Reg SHO consolidated daily short-sale volume file for one date (YYYYMMDD)."""
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    text = _http.get(url).text
    df = pd.read_csv(io.StringIO(text), sep="|")
    return df[df["Date"].astype(str).str.isdigit()]
