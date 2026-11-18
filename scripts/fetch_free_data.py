"""Download a first set of free, real data (Section A.4 'Minimum Viable Data Plan').

The script stores raw downloads in data/raw (write-once, Spec §0.1). It needs a .env file with
SEC_USER_AGENT (name and email, required by the SEC) and FRED_API_KEY (free from fred.stlouisfed.org).

    python scripts/fetch_free_data.py --tickers AAPL MSFT NVDA --fred

Downloaded data stays on the local machine: data/ is listed in .gitignore and is never pushed to
GitHub. Licensed WRDS data (CRSP, Compustat, I/B/E/S, OptionMetrics) is fetched with
atlas.ingest.wrds_client and must never be redistributed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from atlas.config import load_settings  # noqa: E402
from atlas.data.storage import DataStore  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT"])
    p.add_argument("--fred", action="store_true", help="also download the macro series of Spec §2.1 from FRED")
    a = p.parse_args()
    S = load_settings()
    store = DataStore(S.path("data_root"))
    stamp = pd.Timestamp.now().strftime("%Y%m%d")

    from atlas.ingest.edgar import EdgarClient
    ed = EdgarClient()
    tmap = ed.ticker_map()
    rows = []
    for t in a.tickers:
        cik = tmap.loc[tmap["ticker"] == t.upper(), "cik"]
        if cik.empty:
            print(f"{t}: not found in the SEC ticker map")
            continue
        df = ed.facts_point_in_time(cik.iloc[0]).assign(ticker=t.upper())
        rows.append(df)
        print(f"{t}: {len(df)} point-in-time facts")
    if rows:
        store.write("raw", f"edgar_facts_{stamp}", pd.concat(rows, ignore_index=True))

    if a.fred:
        from atlas.ingest.fred import SERIES, FredClient
        fr = FredClient()
        for name, sid in SERIES.items():
            v = fr.vintages(sid)
            store.write("raw", f"fred_{sid}_{stamp}", v)
            print(f"FRED {sid} ({name}): {len(v)} vintage rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
