"""WRDS client: CRSP, Compustat, I/B/E/S and OptionMetrics (Technical Reference, Part 2.2–2.3).

WRDS is reached through the `wrds` package, which opens a PostgreSQL connection to WRDS servers.
The first connection asks for the WRDS username and password and can store them in ~/.pgpass.

Licensing: data downloaded from WRDS is for the account holder's research only and must never be
committed to GitHub or shared (Technical Reference, Part 2.1). The queries below save results in the
RAW layer under data/, which .gitignore excludes.

Undergraduate WRDS accounts are disabled between semesters unless they include summer access, so
large downloads should be scheduled during term (see the hosting discussion in the README).
"""

from __future__ import annotations

import pandas as pd

from atlas._optional import require
from atlas.config import Settings


class WRDSClient:
    def __init__(self, username: str | None = None):
        wrds = require("wrds", "data")
        self.db = wrds.Connection(wrds_username=username or Settings.secret("WRDS_USERNAME"))

    # ---- CRSP (Spec §0.3, §0.5) -----------------------------------------------------------------
    def crsp_daily(self, start: str, end: str) -> pd.DataFrame:
        """Daily stock file: prices, returns, shares, volume and adjustment factors.

        CRSP's `ret` already includes dividends; `prc` is negative when it is a bid/ask midpoint, so
        its absolute value is used downstream. `cfacpr` is CRSP's cumulative price adjustment factor.
        """
        q = f"""
            SELECT a.permno AS entity_id, a.date, ABS(a.prc) AS price, a.ret, a.retx,
                   a.shrout * 1000 AS shares_outstanding, a.vol AS volume, a.cfacpr, a.cfacshr,
                   b.shrcd, b.exchcd, b.siccd, b.ticker
            FROM crsp.dsf AS a
            LEFT JOIN crsp.msenames AS b
              ON a.permno = b.permno AND a.date BETWEEN b.namedt AND b.nameendt
            WHERE a.date BETWEEN '{start}' AND '{end}'
              AND b.shrcd IN (10, 11)            -- ordinary common shares only
        """
        return self.db.raw_sql(q, date_cols=["date"])

    def crsp_delistings(self) -> pd.DataFrame:
        """Delisting returns (Spec §0.5). Missing values are handled in atlas.data.universe."""
        q = "SELECT permno AS entity_id, dlstdt AS date, dlret AS delisting_return, dlstcd FROM crsp.dsedelist"
        return self.db.raw_sql(q, date_cols=["date"])

    # ---- Compustat (Spec §1) ------------------------------------------------------------------
    def compustat_quarterly(self, start: str, end: str) -> pd.DataFrame:
        """Quarterly fundamentals with the report date `rdq`, used as the knowledge date (Spec §0.4)."""
        q = f"""
            SELECT gvkey, datadate AS event_date, rdq AS knowledge_date, fyearq, fqtr,
                   saleq AS revenue, oiadpq AS ebit, niq AS net_income, oancfy AS operating_cash_flow_ytd,
                   atq AS total_assets, ltq AS total_liabilities, dlttq + dlcq AS total_debt,
                   ceqq AS equity, cheq AS cash, saleq - cogsq AS gross_profit, actq, lctq, req,
                   dpq AS depreciation, xsgaq AS sga, rectq AS receivables, cshoq AS shares
            FROM comp.fundq
            WHERE datadate BETWEEN '{start}' AND '{end}'
              AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
        """
        return self.db.raw_sql(q, date_cols=["event_date", "knowledge_date"])

    def ccm_link(self) -> pd.DataFrame:
        """CRSP–Compustat link table, which maps gvkey to permno over time (security master input)."""
        q = """SELECT gvkey, lpermno AS permno, linkdt AS valid_from, linkenddt AS valid_to
               FROM crsp.ccmxpf_linktable WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')"""
        return self.db.raw_sql(q, date_cols=["valid_from", "valid_to"])

    # ---- I/B/E/S (Spec §1.6, §5.2) ------------------------------------------------------------
    def ibes_summary(self, start: str, end: str) -> pd.DataFrame:
        """Monthly consensus EPS statistics: mean, standard deviation, number of up/down revisions."""
        q = f"""
            SELECT ticker, statpers AS date, fpedats, meanest, stdev, numest, numup, numdown
            FROM ibes.statsum_epsus
            WHERE statpers BETWEEN '{start}' AND '{end}' AND fpi = '1'
        """
        return self.db.raw_sql(q, date_cols=["date", "fpedats"])

    # ---- OptionMetrics (Spec §1.6, §4.2) ------------------------------------------------------
    def option_prices(self, year: int, secids: list[int] | None = None) -> pd.DataFrame:
        """Option quotes for one year. Tables are split by year (opprcd2023, ...)."""
        where = f"AND secid IN ({','.join(map(str, secids))})" if secids else ""
        q = f"""
            SELECT secid, date, exdate, cp_flag, strike_price / 1000.0 AS strike, best_bid, best_offer,
                   open_interest, impl_volatility, delta, gamma
            FROM optionm.opprcd{year} WHERE 1 = 1 {where}
        """
        return self.db.raw_sql(q, date_cols=["date", "exdate"])

    def close(self) -> None:
        self.db.close()
