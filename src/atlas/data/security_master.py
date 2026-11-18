"""Security master (Spec §0.2).

Tickers change and get reused (FB -> META), so every company receives a permanent internal ID
mapped to all of its identifiers over time:

    security_master(internal_id, ticker, ISIN, CUSIP, FIGI, CIK, valid_from, valid_to)

Each identifier row is valid for a date range. A lookup "as of date d" returns the company that
held that identifier on that date, so ten years of history never mix two different companies.
"""

from __future__ import annotations

import pandas as pd

ID_COLUMNS = ["ticker", "isin", "cusip", "figi", "cik"]


class SecurityMaster:
    """Maps identifiers to permanent internal IDs with validity dates."""

    def __init__(self, table: pd.DataFrame | None = None):
        cols = ["internal_id", *ID_COLUMNS, "name", "valid_from", "valid_to"]
        self.table = table.copy() if table is not None else pd.DataFrame(columns=cols)
        for c in ("valid_from", "valid_to"):
            self.table[c] = pd.to_datetime(self.table[c])
        # An open-ended row (identifier still in use) is stored with valid_to = far future.
        self.table["valid_to"] = self.table["valid_to"].fillna(pd.Timestamp("2262-01-01"))

    def add(self, internal_id: str, valid_from, valid_to=None, name: str | None = None, **ids) -> None:
        """Add an identifier record. Unknown identifier types are rejected to avoid silent typos."""
        unknown = set(ids) - set(ID_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown identifier types: {unknown}")
        row = {"internal_id": internal_id, "name": name, "valid_from": pd.Timestamp(valid_from),
               "valid_to": pd.Timestamp(valid_to) if valid_to else pd.Timestamp("2262-01-01")}
        row.update({k: ids.get(k) for k in ID_COLUMNS})
        self.table = pd.concat([self.table, pd.DataFrame([row])], ignore_index=True)

    def lookup(self, value: str, as_of, id_type: str = "ticker") -> str | None:
        """Return the internal ID that held `value` on date `as_of`, or None.

        Example from Spec §0.2: a query for "FB" in 2019 and a query for "META" in 2024 both return
        the same internal ID.
        """
        as_of = pd.Timestamp(as_of)
        t = self.table
        hit = t[(t[id_type] == value) & (t["valid_from"] <= as_of) & (t["valid_to"] > as_of)]
        if len(hit) > 1:
            raise ValueError(f"Identifier {value} maps to several companies on {as_of.date()}.")
        return None if hit.empty else str(hit["internal_id"].iloc[0])

    def map_column(self, df: pd.DataFrame, id_col: str = "ticker", date_col: str = "date",
                   id_type: str = "ticker") -> pd.Series:
        """Vectorised lookup: return the internal ID for every (identifier, date) row of `df`."""
        merged = df[[id_col, date_col]].reset_index().merge(
            self.table[["internal_id", id_type, "valid_from", "valid_to"]],
            left_on=id_col, right_on=id_type, how="left")
        ok = (merged[date_col] >= merged["valid_from"]) & (merged[date_col] < merged["valid_to"])
        merged = merged[ok].drop_duplicates("index")
        return merged.set_index("index")["internal_id"].reindex(df.index)
