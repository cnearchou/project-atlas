"""Point-in-time (bitemporal) storage (Spec §0.4).

"The most important rule in the system." Every data point is stored with two dates:

    (event_date, knowledge_date, value)
    e.g. (Q3 revenue period end = 30 Sep, filed = 2 Nov, $520M)

A query "as of date d" returns only rows where knowledge_date <= d. Restatements are stored as new
rows, never as overwrites, so a backtest only sees what was knowable at the time. This prevents
look-ahead bias, the most common cause of fake backtest results.
"""

from __future__ import annotations

import pandas as pd


class PointInTimeTable:
    """A bitemporal table: one row per (entity, field, event_date, knowledge_date)."""

    COLUMNS = ["entity_id", "field", "event_date", "knowledge_date", "value"]

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = pd.DataFrame(columns=self.COLUMNS) if df is None else df[self.COLUMNS].copy()
        for c in ("event_date", "knowledge_date"):
            self.df[c] = pd.to_datetime(self.df[c])

    def append(self, rows: pd.DataFrame) -> None:
        """Add new rows. Existing rows are never changed; a restatement is simply a newer row."""
        rows = rows[self.COLUMNS].copy()
        for c in ("event_date", "knowledge_date"):
            rows[c] = pd.to_datetime(rows[c])
        self.df = pd.concat([self.df, rows], ignore_index=True)

    def as_of(self, d) -> pd.DataFrame:
        """Return the latest known value of each (entity, field, event_date) as of date d.

        Rows with knowledge_date > d are invisible. Among the visible versions of the same fact,
        the most recently known one wins (so a restatement known by d replaces the original).
        """
        d = pd.Timestamp(d)
        visible = self.df[self.df["knowledge_date"] <= d]
        visible = visible.sort_values("knowledge_date")
        return visible.groupby(["entity_id", "field", "event_date"], as_index=False).last()

    def latest_value(self, d, field: str) -> pd.Series:
        """Latest event_date's value of `field` for every entity, as known on date d."""
        snap = self.as_of(d)
        snap = snap[snap["field"] == field].sort_values("event_date")
        return snap.groupby("entity_id")["value"].last()


def asof_join(left: pd.DataFrame, right: pd.DataFrame, by: str = "entity_id", left_on: str = "date",
              right_on: str = "knowledge_date", max_age: pd.Timedelta | None = None) -> pd.DataFrame:
    """Backward as-of join (Spec §0.6 'Mixed frequencies').

    Each left row (for example a daily price) takes the most recent right row (for example quarterly
    accounts) whose knowledge_date is on or before the left date. An optional maximum age stops very
    old values being carried forward indefinitely (the forward-fill limit in Spec §0.6).
    """
    l = left.sort_values(left_on).copy()
    r = right.sort_values(right_on).copy()
    l[left_on] = pd.to_datetime(l[left_on])
    r[right_on] = pd.to_datetime(r[right_on])
    out = pd.merge_asof(l, r, left_on=left_on, right_on=right_on, by=by, direction="backward",
                        tolerance=max_age)
    return out.sort_index()
