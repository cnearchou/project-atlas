"""Research registry (Spec §9.1).

Every experiment is appended to a CSV table, *including failures*:
    experiments(id, hypothesis_text, signal_definition, parameters, universe, period, n_variants_tried,
                ic_mean, ic_sd, t_nw, sharpe, dsr, passed_gate, date)
`n_variants_tried` is essential, because the deflated Sharpe ratio (Spec §9.3) corrects for how many
ideas were tested. Hypotheses are registered *before* they are tested (status 'registered'), with the
statistic and pass threshold fixed in advance (Section A.2).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

COLUMNS = ["id", "hypothesis_text", "signal_definition", "parameters", "universe", "period", "n_variants_tried",
           "ic_mean", "ic_sd", "t_nw", "sharpe", "dsr", "passed_gate", "status", "date"]


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_csv(self.path)
        return pd.DataFrame(columns=COLUMNS)

    def _append(self, row: dict) -> str:
        row = {c: row.get(c) for c in COLUMNS}
        row["id"] = row["id"] or uuid.uuid4().hex[:10]
        row["date"] = row["date"] or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        if isinstance(row["parameters"], dict):
            row["parameters"] = json.dumps(row["parameters"], default=str)
        df = pd.concat([self.load(), pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.path, index=False)
        return row["id"]

    def register(self, hypothesis_text: str, signal_definition: str, test: str, threshold: str, **kw) -> str:
        """Pre-register a hypothesis with its test fixed in advance."""
        return self._append({"hypothesis_text": hypothesis_text, "signal_definition": signal_definition,
                             "parameters": {"test": test, "threshold": threshold, **kw.pop("parameters", {})},
                             "status": "registered", **kw})

    def log(self, **result) -> str:
        """Log a completed experiment (pass or fail)."""
        result.setdefault("status", "tested")
        return self._append(result)

    def n_trials(self, signal_family: str) -> int:
        """Number of variants already tried for a family of signals (for the deflated Sharpe ratio)."""
        df = self.load()
        if df.empty:
            return 0
        return int(df["signal_definition"].fillna("").str.startswith(signal_family).sum())
