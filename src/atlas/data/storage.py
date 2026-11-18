"""Storage layers (Spec §0.1).

The specification separates four layers: RAW (as downloaded, never modified), CLEAN (standardised,
adjusted, point-in-time), FEATURES (one value per company per date per signal) and MODEL OUTPUTS
(forecasts, probabilities, risk). Every table is keyed by (entity_id, date).

Data is stored as Parquet when pyarrow is installed and queried with DuckDB when it is installed.
Without those optional packages the store falls back to pickle files read with pandas, so the
pipeline still runs; only speed and file size differ.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas._optional import optional_import

LAYERS = ("raw", "clean", "features", "outputs")
KEY = ["entity_id", "date"]


class DataStore:
    """A small file-based store with one folder per layer and one file per table."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        for layer in LAYERS:
            (self.root / layer).mkdir(parents=True, exist_ok=True)
        # Parquet needs pyarrow (or fastparquet); DuckDB is optional for SQL queries.
        self._parquet = optional_import("pyarrow") is not None
        self._duckdb = optional_import("duckdb")

    def _file(self, layer: str, table: str) -> Path:
        if layer not in LAYERS:
            raise ValueError(f"Unknown layer '{layer}'. Expected one of {LAYERS}.")
        return self.root / layer / (f"{table}.parquet" if self._parquet else f"{table}.pkl")

    def write(self, layer: str, table: str, df: pd.DataFrame) -> Path:
        """Write a table. RAW tables are write-once: an existing raw table is never overwritten.

        The write-once rule implements "RAW (as downloaded, never modified)" from Spec §0.1: a new
        download is saved under a new table name (for example with a date suffix) instead.
        """
        path = self._file(layer, table)
        if layer == "raw" and path.exists():
            raise FileExistsError(f"Raw table '{table}' already exists and is never modified (Spec §0.1).")
        if self._parquet:
            df.to_parquet(path, index=False)
        else:
            df.to_pickle(path)
        return path

    def read(self, layer: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        path = self._file(layer, table)
        if self._parquet:
            return pd.read_parquet(path, columns=columns)
        df = pd.read_pickle(path)
        return df[columns] if columns else df

    def exists(self, layer: str, table: str) -> bool:
        return self._file(layer, table).exists()

    def sql(self, query: str) -> pd.DataFrame:
        """Run a DuckDB SQL query directly on Parquet files.

        Tables are referenced as '<layer>/<table>.parquet' relative to the store root, for example:
        SELECT * FROM 'clean/prices.parquet' WHERE date >= '2020-01-01'.
        """
        if self._duckdb is None or not self._parquet:
            raise ImportError("SQL queries need duckdb and pyarrow: pip install 'project-atlas[storage]'")
        con = self._duckdb.connect()
        con.execute(f"SET file_search_path='{self.root}'")
        return con.execute(query).df()
