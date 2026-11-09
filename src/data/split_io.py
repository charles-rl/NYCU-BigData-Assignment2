"""Shared helpers for the temporal-split CLI and the test suite."""

from __future__ import annotations

import glob
import os
from typing import cast

import pandas as pd

DEFAULT_DB = "processed_tables/db"


def load_tables(db: str = DEFAULT_DB) -> dict[str, pd.DataFrame]:
    """Load every parquet table in `db` as {table_name: DataFrame}."""
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(f"{db}/*.parquet")):
        tables[os.path.basename(path).split(".")[0]] = pd.read_parquet(path)
    return tables


def parse_cutoff(value: str) -> pd.Timestamp:
    """Parse a cutoff date. The parquet columns are tz-naive UTC."""
    ts = cast(pd.Timestamp, pd.Timestamp(value))
    if ts.tzinfo is not None:
        raise ValueError(
            f"cutoff must be tz-naive (got {value}); the tables store naive UTC times"
        )
    return ts
