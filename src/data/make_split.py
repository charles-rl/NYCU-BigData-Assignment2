"""Build the temporal split and report the leakage checks.

Run from the repo root:
    python -m src.data.make_split
    python -m src.data.make_split --cutoff 2026-10-24 --days 7

Writes (gitignored):
    splits/features/*.parquet        feature view, only data before the cutoff
    splits/answer.csv                the 6 labels for the cohort users
    splits/all_zero_submission.csv   sanity submission

Labels are extracted from the RAW tables. `mask_future_leakage` drops every row
at or after the cutoff, so labelling its output gives all zeros.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
from pathlib import Path
from typing import cast

import pandas as pd

from src.data.split_io import DEFAULT_DB, load_tables, parse_cutoff
from src.data.temporal_split import (
    ACTIVITY_LABELS,
    FUTURE_COLS,
    LABEL_COLS,
    ROW_TIME,
    extract_labels,
    mask_future_leakage,
)


class Split(namedtuple("Split", "cutoff end features labels cohort")):
    """The two views of one temporal split."""

    __slots__ = ()

    cutoff: pd.Timestamp
    end: pd.Timestamp
    features: dict[str, pd.DataFrame]
    labels: pd.DataFrame
    cohort: pd.DataFrame


def data_window(tables: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest and latest row time across the tables."""
    stamps = [
        cast(pd.Series, tables[name][col])
        for name, col in ROW_TIME.items()
        if name in tables and col in tables[name].columns
    ]
    if not stamps:
        raise ValueError("no row-time column found; cannot determine the data window")
    return (
        cast(pd.Timestamp, min(s.min() for s in stamps)),
        cast(pd.Timestamp, max(s.max() for s in stamps)),
    )


def cohort_users(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cohort users only: the test list contains cohort users only."""
    users = tables["Users"]
    if "is_cohort" not in users.columns:
        return users
    return cast(pd.DataFrame, users[users["is_cohort"].fillna(False).astype(bool)])


def build_split(
    tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp, days: int = 7
) -> Split:
    """Mask the features before `cutoff` and label the window after it."""
    end = cast(pd.Timestamp, cutoff + pd.Timedelta(days=days))
    features = mask_future_leakage(tables, cutoff)
    cohort = cohort_users(tables)
    labels = extract_labels(tables, cutoff, end, cohort)
    return Split(cutoff, end, features, labels, cohort)


def leakage_failures(tables: dict[str, pd.DataFrame], split: Split) -> list[str]:
    """Every way the split can leak. An empty list means the split is clean."""
    failures: list[str] = []

    if split.end <= split.cutoff:
        failures.append(f"label window {split.cutoff}..{split.end} is not after the cutoff")

    # 1. No row in the feature view may be stamped at or after the cutoff.
    for name, col in ROW_TIME.items():
        if name in split.features and col in split.features[name].columns:
            n = int((split.features[name][col] >= split.cutoff).sum())
            if n:
                failures.append(f"{name}.{col}: {n} rows at/after the cutoff")

    # 2. No value that becomes known later may survive.
    for name, cols in FUTURE_COLS.items():
        for col in cols:
            if name in split.features and col in split.features[name].columns:
                n = int((split.features[name][col] >= split.cutoff).sum())
                if n:
                    failures.append(f"{name}.{col}: {n} values at/after the cutoff")

    # 3. The labels must have positives, else the cutoff is wrong.
    if split.labels[LABEL_COLS].sum(axis=1).eq(0).all():
        failures.append("labels: every user has 0 positives - check the cutoff")

    # 4. Clock skew: the label window must not be visible in the feature view.
    #    Some labels are timed by the real event time (`created_at`,
    #    `submitted_at`) while the feature view is filtered by row time
    #    (`first_seen_at`). An event in [cutoff, end) whose row is already in
    #    the feature view means the graph contains the very thing we label.
    for _label, table, _user_col, event_col, _row_filter in ACTIVITY_LABELS:
        row_col = ROW_TIME.get(table)
        df = tables.get(table)
        if df is None or row_col is None or event_col == row_col:
            continue
        if event_col not in df.columns or row_col not in df.columns:
            continue
        event = pd.to_datetime(df[event_col])
        in_window = (event >= split.cutoff) & (event < split.end)
        visible = df[row_col] < split.cutoff
        n = int((in_window & visible).sum())
        if n:
            failures.append(
                f"{table}: {n} events in the label window are visible in the feature view "
                f"({event_col} in window while {row_col} < cutoff)"
            )

    return failures


def hidden_counts(tables: dict[str, pd.DataFrame], split: Split) -> dict[str, int]:
    """How many values the mask had to hide, per future column."""
    counts: dict[str, int] = {}
    for name, cols in FUTURE_COLS.items():
        for col in cols:
            if name not in tables or col not in tables[name].columns:
                continue
            raw = int((tables[name][col] >= split.cutoff).sum())
            kept = int((split.features[name][col] >= split.cutoff).sum())
            counts[f"{name}.{col}"] = raw - kept
    return counts


def write_split(out: Path, split: Split) -> None:
    """Write the feature view, the answer file, and a zero baseline."""
    (out / "features").mkdir(parents=True, exist_ok=True)
    for name, df in split.features.items():
        df.to_parquet(out / "features" / f"{name}.parquet")

    truth = split.labels[["UserID", *LABEL_COLS]].copy()
    truth.to_csv(out / "answer.csv", index=False)

    baseline = truth.copy()
    baseline[LABEL_COLS] = 0
    baseline.to_csv(out / "all_zero_submission.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the temporal split and check leakage")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"parquet folder (default: {DEFAULT_DB})")
    ap.add_argument("--cutoff", default=None,
                    help="feature cutoff, naive UTC (default: last day of data - 7 days)")
    ap.add_argument("--days", type=int, default=7, help="label window length after the cutoff")
    ap.add_argument("--out", default="splits", help="output folder (gitignored)")
    args = ap.parse_args(argv)

    tables = load_tables(args.db)
    if not tables:
        print(f"no parquet files in {args.db}")
        return 1

    earliest, latest = data_window(tables)
    cutoff = (
        parse_cutoff(args.cutoff)
        if args.cutoff
        else cast(pd.Timestamp, latest.normalize() - pd.Timedelta(days=args.days))
    )

    split = build_split(tables, cutoff, args.days)

    print("Building the temporal split...")
    print(f"Data window in files:      {earliest} .. {latest}")
    print(f"Validation feature cutoff: {split.cutoff}")
    print(f"Validation label window:   {split.cutoff} to {split.end}")
    if split.cutoff > latest:
        print(f"!! WARNING: cutoff {split.cutoff} is after your last row ({latest}) - "
              f"the label window contains no data and every label will be 0.")

    print("\nrows before -> after masking:")
    for name in tables:
        print(f"  {name:<17} {len(tables[name]):>6} -> {len(split.features[name]):>6}")

    # Labels from the masked view are all zero: that is the trap this split avoids.
    naive = extract_labels(split.features, split.cutoff, split.end, split.cohort)
    print(f"\nlabels for {len(split.labels)} cohort users (positives):")
    for col in LABEL_COLS:
        print(f"  {col:<18} {int(split.labels[col].sum()):>5}"
              f"   (from masked tables it would be {int(naive[col].sum())})")

    print("\nleakage check (values the mask had to hide):")
    for key, count in hidden_counts(tables, split).items():
        print(f"  {key:<26} {count:>5} hidden")

    failures = leakage_failures(tables, split)
    for failure in failures:
        print(f"  FAIL {failure}")
    print("leakage check:", "FAILED" if failures else "OK - no feature is known at/after the cutoff")

    if failures:
        print("\nnot writing the split because the leakage check failed")
        return 1

    out = Path(args.out)
    write_split(out, split)
    print(f"\nwrote {out}/features/*.parquet  ({len(split.features)} tables, "
          f"{sum(len(d) for d in split.features.values())} rows)")
    print(f"wrote {out}/answer.csv               ({len(split.labels)} rows)")
    print(f"wrote {out}/all_zero_submission.csv  (sanity submission)")
    print("\nscore it with:\n  python scripts/evaluate.py "
          f"--pred {out}/all_zero_submission.csv --truth {out}/answer.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
