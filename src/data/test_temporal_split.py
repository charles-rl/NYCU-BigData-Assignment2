"""Run the temporal split end-to-end on the real parquet tables.

Run from the repo root:
    python -m src.data.test_temporal_split
    python -m src.data.test_temporal_split --cutoff 2026-10-08 --days 7

What it does:
  1. loads processed_tables/db/*.parquet
  2. feature view  = mask_future_leakage(tables, cutoff)      -> rows/cols after the cutoff removed
  3. label view    = extract_labels(RAW tables, cutoff, end)  -> ground truth for the next `days` days
  4. asserts there is no leakage in the feature view and that the labels are not all zero
  5. writes the split to splits/ (gitignored)

Two traps this script is designed to catch:
  * the parquet timestamps are tz-NAIVE (README: "Times are in UTC (no time zone is
    stored)"), so a tz-aware cutoff raises `TypeError: Invalid comparison between
    dtype=datetime64[ns] and Timestamp`.
  * labels must come from the RAW tables. mask_future_leakage drops every row with
    row_time >= cutoff, so extracting labels from its output gives all zeros.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import cast

import pandas as pd

from src.data.temporal_split import extract_labels, mask_future_leakage

DB = "processed_tables/db"
ROW_TIME = {
    "Users": "first_seen_at",
    "Repositories": "first_seen_at",
    "PullRequests": "first_seen_at",
    "Issues": "first_seen_at",
    "IssueComments": "created_at",
    "PRReviews": "first_seen_at",
    "PRReviewComments": "created_at",
    "Pushes": "created_at",
}
# columns that may only be used in features when they are < cutoff
FUTURE_COLS = {
    "PullRequests": ["merged_at", "closed_at"],
    "Issues": ["closed_at"],
    "PRReviews": ["dismissed_at"],
}
LABEL_COLS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]


def load_tables(db: str = DB) -> dict[str, pd.DataFrame]:
    tables = {}
    for p in sorted(glob.glob(f"{db}/*.parquet")):
        tables[os.path.basename(p).split(".")[0]] = pd.read_parquet(p)
    return tables


def parse_cutoff(s: str) -> pd.Timestamp:
    """Naive UTC timestamp — the parquet columns are tz-naive."""
    ts = cast(pd.Timestamp, pd.Timestamp(s))
    if ts.tzinfo is not None:
        raise ValueError(f"cutoff must be tz-naive (got {s}); the tables store naive UTC times")
    return ts


def main() -> int:
    ap = argparse.ArgumentParser(description="exercise mask_future_leakage + extract_labels")
    ap.add_argument("--cutoff", default=None,
                    help="feature cutoff, naive UTC (default: last day of data - 7 days)")
    ap.add_argument("--days", type=int, default=7, help="label window length after the cutoff")
    ap.add_argument("--out", default="splits", help="output folder (gitignored)")
    args = ap.parse_args()

    tables = load_tables()
    if not tables:
        print(f"no parquet files in {DB}")
        return 1

    latest = cast(pd.Timestamp, max(
        cast(pd.Series, tables[t][ROW_TIME[t]]).max() for t in tables if ROW_TIME[t] in tables[t].columns))
    earliest = cast(pd.Timestamp, min(
        cast(pd.Series, tables[t][ROW_TIME[t]]).min() for t in tables if ROW_TIME[t] in tables[t].columns))
    cutoff = cast(pd.Timestamp, parse_cutoff(args.cutoff) if args.cutoff
                  else latest.normalize() - pd.Timedelta(days=args.days))
    end = cast(pd.Timestamp, cutoff + pd.Timedelta(days=args.days))

    print("Testing temporal split functions...")
    print(f"Data window in files:      {earliest} .. {latest}")
    print(f"Validation feature cutoff: {cutoff}")
    print(f"Validation label window:   {cutoff} to {end}")
    if cutoff > latest:
        print(f"!! WARNING: cutoff {cutoff} is after your last row ({latest}) — the label window "
              f"contains no data and every label will be 0.")

    # ---- 1. features: everything visible before the cutoff -------------------
    features = mask_future_leakage(tables, cutoff)
    print("\nrows before -> after masking:")
    for name in tables:
        print(f"  {name:<17} {len(tables[name]):>6} -> {len(features[name]):>6}")

    # ---- 2. labels: from the RAW tables, not the masked ones ----------------
    cohort = cast(pd.DataFrame, tables["Users"][cast(pd.Series, tables["Users"]["is_cohort"]).astype(bool)])
    labels = extract_labels(tables, cutoff, end, cohort)
    naive_labels = extract_labels(features, cutoff, end, cohort)  # the trap, for comparison

    print(f"\nlabels for {len(labels)} cohort users (positives):")
    for c in LABEL_COLS:
        print(f"  {c:<18} {int(labels[c].sum()):>5}"
              f"   (from masked tables it would be {int(naive_labels[c].sum())})")

    # ---- 3. leakage assertions ---------------------------------------------
    failures: list[str] = []
    for name, tc in ROW_TIME.items():
        if name in features and tc in features[name].columns:
            n = int((features[name][tc] >= cutoff).sum())
            if n:
                failures.append(f"{name}.{tc}: {n} rows at/after cutoff")
    for name, cols in FUTURE_COLS.items():
        for c in cols:
            if name in features and c in features[name].columns:
                n = int((features[name][c] >= cutoff).sum())
                if n:
                    failures.append(f"{name}.{c}: {n} values at/after cutoff")
    if (labels[LABEL_COLS].sum(axis=1) == 0).all():
        failures.append("labels: every cohort user has 0 positives — check the cutoff")

    masked_counts = {
        f"{name}.{c}": int((tables[name][c] >= cutoff).sum()) - int((features[name][c] >= cutoff).sum())
        for name, cols in FUTURE_COLS.items() for c in cols
    }
    print("\nleakage check (values the mask had to hide):")
    for k, v in masked_counts.items():
        print(f"  {k:<26} {v:>5} hidden")
    for f in failures:
        print(f"  FAIL {f}")
    print("leakage check:", "FAILED" if failures else "OK — no feature is known at/after the cutoff")

    # ---- 4. write the split -------------------------------------------------
    out = Path(args.out)
    (out / "features").mkdir(parents=True, exist_ok=True)
    for name, df in features.items():
        df.to_parquet(out / "features" / f"{name}.parquet")

    truth = labels[["UserID", *LABEL_COLS]].copy()
    truth.to_csv(out / "answer.csv", index=False)

    pred = truth.copy()
    pred[LABEL_COLS] = 0
    pred.to_csv(out / "all_zero_submission.csv", index=False)

    print(f"\nwrote {out}/features/*.parquet  ({len(features)} tables, "
          f"{sum(len(d) for d in features.values())} rows)")
    print(f"wrote {out}/answer.csv               ({len(truth)} rows, format of an answer file)")
    print(f"wrote {out}/all_zero_submission.csv  (sanity submission)")
    print(f"\nscore it with:\n  python scripts/evaluate.py --pred {out}/all_zero_submission.csv "
          f"--truth {out}/answer.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
