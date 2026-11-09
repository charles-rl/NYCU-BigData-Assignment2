"""Temporal split: a leakage-free feature view plus a future label view.

Two functions define the split:

  * `mask_future_leakage(tables, cutoff)` returns the feature view: every row
    whose row time is at or after the cutoff is dropped, and every column that
    only becomes known later is blanked. The result is exactly the information
    a model would have at the cutoff.
  * `extract_labels(tables, cutoff, end, users)` returns the label view: the 6
    activities a user performs in the half-open window [cutoff, end).

The two views never overlap: the feature view is `< cutoff`, the label window
is `>= cutoff`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

# Row time per table: the time at which the row becomes visible, so the row
# must be dropped from the feature view when row_time >= cutoff.
ROW_TIME: dict[str, str] = {
    "Users": "first_seen_at",
    "Repositories": "first_seen_at",
    "PullRequests": "first_seen_at",
    "Issues": "first_seen_at",
    "IssueComments": "created_at",
    "PRReviews": "first_seen_at",
    "PRReviewComments": "created_at",
    "Pushes": "created_at",
}

# Columns that only become known after the cutoff. Filtering rows by row time
# is not enough for these: the row exists earlier, the value is written later.
# Tuple per entry: (table, gating column, columns blanked together with it).
FUTURE_MASKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("PullRequests", "merged_at", ("merged_at", "merged_by_id")),
    ("PullRequests", "closed_at", ("closed_at",)),
    ("Issues", "closed_at", ("closed_at", "state_reason")),
    ("PRReviews", "dismissed_at", ("dismissed_at",)),
)


def _gated_columns() -> dict[str, list[str]]:
    """Gating column names per table, i.e. the values that must be hidden."""
    gated: dict[str, list[str]] = {}
    for table, gate, _ in FUTURE_MASKS:
        gated.setdefault(table, []).append(gate)
    return gated


# FUTURE_COLS["Issues"] == ["closed_at"], and so on.
FUTURE_COLS: dict[str, list[str]] = _gated_columns()

# One entry per label: (label, table, user column, event-time column, filter).
# `filter` is None or "not_pull_request" (drop PR conversation rows).
ACTIVITY_LABELS: tuple[tuple[str, str, str, str, str | None], ...] = (
    ("open_pr", "PullRequests", "author_id", "created_at", None),
    ("open_issue", "Issues", "author_id", "created_at", "not_pull_request"),
    ("issue_comment", "IssueComments", "author_id", "created_at", None),
    ("pr_review", "PRReviews", "reviewer_id", "submitted_at", None),
    ("pr_review_comment", "PRReviewComments", "author_id", "created_at", None),
    ("push", "Pushes", "user_id", "created_at", None),
)

LABEL_COLS: list[str] = [label for label, *_ in ACTIVITY_LABELS]


def _is_datetime(series: pd.Series) -> bool:
    return bool(pd.api.types.is_datetime64_any_dtype(series))


def mask_future_leakage(
    tables: dict[str, pd.DataFrame], cutoff_time: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Return the feature view: only data visible strictly before `cutoff_time`."""
    filtered: dict[str, pd.DataFrame] = {}

    # 1. Drop rows whose row time is at or after the cutoff.
    for name, df in tables.items():
        row_time = ROW_TIME.get(name)
        if row_time and row_time in df.columns:
            filtered[name] = cast(pd.DataFrame, df[df[row_time] < cutoff_time]).copy()
        else:
            filtered[name] = df.copy()

    # 2. Blank the values that only become known later. Gating on NaT yields
    #    False, so already-missing values stay missing.
    for table, gate, companions in FUTURE_MASKS:
        df = filtered.get(table)
        if df is None or gate not in df.columns:
            continue
        late = df[gate] >= cutoff_time
        if not late.any():
            continue
        for col in companions:
            if col not in df.columns:
                continue
            df.loc[late, col] = pd.NaT if _is_datetime(cast(pd.Series, df[col])) else np.nan

    return filtered


def extract_labels(
    tables: dict[str, pd.DataFrame],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    users_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return the 6 binary activity labels for the window [start_time, end_time).

    A table that is absent (or lacks a needed column) contributes all zeros
    rather than raising, so a partial table set still yields a usable frame.
    """
    labels = pd.DataFrame()
    labels["user_id"] = users_df["user_id"].to_numpy()
    labels["UserID"] = users_df["github_user_id"].to_numpy()

    for name, table, user_col, time_col, row_filter in ACTIVITY_LABELS:
        df = tables.get(table)
        active_users: set[int] = set()
        if df is not None and not df.empty and user_col in df.columns and time_col in df.columns:
            event_time = pd.to_datetime(df[time_col])
            window = (event_time >= start_time) & (event_time < end_time)
            if row_filter == "not_pull_request" and "is_pull_request" in df.columns:
                # a missing flag means "not a PR conversation"
                is_pr = (
                    df["is_pull_request"].astype("boolean").fillna(False).astype(bool)
                )
                window &= ~is_pr
            active_users = set(df.loc[window, user_col].dropna().astype(int))
        labels[name] = labels["user_id"].isin(sorted(active_users)).astype(int)

    return labels
