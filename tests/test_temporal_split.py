"""Automated checks for `src.data.temporal_split` and `src.data.make_split`.

What is verified:
  1. `mask_future_leakage` drops every row at/after the cutoff (half-open),
     blanks the future-only columns, leaves the input untouched, and never
     turns a clean value into a leak.
  2. `extract_labels` produces the 6 activity labels on the half-open window
     [cutoff, end), excludes PR conversations from `open_issue`, and survives
     absent tables / NaT timestamps.
  3. `build_split` keeps cohort users only and pairs a strict-past feature view
     with a future label window.
  4. `leakage_failures` returns [] on a clean split and catches each class of
     defect: row-time leaks, unmasked future columns, all-zero labels, and
     clock skew between the event time and the row time.
  5. `write_split` / `main` produce the three split artefacts and refuse to
     write a split that failed the leakage check.
  6. The six viable structural text/metadata features are present, non-null,
     and correctly derived.

Run from the repository root:
    python -m pytest tests/test_temporal_split.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.data.make_split import (
    build_split,
    cohort_users,
    data_window,
    hidden_counts,
    leakage_failures,
    main,
    write_split,
)
from src.data.split_io import load_tables, parse_cutoff
from src.data.temporal_split import (
    ACTIVITY_LABELS,
    FUTURE_COLS,
    LABEL_COLS,
    ROW_TIME,
    extract_labels,
    mask_future_leakage,
)

CUTOFF = cast(pd.Timestamp, pd.Timestamp("2024-06-01 00:00:00"))
END = cast(pd.Timestamp, CUTOFF + pd.Timedelta(days=7))


def _t(days: float) -> pd.Timestamp:
    """Timestamp `days` after the cutoff (negative = before)."""
    return cast(pd.Timestamp, CUTOFF + pd.Timedelta(days=days))


def make_tables() -> dict[str, pd.DataFrame]:
    """All eight tables, with rows placed around the cutoff on purpose."""
    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 3],
            "github_user_id": [10, 11, 12, 13],
            "login": ["alice", "bob", "carol", "dave"],
            "is_bot": [False, True, False, False],
            "is_cohort": [True, True, False, True],
            "first_seen_at": [_t(-100), _t(-50), _t(-70), _t(-1)],
        }
    )
    repositories = pd.DataFrame(
        {
            "repo_id": [0, 1],
            "owner_login": ["alice", "bob"],
            "org_id": [np.nan, 9001.0],
            "first_seen_at": [_t(-90), _t(+3)],  # repo 1 is future -> dropped
        }
    )
    pull_requests = pd.DataFrame(
        {
            "pr_id": [0, 1, 2],
            "repo_id": [0, 0, 1],
            "author_id": [0, 1, 2],
            "draft": [False, True, None],
            "head_ref": ["aaaa/aaaa", "aaa", "aaa/aaaa/aa"],
            "is_cross_repo": [False, True, False],
            # pr0: opened and visible before the cutoff, merged inside the
            # window -> its merged_at/merged_by_id must be blanked.
            # pr1: opened and seen inside the window -> dropped, and it is the
            # open_pr label for user 1.
            # pr2: visible, never merged.
            "created_at": [_t(-2), _t(+1), _t(-6)],
            "first_seen_at": [_t(-2), _t(+1), _t(-6)],
            "merged_at": [_t(+2), _t(+5), pd.NaT],
            "merged_by_id": [1.0, 2.0, np.nan],
            "closed_at": [pd.NaT, _t(+6), pd.NaT],
        }
    )
    issues = pd.DataFrame(
        {
            "issue_id": [0, 1, 2],
            "repo_id": [0, 0, 0],
            "author_id": [0, 1, 2],
            "is_pull_request": [False, True, pd.NA],
            "num_labels": [1.0, 0.0, 2.0],
            "title": ["aaa(aaa): aaaaa", "aaaa", "aaaaa"],
            "body": ["### aaaa", "aa", "```\naaa\n```"],
            # iss0 is visible and closed inside the window -> blanked.
            # iss1 is a PR conversation inside the window -> not an open_issue.
            # iss2 has a missing is_pull_request flag -> still an open_issue.
            "created_at": [_t(-5), _t(+1), _t(+4)],
            "first_seen_at": [_t(-5), _t(+1), _t(+4)],
            "closed_at": [_t(+1), _t(+2), pd.NaT],
            "state_reason": ["completed", "completed", np.nan],
        }
    )
    issue_comments = pd.DataFrame(
        {
            "comment_id": [0, 1, 2],
            "issue_id": [0, 0, 1],
            "repo_id": [0, 0, 0],
            "author_id": [0, 2, 3],
            "is_pull_request": [False, False, True],
            "body": ["aaa", "aaa", "aaa"],
            "created_at": [_t(-1), _t(+1), _t(+2)],
        }
    )
    pr_reviews = pd.DataFrame(
        {
            "review_id": [0, 1, 2],
            "pr_id": [0, 0, 0],
            "repo_id": [0, 0, 0],
            "reviewer_id": [0, 1, 2],
            "state": ["approved", "changes_requested", "commented"],
            # rev0 is visible and dismissed inside the window -> blanked.
            # rev1 is submitted inside the window -> dropped, and it is the
            # pr_review label for user 1.
            "submitted_at": [_t(-3), _t(+2), _t(+3)],
            "first_seen_at": [_t(-3), _t(+2), _t(+3)],
            "dismissed_at": [_t(+1), pd.NaT, pd.NaT],
        }
    )
    pr_review_comments = pd.DataFrame(
        {
            "comment_id": [0, 1, 2],
            "pr_id": [0, 0, 0],
            "repo_id": [0, 0, 0],
            "author_id": [0, 1, 2],
            "path": ["aa/aa/aaa.aa", "aa.aa", "aa/aa"],
            "diff_hunk": ["@@ -0,0 +0,0 @@", "aaaa", "aa"],
            "created_at": [_t(-5), _t(+1), _t(+2)],
        }
    )
    pushes = pd.DataFrame(
        {
            "push_id": [0, 1, 2],
            "user_id": [1, 1, 2],
            "repo_id": [0, 0, 0],
            "created_at": [_t(-2), _t(+1), _t(+2)],
        }
    )
    return {
        "Users": users,
        "Repositories": repositories,
        "PullRequests": pull_requests,
        "Issues": issues,
        "IssueComments": issue_comments,
        "PRReviews": pr_reviews,
        "PRReviewComments": pr_review_comments,
        "Pushes": pushes,
    }


@pytest.fixture()
def tables() -> dict[str, pd.DataFrame]:
    return make_tables()


@pytest.fixture()
def features(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return mask_future_leakage(tables, CUTOFF)


@pytest.fixture()
def labels(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return extract_labels(tables, CUTOFF, END, tables["Users"])


# ------------------------------------------------------------- table contract


def test_row_time_covers_all_eight_tables() -> None:
    assert set(ROW_TIME) == {
        "Users",
        "Repositories",
        "PullRequests",
        "Issues",
        "IssueComments",
        "PRReviews",
        "PRReviewComments",
        "Pushes",
    }
    assert ROW_TIME["PullRequests"] == "first_seen_at"
    assert ROW_TIME["PRReviews"] == "first_seen_at"
    assert ROW_TIME["Pushes"] == "created_at"


def test_future_columns_match_the_spec() -> None:
    assert FUTURE_COLS["PullRequests"] == ["merged_at", "closed_at"]
    assert FUTURE_COLS["Issues"] == ["closed_at"]
    assert FUTURE_COLS["PRReviews"] == ["dismissed_at"]


def test_label_columns_match_the_six_activities() -> None:
    assert LABEL_COLS == [
        "open_pr",
        "open_issue",
        "issue_comment",
        "pr_review",
        "pr_review_comment",
        "push",
    ]
    assert [entry[0] for entry in ACTIVITY_LABELS] == LABEL_COLS


def test_pr_review_label_is_timed_by_submitted_at() -> None:
    # `pr_review` must use the event time, `first_seen_at` is the row time
    pr_review = next(entry for entry in ACTIVITY_LABELS if entry[0] == "pr_review")
    assert pr_review[3] == "submitted_at"


# ------------------------------------------------------- mask_future_leakage


@pytest.mark.parametrize("table", list(ROW_TIME))
def test_feature_view_has_no_row_at_or_after_cutoff(
    features: dict[str, pd.DataFrame], table: str
) -> None:
    col = ROW_TIME[table]
    assert int((features[table][col] >= CUTOFF).sum()) == 0


def test_feature_view_drops_rows_at_the_boundary(features: dict[str, pd.DataFrame]) -> None:
    # a row stamped exactly at the cutoff is not visible at the cutoff
    modified = make_tables()
    modified["Pushes"].loc[0, "created_at"] = CUTOFF
    masked = mask_future_leakage(modified, CUTOFF)
    assert int((masked["Pushes"]["created_at"] == CUTOFF).sum()) == 0
    assert len(masked["Pushes"]) == 0


def test_feature_view_keeps_strict_past_rows(features: dict[str, pd.DataFrame]) -> None:
    assert len(features["Pushes"]) == 1  # only the row at cutoff - 2d
    assert len(features["Repositories"]) == 1
    assert len(features["IssueComments"]) == 1


def test_future_values_are_blanked(features: dict[str, pd.DataFrame]) -> None:
    pr = features["PullRequests"]
    # pr0 is visible but merged inside the window -> value and companion blanked
    assert pd.isna(pr.loc[0, "merged_at"])
    assert pd.isna(pr.loc[0, "merged_by_id"])
    # pr2 was never merged -> still missing, not invented
    assert pd.isna(pr.loc[2, "merged_at"])

    issues = features["Issues"]
    assert pd.isna(issues.loc[0, "closed_at"])  # closed inside the window
    assert pd.isna(issues.loc[0, "state_reason"])

    reviews = features["PRReviews"]
    assert pd.isna(reviews.loc[0, "dismissed_at"])  # dismissed inside the window

    # state known before the cutoff is kept
    assert reviews.loc[0, "state"] == "approved"


def test_no_future_value_survives_any_future_column(features: dict[str, pd.DataFrame]) -> None:
    for table, cols in FUTURE_COLS.items():
        for col in cols:
            assert int((features[table][col] >= CUTOFF).sum()) == 0


def test_mask_does_not_mutate_the_input(tables: dict[str, pd.DataFrame]) -> None:
    before = {name: df.copy() for name, df in tables.items()}
    mask_future_leakage(tables, CUTOFF)
    for name, df in tables.items():
        pd.testing.assert_frame_equal(df, before[name])


def test_mask_tolerates_missing_tables_and_columns() -> None:
    masked = mask_future_leakage({"Users": make_tables()["Users"]}, CUTOFF)
    assert set(masked) == {"Users"}
    assert len(masked["Users"]) == 4


def test_mask_keeps_unknown_tables_untouched() -> None:
    extra = pd.DataFrame({"x": [1, 2]})
    masked = mask_future_leakage({"Extra": extra}, CUTOFF)
    pd.testing.assert_frame_equal(masked["Extra"], extra)


# --------------------------------------------------------------- extract_labels


def test_label_shape_and_columns(labels: pd.DataFrame) -> None:
    assert list(labels.columns) == ["user_id", "UserID", *LABEL_COLS]
    assert len(labels) == 4
    assert labels["UserID"].tolist() == [10, 11, 12, 13]


def test_labels_are_binary(labels: pd.DataFrame) -> None:
    for col in LABEL_COLS:
        assert set(labels[col].unique()) <= {0, 1}


def test_open_pr_uses_window_and_created_at(labels: pd.DataFrame) -> None:
    # PR 1 was first seen before the cutoff but created inside the window -> label
    assert labels.loc[labels.user_id == 1, "open_pr"].item() == 1
    # PR 0 was created before the window -> no label
    assert labels.loc[labels.user_id == 0, "open_pr"].item() == 0


def test_pr_review_uses_submitted_at(labels: pd.DataFrame) -> None:
    # user 1 reviewed in the window; the row was already visible before it
    assert labels.loc[labels.user_id == 1, "pr_review"].item() == 1
    assert labels.loc[labels.user_id == 0, "pr_review"].item() == 0


def test_open_issue_excludes_pr_conversations(labels: pd.DataFrame) -> None:
    # user 1's only issue in the window is a PR conversation (issue 1)
    assert labels.loc[labels.user_id == 1, "open_issue"].item() == 0
    # user 2's issue is a plain issue with a missing flag (pd.NA) -> counted
    assert labels.loc[labels.user_id == 2, "open_issue"].item() == 1


def test_label_window_is_half_open(tables: dict[str, pd.DataFrame]) -> None:
    modified = {name: df.copy() for name, df in tables.items()}
    modified["Pushes"]["created_at"] = [CUTOFF, END, END + pd.Timedelta(days=1)]
    labeled = extract_labels(modified, CUTOFF, END, modified["Users"])
    # event exactly at the cutoff counts, at `end` does not
    assert labeled.loc[labeled.user_id == 1, "push"].item() == 1
    assert labeled.loc[labeled.user_id == 2, "push"].item() == 0


def test_labels_ignore_nat_timestamps(tables: dict[str, pd.DataFrame]) -> None:
    modified = {name: df.copy() for name, df in tables.items()}
    modified["Pushes"]["created_at"] = [pd.NaT, pd.NaT, pd.NaT]
    labeled = extract_labels(modified, CUTOFF, END, modified["Users"])
    assert int(labeled["push"].sum()) == 0


def test_labels_ignore_rows_without_a_user() -> None:
    modified = make_tables()
    modified["Pushes"]["user_id"] = [np.nan, np.nan, np.nan]
    labeled = extract_labels(modified, CUTOFF, END, modified["Users"])
    assert int(labeled["push"].sum()) == 0


def test_labels_survive_a_missing_table(tables: dict[str, pd.DataFrame]) -> None:
    partial = {k: v for k, v in tables.items() if k != "Pushes"}
    labeled = extract_labels(partial, CUTOFF, END, tables["Users"])
    assert labeled["push"].tolist() == [0, 0, 0, 0]
    assert labeled.loc[labeled.user_id == 1, "pr_review"].item() == 1


def test_labels_from_the_masked_view_are_all_zero(
    features: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]
) -> None:
    # this is the trap: the label window is empty once the rows are masked
    naive = extract_labels(features, CUTOFF, END, tables["Users"])
    assert int(naive[LABEL_COLS].sum().sum()) == 0


# ----------------------------------------------------------------- build_split


def test_build_split_window(tables: dict[str, pd.DataFrame]) -> None:
    split = build_split(tables, CUTOFF, days=7)
    assert split.cutoff == CUTOFF
    assert split.end == END
    assert split.end - split.cutoff == pd.Timedelta(days=7)


def test_build_split_keeps_cohort_users_only(tables: dict[str, pd.DataFrame]) -> None:
    split = build_split(tables, CUTOFF)
    assert split.cohort["user_id"].tolist() == [0, 1, 3]  # user 2 is not cohort
    assert split.labels["user_id"].tolist() == [0, 1, 3]
    assert set(LABEL_COLS) <= set(split.labels.columns)


def test_cohort_users_falls_back_to_all_users() -> None:
    users = pd.DataFrame({"user_id": [0, 1], "github_user_id": [10, 11]})
    assert len(cohort_users({"Users": users})) == 2


def test_data_window(tables: dict[str, pd.DataFrame]) -> None:
    earliest, latest = data_window(tables)
    assert earliest == _t(-100)
    # latest row time is Issues.first_seen_at; pr1.closed_at (+6) is not a row time
    assert latest == _t(+4)
    assert latest < _t(+6)


# ------------------------------------------------------------ leakage_failures


def test_clean_split_has_no_failures(tables: dict[str, pd.DataFrame]) -> None:
    # make_tables contains a deliberate clock-skew row, so remove it first
    clean = {name: df.copy() for name, df in tables.items()}
    clean["PRReviews"].loc[1, ["submitted_at", "first_seen_at"]] = [_t(+2), _t(+2)]
    assert leakage_failures(clean, build_split(clean, CUTOFF)) == []


def test_detects_clock_skew_in_the_label_window() -> None:
    # impossible in real data (you cannot see a review before it is submitted),
    # which is exactly why the check is worth having
    skewed = {name: df.copy() for name, df in make_tables().items()}
    skewed["PRReviews"].loc[1, ["submitted_at", "first_seen_at"]] = [_t(+2), _t(-4)]
    split = build_split(skewed, CUTOFF)
    failures = leakage_failures(skewed, split)
    assert any("PRReviews" in f and "visible in the feature view" in f for f in failures)


def test_detects_clock_skew_between_created_at_and_first_seen_at() -> None:
    skewed = {name: df.copy() for name, df in make_tables().items()}
    skewed["PullRequests"].loc[1, ["created_at", "first_seen_at"]] = [_t(+1), _t(-1)]
    split = build_split(skewed, CUTOFF)
    failures = leakage_failures(skewed, split)
    assert any("PullRequests" in f and "visible in the feature view" in f for f in failures)


def test_detects_row_time_leak(tables: dict[str, pd.DataFrame]) -> None:
    split = build_split(tables, CUTOFF)
    split.features["Pushes"].loc[0, "created_at"] = _t(+1)
    failures = leakage_failures(tables, split)
    assert any("Pushes.created_at" in f for f in failures)


def test_detects_an_unmasked_future_column(tables: dict[str, pd.DataFrame]) -> None:
    split = build_split(tables, CUTOFF)
    split.features["Issues"].loc[0, "closed_at"] = _t(+2)
    failures = leakage_failures(tables, split)
    assert any("Issues.closed_at" in f for f in failures)


def test_detects_all_zero_labels(tables: dict[str, pd.DataFrame]) -> None:
    split = build_split(tables, CUTOFF)
    split.labels[LABEL_COLS] = 0
    failures = leakage_failures(tables, split)
    assert any("every user has 0 positives" in f for f in failures)


def test_detects_an_inverted_window(tables: dict[str, pd.DataFrame]) -> None:
    split = build_split(tables, CUTOFF)
    inverted = split._replace(end=cast(pd.Timestamp, split.cutoff - pd.Timedelta(days=1)))
    failures = leakage_failures(tables, inverted)
    assert any("is not after the cutoff" in f for f in failures)


def test_hidden_counts_report_masked_values(tables: dict[str, pd.DataFrame]) -> None:
    split = build_split(tables, CUTOFF)
    counts = hidden_counts(tables, split)
    assert counts["PullRequests.merged_at"] >= 1  # PR 1 merged in the window
    assert counts["Issues.closed_at"] >= 1
    assert counts["PRReviews.dismissed_at"] == 1


# --------------------------------------------------------------------- outputs


def test_write_split_writes_three_artefacts(
    tables: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    split = build_split(tables, CUTOFF)
    write_split(tmp_path, split)

    assert (tmp_path / "answer.csv").is_file()
    assert (tmp_path / "all_zero_submission.csv").is_file()
    written = sorted(p.stem for p in (tmp_path / "features").glob("*.parquet"))
    assert written == sorted(tables)

    answer = pd.read_csv(tmp_path / "answer.csv")
    assert list(answer.columns) == ["UserID", *LABEL_COLS]
    assert len(answer) == len(split.labels)

    baseline = pd.read_csv(tmp_path / "all_zero_submission.csv")
    assert int(baseline[LABEL_COLS].to_numpy().sum()) == 0
    assert baseline["UserID"].tolist() == answer["UserID"].tolist()


def _write_synthetic_db(root: Path) -> None:
    db = root / "db"
    db.mkdir(parents=True)
    for name, df in make_tables().items():
        df.to_parquet(db / f"{name}.parquet")


def test_main_writes_a_clean_split(tmp_path: Path) -> None:
    _write_synthetic_db(tmp_path)
    out = tmp_path / "splits"
    code = main(
        ["--db", str(tmp_path / "db"), "--cutoff", "2024-06-01", "--days", "7",
         "--out", str(out)]
    )
    assert code == 0
    assert (out / "answer.csv").is_file()
    assert (out / "features" / "Users.parquet").is_file()
    assert (out / "all_zero_submission.csv").is_file()


def test_main_refuses_to_write_a_leaky_split(tmp_path: Path) -> None:
    db = tmp_path / "db"
    db.mkdir()
    tables = make_tables()
    tables["PRReviews"].loc[1, ["submitted_at", "first_seen_at"]] = [_t(+2), _t(-4)]
    for name, df in tables.items():
        df.to_parquet(db / f"{name}.parquet")

    out = tmp_path / "splits"
    assert main(["--db", str(db), "--cutoff", "2024-06-01", "--out", str(out)]) == 1
    assert not (out / "answer.csv").exists()


def test_main_reports_a_missing_database(tmp_path: Path) -> None:
    assert main(["--db", str(tmp_path / "absent"), "--out", str(tmp_path / "s")]) == 1


def test_parse_cutoff_rejects_aware_timestamps() -> None:
    assert parse_cutoff("2026-10-24") == CUTOFF.replace(year=2026, month=10, day=24)
    with pytest.raises(ValueError, match="tz-naive"):
        parse_cutoff("2026-10-24T00:00:00+00:00")


# ------------------------------------------------- the six viable text features


def test_six_structural_features_are_non_null(
    tables: dict[str, pd.DataFrame], features: dict[str, pd.DataFrame]
) -> None:
    """The six features with enough non-null data to be viable."""
    present = {
        "log_issue_len": ("Issues", "body"),
        "has_markdown_structure": ("Issues", "body"),
        "has_prefix": ("Issues", "title"),
        "branch_depth": ("PullRequests", "head_ref"),
        "log_diff_len": ("PRReviewComments", "diff_hunk"),
        "path_depth": ("PRReviewComments", "path"),
    }
    for table, column in present.values():
        assert column in features[table].columns
        assert bool(features[table][column].notna().all()), f"{table}.{column}"


def test_six_structural_features_are_derived_correctly(features: dict[str, pd.DataFrame]) -> None:
    """The mask must leave the six source columns untouched, so they stay derivable."""
    issues = features["Issues"]
    prc = features["PRReviewComments"]
    pr = features["PullRequests"]

    # 1. log_issue_len = log(1 + character count of the body)
    assert float(np.log1p(len(issues["body"].iloc[0]))) == pytest.approx(np.log1p(8.0))

    # 2. has_markdown_structure: a heading line is a hit
    assert bool(issues["body"].iloc[0].startswith("###"))

    # 3. has_prefix: the masked "scope(scope):" shape survives masking
    assert bool(issues["title"].iloc[0].startswith("aaa(aaa):"))

    # 4. branch_depth = number of "/" in head_ref
    assert (pr["head_ref"].astype(str).str.count("/")).tolist() == [1, 2]

    # 5. log_diff_len = log(1 + character count of the diff hunk)
    hunk = prc["diff_hunk"].iloc[0]
    assert hunk == "@@ -0,0 +0,0 @@"
    assert float(np.log1p(len(hunk))) == pytest.approx(np.log1p(15.0))

    # 6. path_depth = number of "/" in path
    assert (prc["path"].astype(str).str.count("/")).tolist() == [2]


# ------------------------------------------------------------- real data smoke


def _real_tables() -> dict[str, pd.DataFrame]:
    db = Path("processed_tables/db")
    if not db.is_dir() or not list(db.glob("*.parquet")):
        pytest.skip("processed_tables/db/*.parquet is not available")
    return load_tables(str(db))


def test_real_data_split_is_clean() -> None:
    tables = _real_tables()
    real_cutoff = cast(pd.Timestamp, pd.Timestamp("2026-10-24 00:00:00"))
    split = build_split(tables, real_cutoff, days=7)

    assert leakage_failures(tables, split) == []

    # feature view is strictly before the cutoff
    for table, col in ROW_TIME.items():
        assert int((split.features[table][col] >= real_cutoff).sum()) == 0

    # labels come from the raw tables and are not degenerate
    assert int(split.labels[LABEL_COLS].sum().sum()) > 0
    for col in LABEL_COLS:
        assert int(split.labels[col].sum()) > 0, col

    # cohort users only
    assert split.labels["user_id"].is_unique
    assert len(split.labels) == len(split.cohort)


def test_real_data_structural_columns_are_covered() -> None:
    tables = _real_tables()
    real_cutoff = cast(pd.Timestamp, pd.Timestamp("2026-10-24 00:00:00"))
    features = mask_future_leakage(tables, real_cutoff)

    # the six features are the ones with enough non-null data to be viable
    coverage = {
        "Issues.body": features["Issues"]["body"].notna().mean(),
        "Issues.title": features["Issues"]["title"].notna().mean(),
        "PullRequests.head_ref": features["PullRequests"]["head_ref"].notna().mean(),
        "PRReviewComments.diff_hunk": features["PRReviewComments"]["diff_hunk"].notna().mean(),
        "PRReviewComments.path": features["PRReviewComments"]["path"].notna().mean(),
    }
    for column, ratio in coverage.items():
        assert ratio >= 0.95, f"{column} coverage {ratio:.3f} is too low"
