"""Automated checks for `src.features.build_graph`.

What is verified:
  1. Users node: 21 recency/frequency/profile features, correct semantics
     (days_since clip at 30, has_ever flags, 7-day half-open count window,
     is_bot / is_cohort / user_age_days).
  2. Every event node carries its age relative to the cutoff, with the
     spec dimension per node type.
  3. Repositories -> Users `owner` edges come from the owner_login -> login
     join, not from a repo_id row number, and org-owned repositories get no
     owner edge.
  4. Every relation has one typed reverse (`rev_*`) edge with transposed
     edge_index.
  5. No NaN in any node feature tensor.

Run from the repository root:
    python -m pytest tests/test_build_graph.py -q
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.build_graph import build_hetero_graph, extract_user_behavior_features

CUTOFF = cast(pd.Timestamp, pd.Timestamp("2024-06-01 00:00:00"))

ACTIVITIES = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
]

USER_STATIC_FEATURES = ["is_bot", "is_cohort", "user_age_days"]

# node type -> (expected feature dim, DataFrame key holding its age column)
EVENT_AGE_SPEC = {
    "Repositories": (2, "Repositories"),
    "PullRequests": (4, "PullRequests"),
    "Issues": (6, "Issues"),
    "IssueComments": (2, "IssueComments"),
    "PRReviews": (3, "PRReviews"),
    "PRReviewComments": (3, "PRReviewComments"),
    "Pushes": (1, "Pushes"),
}


def _days_ago(days: float) -> pd.Timestamp:
    return cast(pd.Timestamp, CUTOFF - pd.Timedelta(days=days))


def make_tables() -> dict[str, pd.DataFrame]:
    """Small hand-checkable snapshot of the eight tables, all rows < CUTOFF."""
    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 3],
            "github_user_id": [10, 11, 12, 13],
            "login": ["alice", "bob", "carol", "dave"],
            "is_bot": [False, True, False, False],
            "is_cohort": [True, False, True, False],
            "first_seen_at": [_days_ago(100), _days_ago(10), _days_ago(50), _days_ago(30)],
        }
    )

    repositories = pd.DataFrame(
        {
            "repo_id": [0, 1, 2],
            "github_repo_id": [100, 101, 102],
            # repo 0 and repo 2 are user-owned (owner is a Users row),
            # repo 1 is organization-owned (owner_login has no Users row).
            "name": ["alice/aaaa", "aaaa/aaaa", "bob/aaaa"],
            "owner_login": ["alice", "ghost-org", "bob"],
            "org_id": [np.nan, 9001.0, np.nan],
            "org_login": [np.nan, "ghost-org", np.nan],
            "first_seen_at": [_days_ago(90), _days_ago(20), _days_ago(5)],
        }
    )

    pull_requests = pd.DataFrame(
        {
            "pr_id": [0, 1],
            "github_pr_id": [1000, 1001],
            "repo_id": [0, 2],
            "author_id": [0, 1],
            "draft": [False, True],
            "head_ref": ["aaaa/aaaa", "aaa"],
            "is_cross_repo": [False, True],
            "first_seen_at": [_days_ago(3), _days_ago(2)],
        }
    )

    issues = pd.DataFrame(
        {
            "issue_id": [0, 1, 2],
            "github_issue_id": [2000, 2001, 2002],
            "repo_id": [0, 0, 2],
            "author_id": [0, 0, 1],
            # issue 1 is the PR conversation, so it must not count as open_issue
            "is_pull_request": [False, True, False],
            "num_labels": [2.0, 0.0, 1.0],
            "title": ["aaa(aaa): aaaaa", "aaaa", "aaaa: aa"],
            "body": ["### aaaaa", "aaaa", "aaa `aa` aa"],
            "first_seen_at": [_days_ago(10), _days_ago(1), _days_ago(4)],
        }
    )

    issue_comments = pd.DataFrame(
        {
            "comment_id": [0, 1, 2, 3],
            "issue_id": [0, 0, 0, 1],
            "repo_id": [0, 0, 0, 0],
            "author_id": [0, 0, 0, 1],
            "is_pull_request": [False, False, False, True],
            # boundary row at exactly cutoff - 7d (included), one 8 days old
            # (excluded), and one at the cutoff itself (must be excluded).
            "created_at": [_days_ago(7), _days_ago(2), _days_ago(8), CUTOFF],
        }
    )

    pr_reviews = pd.DataFrame(
        {
            "review_id": [0, 1],
            "pr_id": [0, 0],
            "repo_id": [0, 0],
            "reviewer_id": [0, 1],
            "state": ["approved", "changes_requested"],
            "first_seen_at": [_days_ago(40), _days_ago(1)],
        }
    )

    pr_review_comments = pd.DataFrame(
        {
            "comment_id": [0, 1],
            "pr_id": [0, 1],
            "repo_id": [0, 2],
            "author_id": [0, 1],
            "path": ["aa/aa/aaa.aa", "aa.aa"],
            "diff_hunk": ["@@ -0,0 +0,0 @@", "aaaa"],
            "created_at": [_days_ago(5), _days_ago(3)],
        }
    )

    pushes = pd.DataFrame(
        {
            "push_id": [0, 1, 2],
            "user_id": [1, 1, 1],
            "repo_id": [2, 2, 2],
            "created_at": [_days_ago(1), _days_ago(2), _days_ago(9)],
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


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return make_tables()


@pytest.fixture(scope="module")
def data(tables: dict[str, pd.DataFrame]):
    return build_hetero_graph(tables, CUTOFF)


@pytest.fixture(scope="module")
def user_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return extract_user_behavior_features(tables, CUTOFF)


# ---------------------------------------------------------------- user features


def test_user_feature_dimension_is_21(user_features: pd.DataFrame) -> None:
    assert user_features.shape == (4, 21)


def test_user_feature_names_match_spec(user_features: pd.DataFrame) -> None:
    expected = []
    for act in ACTIVITIES:
        expected += [f"days_since_{act}", f"has_ever_{act}", f"count_7d_{act}"]
    expected += USER_STATIC_FEATURES
    assert list(user_features.columns) == expected


def test_user_features_have_no_nan(user_features: pd.DataFrame, data) -> None:
    assert int(user_features.isna().sum().sum()) == 0
    x = data["Users"].x
    assert x.dtype == torch.float32
    assert x.shape == (4, 21)
    assert int(torch.isnan(x).sum()) == 0


@pytest.mark.parametrize(
    ("user_idx", "act", "days_since"),
    [
        (0, "open_pr", 3.0),
        (0, "open_issue", 10.0),
        (0, "issue_comment", 2.0),
        (0, "pr_review", 30.0),  # 40 days old -> clipped at 30
        (0, "pr_review_comment", 5.0),
        (0, "push", 30.0),  # never active -> default 30
        (1, "push", 1.0),
        (1, "issue_comment", 0.0),  # only event is at the cutoff -> lower clip
        (2, "open_pr", 30.0),
        (3, "push", 30.0),
    ],
)
def test_days_since(
    user_features: pd.DataFrame, user_idx: int, act: str, days_since: float
) -> None:
    assert float(user_features.loc[user_idx, f"days_since_{act}"]) == pytest.approx(days_since)


@pytest.mark.parametrize(
    ("user_idx", "act", "has_ever", "count_7d"),
    [
        (0, "open_pr", 1.0, 1.0),
        (0, "open_issue", 1.0, 0.0),  # 10 days old, outside the window
        (0, "issue_comment", 1.0, 2.0),  # 7d boundary in, 8d out, cutoff out
        (0, "pr_review", 1.0, 0.0),
        (0, "pr_review_comment", 1.0, 1.0),
        (0, "push", 0.0, 0.0),
        (1, "push", 1.0, 2.0),  # 1d and 2d in, 9d out
        (2, "issue_comment", 0.0, 0.0),
    ],
)
def test_has_ever_and_count_7d(
    user_features: pd.DataFrame, user_idx: int, act: str, has_ever: float, count_7d: float
) -> None:
    assert float(user_features.loc[user_idx, f"has_ever_{act}"]) == pytest.approx(has_ever)
    assert float(user_features.loc[user_idx, f"count_7d_{act}"]) == pytest.approx(count_7d)


def test_count_7d_window_is_half_open(user_features: pd.DataFrame) -> None:
    # comment at exactly cutoff - 7d is counted, comment at the cutoff is not
    assert float(user_features.loc[0, "count_7d_issue_comment"]) == 2.0
    assert float(user_features.loc[1, "count_7d_issue_comment"]) == 0.0


def test_open_issue_excludes_pr_conversations(user_features: pd.DataFrame) -> None:
    # user 0 has an Issues row at 1 day old but it is a PR conversation (issue 1)
    assert float(user_features.loc[0, "days_since_open_issue"]) == pytest.approx(10.0)
    assert float(user_features.loc[0, "count_7d_open_issue"]) == 0.0


def test_user_static_profile_features(user_features: pd.DataFrame) -> None:
    assert user_features["is_bot"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert user_features["is_cohort"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert user_features["user_age_days"].round(6).tolist() == [100.0, 10.0, 50.0, 30.0]


# ------------------------------------------------------------------ event ages


def test_all_eight_node_types_are_present(data) -> None:
    assert set(data.node_types) == {"Users", *EVENT_AGE_SPEC}
    assert len(data.node_types) == 8


@pytest.mark.parametrize(("node_type", "spec"), list(EVENT_AGE_SPEC.items()))
def test_event_node_dimensions(data, node_type: str, spec: tuple[int, str]) -> None:
    expected_dim = spec[0]
    x = data[node_type].x
    assert x.shape[1] == expected_dim
    assert x.dtype == torch.float32


@pytest.mark.parametrize("node_type", ["Users", *EVENT_AGE_SPEC])
def test_no_nan_in_any_node_feature(data, node_type: str) -> None:
    x = data[node_type].x
    assert int(torch.isnan(x).sum()) == 0
    assert int(torch.isinf(x).sum()) == 0
    assert bool(torch.isfinite(x).all())


@pytest.mark.parametrize(
    ("node_type", "row_idx", "age"),
    [
        ("Repositories", 0, 90.0),
        ("Repositories", 2, 5.0),
        ("PullRequests", 0, 3.0),
        ("PullRequests", 1, 2.0),
        ("Issues", 0, 10.0),
        ("Issues", 2, 4.0),
        ("IssueComments", 0, 7.0),
        ("IssueComments", 3, 0.0),
        ("PRReviews", 0, 40.0),
        ("PRReviews", 1, 1.0),
        ("PRReviewComments", 0, 5.0),
        ("Pushses", 0, 1.0),
    ],
)
def test_event_age_values(data, node_type: str, row_idx: int, age: float) -> None:
    node_type = "Pushes" if node_type == "Pushses" else node_type
    assert float(data[node_type].x[row_idx, 0]) == pytest.approx(age)


def test_event_age_column_matches_cutoff_delta(tables: dict[str, pd.DataFrame], data) -> None:
    for node_type, (_, table_name) in EVENT_AGE_SPEC.items():
        df = tables[table_name]
        time_col = "created_at" if "created_at" in df.columns else "first_seen_at"
        expected = (CUTOFF - df[time_col]).dt.total_seconds().to_numpy() / 86400.0
        got = data[node_type].x[:, 0].numpy()
        np.testing.assert_allclose(got, np.clip(expected, 0.0, None), rtol=0, atol=1e-5)


def test_structural_features_are_derived(data) -> None:
    repo_is_org = data["Repositories"].x[:, 1].tolist()
    assert repo_is_org == [0.0, 1.0, 0.0]

    pr = data["PullRequests"].x
    assert pr[:, 1].tolist() == [0.0, 1.0]  # draft
    assert pr[:, 2].tolist() == [0.0, 1.0]  # is_cross_repo
    assert pr[:, 3].tolist() == [1.0, 0.0]  # branch depth = count of "/"

    iss = data["Issues"].x
    assert iss[:, 1].tolist() == [0.0, 1.0, 0.0]  # is_pull_request
    assert iss[:, 2].tolist() == [2.0, 0.0, 1.0]  # num_labels
    assert iss[:, 4].tolist() == [1.0, 0.0, 1.0]  # markdown header / code
    assert iss[:, 5].tolist() == [1.0, 0.0, 1.0]  # conventional prefix

    rev = data["PRReviews"].x
    assert rev[:, 1].tolist() == [1.0, 0.0]  # approved
    assert rev[:, 2].tolist() == [0.0, 1.0]  # changes_requested

    prc = data["PRReviewComments"].x
    assert prc[:, 2].tolist() == [2.0, 0.0]  # path depth


def test_stack_f32_sanitises_values() -> None:
    from src.features.build_graph import _stack_f32

    out = _stack_f32(np.array([np.nan, 1.5]), np.array([-2.0, 3.0]), np.array([0, 1]))
    assert out.dtype == np.float32
    assert out.shape == (2, 3)
    assert out[0].tolist() == [0.0, 0.0, 0.0]
    assert out[1].tolist() == [1.5, 3.0, 1.0]


def test_nat_timestamps_do_not_produce_nan(tables: dict[str, pd.DataFrame]) -> None:
    modified = {k: v.copy() for k, v in tables.items()}
    modified["Repositories"].loc[0, "first_seen_at"] = pd.NaT
    modified["Users"].loc[3, "first_seen_at"] = pd.NaT
    modified["Issues"].loc[2, "num_labels"] = np.nan
    # pandas 3 refuses to store None in a bool column, so widen first. The test
    # only cares that a missing `draft` degrades to 0.0, not about its dtype.
    pr = modified["PullRequests"]
    pr["draft"] = pr["draft"].astype(object)
    pr.loc[1, "draft"] = None

    graph = build_hetero_graph(modified, CUTOFF)

    for node_type in ["Users", *EVENT_AGE_SPEC]:
        x = graph[node_type].x
        assert int(torch.isnan(x).sum()) == 0, node_type
    # NaT age degrades to 0.0, not NaN
    assert float(graph["Repositories"].x[0, 0]) == 0.0
    assert float(graph["Users"].x[3, 20]) == 0.0  # user_age_days of user 3
    assert float(graph["Issues"].x[2, 2]) == 0.0  # missing num_labels
    assert float(graph["PullRequests"].x[1, 1]) == 0.0  # missing draft


def test_markdown_flag_ignores_masked_cross_reference(tables: dict[str, pd.DataFrame]) -> None:
    modified = {k: v.copy() for k, v in tables.items()}
    modified["Issues"].loc[0, "body"] = "aaa aaa #000 aaaa"  # masked "#123"
    modified["Issues"].loc[1, "body"] = "```\naaaa\n```"
    graph = build_hetero_graph(modified, CUTOFF)
    assert float(graph["Issues"].x[0, 4]) == 0.0
    assert float(graph["Issues"].x[1, 4]) == 1.0


# ----------------------------------------------------------- owner edges (fixed)


def test_owner_edges_join_login_not_repo_row_number(data) -> None:
    edge_index = data["Repositories", "owner", "Users"].edge_index
    # repo 0 -> user 0 (alice), repo 2 -> user 1 (bob)
    assert edge_index.shape == (2, 2)
    assert edge_index[0].tolist() == [0, 2]
    assert edge_index[1].tolist() == [0, 1]


def test_org_owned_repository_has_no_owner_edge(data) -> None:
    edge_index = data["Repositories", "owner", "Users"].edge_index
    sources = set(edge_index[0].tolist())
    assert 1 not in sources  # repo 1 owner_login "ghost-org" is not a Users row


def test_owner_edge_count_is_not_user_count(data) -> None:
    # regression guard for the old repo_id-as-user fallback, which produced one
    # edge per user regardless of ownership
    num_edges = data["Repositories", "owner", "Users"].edge_index.shape[1]
    assert num_edges == 2
    assert num_edges != data["Users"].num_nodes


def test_owner_edges_resolve_to_matching_logins(tables: dict[str, pd.DataFrame], data) -> None:
    edge_index = data["Repositories", "owner", "Users"].edge_index
    repos = tables["Repositories"]
    users = tables["Users"]
    for pos in range(edge_index.shape[1]):
        repo_row = int(edge_index[0, pos])
        user_row = int(edge_index[1, pos])
        assert repos["owner_login"].iloc[repo_row] == users["login"].iloc[user_row]


# -------------------------------------------------------- reverse relations


def test_forward_and_reverse_edge_type_pairs(data) -> None:
    forward = [et for et in data.edge_types if not et[1].startswith("rev_")]
    reverse = [et for et in data.edge_types if et[1].startswith("rev_")]
    assert len(forward) == 13
    assert len(forward) == len(reverse)
    for src, rel, dst in forward:
        assert (dst, f"rev_{rel}", src) in data.edge_types


def test_reverse_edges_are_transposed(data) -> None:
    for src, rel, dst in data.edge_types:
        if rel.startswith("rev_"):
            continue
        forward_ei = data[src, rel, dst].edge_index
        reverse_ei = data[dst, f"rev_{rel}", src].edge_index
        assert reverse_ei.shape == forward_ei.shape
        assert torch.equal(reverse_ei, forward_ei.flip(0))


@pytest.mark.parametrize(
    ("edge_type", "expected"),
    [
        (("Repositories", "owner", "Users"), [(0, 0), (2, 1)]),
        (("PullRequests", "authored_by", "Users"), [(0, 0), (1, 1)]),
        (("PullRequests", "belongs_to", "Repositories"), [(0, 0), (1, 2)]),
        (("Issues", "authored_by", "Users"), [(0, 0), (1, 0), (2, 1)]),
        (("Issues", "belongs_to", "Repositories"), [(0, 0), (1, 0), (2, 2)]),
        (("IssueComments", "authored_by", "Users"), [(0, 0), (1, 0), (2, 0), (3, 1)]),
        (("IssueComments", "comment_on", "Issues"), [(0, 0), (1, 0), (2, 0), (3, 1)]),
        (("PRReviews", "reviewed_by", "Users"), [(0, 0), (1, 1)]),
        (("PRReviews", "review_of", "PullRequests"), [(0, 0), (1, 0)]),
        (("PRReviewComments", "authored_by", "Users"), [(0, 0), (1, 1)]),
        (("PRReviewComments", "comment_on", "PullRequests"), [(0, 0), (1, 1)]),
        (("Pushes", "pushed_by", "Users"), [(0, 1), (1, 1), (2, 1)]),
        (("Pushes", "pushed_to", "Repositories"), [(0, 2), (1, 2), (2, 2)]),
    ],
)
def test_foreign_key_edges(data, edge_type, expected) -> None:
    edge_index = data[edge_type].edge_index
    got = list(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    assert got == expected


def test_edge_indices_are_in_node_range(data) -> None:
    for src, rel, dst in data.edge_types:
        edge_index = data[src, rel, dst].edge_index
        assert edge_index.dtype == torch.long
        assert edge_index.shape[0] == 2
        if edge_index.numel() == 0:
            continue
        assert int(edge_index[0].min()) >= 0
        assert int(edge_index[0].max()) < data[src].num_nodes
        assert int(edge_index[1].min()) >= 0
        assert int(edge_index[1].max()) < data[dst].num_nodes


# ------------------------------------------------------------- real data smoke


def _load_real_tables():
    from pathlib import Path

    db = Path("processed_tables/db")
    if not db.is_dir() or not list(db.glob("*.parquet")):
        pytest.skip("processed_tables/db/*.parquet is not available")

    from src.data.split_io import load_tables
    from src.data.temporal_split import mask_future_leakage

    raw = load_tables(str(db))
    real_cutoff = cast(pd.Timestamp, pd.Timestamp("2026-10-24 00:00:00"))
    masked = mask_future_leakage(raw, real_cutoff)
    return masked, real_cutoff


def test_real_tables_build_clean_graph() -> None:
    tables, real_cutoff = _load_real_tables()
    graph = build_hetero_graph(tables, real_cutoff)

    assert set(graph.node_types) == {"Users", *EVENT_AGE_SPEC}
    assert graph["Users"].x.shape == (len(tables["Users"]), 21)
    assert graph["Users"].num_nodes > 0

    for node_type, (expected_dim, _) in EVENT_AGE_SPEC.items():
        x = graph[node_type].x
        assert x.shape[1] == expected_dim
        assert int(torch.isnan(x).sum()) == 0

    forward = [et for et in graph.edge_types if not et[1].startswith("rev_")]
    reverse = [et for et in graph.edge_types if et[1].startswith("rev_")]
    assert len(forward) == len(reverse) == 13
    for src, rel, dst in forward:
        assert (dst, f"rev_{rel}", src) in graph.edge_types
        assert torch.equal(
            graph[dst, f"rev_{rel}", src].edge_index,
            graph[src, rel, dst].edge_index.flip(0),
        )

    # owner edges must be a real login join: unique repos, and far fewer than
    # one edge per user
    ei = graph["Repositories", "owner", "Users"].edge_index
    assert ei.shape[1] > 0
    assert len(set(ei[0].tolist())) == ei.shape[1]
    assert ei.shape[1] != graph["Users"].num_nodes
