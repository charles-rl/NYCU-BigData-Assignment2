"""Graph builder COPY for the ablation suite.

This file is a copy of ``src/features/build_graph.py``.  The base pipeline is
left untouched on purpose: ablation experiments must not change the code they
are compared against.

The copy adds two things the base builder does not have:

1. ``edge_weight`` on every edge type.  For event edges the weight is the
   exponential temporal decay from ``docs/VALIDATION_DATA.md``::

       w = exp(-min(dt_days, max_days) / tau_activity)

   where ``dt_days`` is the age of the source event at the split cutoff.  With
   ``temporal_decay=False`` all weights are exactly ``1.0``; that is the
   ``no-temporal-decay`` ablation.  Static ownership edges
   (Repositories -> Users) always keep weight 1.0 because they carry no event
   time.

2. :func:`mask_text_features`, which removes the six structural/textual
   pattern features from the node feature matrices for the ``no-text``
   ablation.  The index map is kept next to the feature construction below.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Measured on this corpus (docs/VALIDATION_DATA.md section 5 and
# IMPLEMENTATION_NOTES.md "Action Item 1").  Push uses the clean-anchor fit.
TEMPORAL_EDGE_TAU_DAYS: dict[str, float] = {
    "open_pr": 5.96,
    "open_issue": 5.37,
    "issue_comment": 6.02,
    "pr_review": 6.73,
    "pr_review_comment": 6.91,
    "push": 7.50,
}

# Which activity's tau applies to the event edges sourced from each table.
ACTIVITY_BY_EVENT_TABLE: dict[str, str] = {
    "PullRequests": "open_pr",
    "Issues": "open_issue",
    "IssueComments": "issue_comment",
    "PRReviews": "pr_review",
    "PRReviewComments": "pr_review_comment",
    "Pushes": "push",
}

# The six structural/textual pattern features, by node type and column index.
# Indices refer to the column order constructed in build_hetero_graph().
TEXT_FEATURE_INDICES: dict[str, tuple[int, ...]] = {
    "PullRequests": (3,),  # branch_depth
    "Issues": (3, 4, 5),  # log_issue_len, has_markdown_structure, has_prefix
    "PRReviewComments": (1, 2),  # log_diff_len, path_depth
}
TEXT_FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "PullRequests": ("branch_depth",),
    "Issues": ("log_issue_len", "has_markdown_structure", "has_prefix"),
    "PRReviewComments": ("log_diff_len", "path_depth"),
}
N_TEXT_FEATURES = sum(len(v) for v in TEXT_FEATURE_INDICES.values())

# Original column counts, used to make mask_text_features idempotence-safe.
ORIGINAL_FEATURE_DIMS: dict[str, int] = {
    "PullRequests": 4,
    "Issues": 6,
    "PRReviewComments": 3,
}


def _decay_weights(
    times,
    cutoff: pd.Timestamp,
    tau_days: float,
    max_days: float | None,
    enabled: bool,
) -> np.ndarray:
    """Per-edge ``exp(-dt / tau)`` weights, aligned with ``times``.

    ``dt`` is the age of the source event at ``cutoff``, in days, clipped to
    ``[0, max_days]``.  Clipping matches the 30-day horizon used by the user
    recency features and keeps very old edges at a small but non-zero weight,
    which avoids a numerically flat softmax over all-old incoming edges.

    With ``enabled=False`` the function returns exactly ones, i.e. the
    ``no-temporal-decay`` ablation.  It deliberately does not skip the tensor
    creation so callers can verify the unit weights.
    """
    n = len(times)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if not enabled:
        return np.ones(n, dtype=np.float32)
    if tau_days <= 0:
        raise ValueError(f"tau_days must be > 0, got {tau_days}")

    stamps = pd.to_datetime(pd.Series(np.asarray(times)))
    delta = (cutoff - stamps).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    fill = float(max_days) if max_days and max_days > 0 else 3650.0
    delta = np.nan_to_num(delta, nan=fill, posinf=fill, neginf=0.0)
    if max_days and max_days > 0:
        delta = np.clip(delta, 0.0, float(max_days))
    else:
        delta = np.clip(delta, 0.0, None)
    return np.exp(-delta / float(tau_days)).astype(np.float32)


def _stack_f32(*columns) -> np.ndarray:
    """Stack 1-D columns into a float32 feature matrix.

    All graph features are non-negative, so NaN becomes 0.0 and values clip at 0.
    """
    arrays = [
        np.nan_to_num(np.asarray(col, dtype=np.float64), nan=0.0)
        .clip(min=0.0)
        .astype(np.float32)
        for col in columns
    ]
    return np.stack(arrays, axis=1).astype(np.float32)


def extract_user_behavior_features(tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> pd.DataFrame:
    """Compute 21-dim recency and frequency features for every user.

    6 activities x (days_since, has_ever, count_7d) + is_bot + is_cohort +
    user_age_days = 21 features.
    """
    users_df = tables["Users"].copy()
    user_ids = users_df["user_id"].values
    n_users = len(user_ids)

    # Mapping from user_id to continuous index 0..N-1
    uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    # Activity sources: (table_name, author_col, time_col, is_pr_flag_filter)
    activities = {
        "open_pr": ("PullRequests", "author_id", "first_seen_at", None),
        "open_issue": ("Issues", "author_id", "first_seen_at", "is_issue_only"),
        "issue_comment": ("IssueComments", "author_id", "created_at", None),
        "pr_review": ("PRReviews", "reviewer_id", "first_seen_at", None),
        "pr_review_comment": ("PRReviewComments", "author_id", "created_at", None),
        "push": ("Pushes", "user_id", "created_at", None),
    }

    feature_dict = {}
    cutoff_7d = cutoff - pd.Timedelta(days=7)

    for act_name, (tbl, u_col, t_col, flag) in activities.items():
        days_since = np.full(n_users, 30.0, dtype=np.float32)
        has_ever = np.zeros(n_users, dtype=np.float32)
        count_7d = np.zeros(n_users, dtype=np.float32)

        if tbl in tables and not tables[tbl].empty:
            df = tables[tbl].dropna(subset=[u_col, t_col])
            if flag == "is_issue_only" and "is_pull_request" in df.columns:
                df = df[~df["is_pull_request"].astype(bool)]

            if not df.empty:
                # Group by user to find most recent time and count in last 7 days
                for uid, group in df.groupby(u_col):
                    if uid in uid_to_idx:
                        idx = uid_to_idx[uid]
                        most_recent = group[t_col].max()
                        delta_days = max(0.0, (cutoff - most_recent).total_seconds() / 86400.0)

                        days_since[idx] = min(30.0, delta_days)
                        has_ever[idx] = 1.0
                        # Half-open window [cutoff - 7d, cutoff)
                        window = (group[t_col] >= cutoff_7d) & (group[t_col] < cutoff)
                        count_7d[idx] = window.sum()

        feature_dict[f"days_since_{act_name}"] = days_since
        feature_dict[f"has_ever_{act_name}"] = has_ever
        feature_dict[f"count_7d_{act_name}"] = count_7d

    # Static user flags
    feature_dict["is_bot"] = tables["Users"]["is_bot"].fillna(False).astype(np.float32).values
    feature_dict["is_cohort"] = tables["Users"]["is_cohort"].fillna(False).astype(np.float32).values

    # Account age
    user_age = (cutoff - tables["Users"]["first_seen_at"]).dt.total_seconds() / 86400.0
    feature_dict["user_age_days"] = user_age.clip(lower=0.0).fillna(0.0).astype(np.float32).values

    return pd.DataFrame(feature_dict, index=users_df.index)


def build_hetero_graph(
    tables: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    *,
    temporal_decay: bool = True,
    max_decay_days: float | None = 30.0,
) -> HeteroData:
    """Build an 8-node HeteroData graph with forward and reverse edges.

    Args:
        tables: the feature view returned by ``build_split`` (all rows are
            strictly before ``cutoff``).
        cutoff: split cutoff; used for ages and temporal edge weights.
        temporal_decay: when True, event edges carry
            ``exp(-min(dt_days, max_decay_days) / tau_activity)``.  When False,
            every ``edge_weight`` is exactly 1.0 (the ablation).
        max_decay_days: clip horizon for ``dt_days``; ``None`` or ``0`` means
            no clipping.
    """
    if max_decay_days is not None and max_decay_days < 0:
        raise ValueError(f"max_decay_days must be >= 0, got {max_decay_days}")
    data = HeteroData()

    # ---------------- 1. Node ID Index Mappings ----------------
    id_maps = {}
    pk_cols = {
        "Users": "user_id",
        "Repositories": "repo_id",
        "PullRequests": "pr_id",
        "Issues": "issue_id",
        "IssueComments": "comment_id",
        "PRReviews": "review_id",
        "PRReviewComments": "comment_id",
        "Pushes": "push_id",
    }

    for tbl, pk in pk_cols.items():
        if tbl in tables and not tables[tbl].empty:
            pks = tables[tbl][pk].values
            id_maps[tbl] = {pk_val: i for i, pk_val in enumerate(pks)}
        else:
            id_maps[tbl] = {}

    # ---------------- 2. Node Features ----------------
    # Users (Recency + Counts + Profile)
    user_feats_df = extract_user_behavior_features(tables, cutoff)
    data["Users"].x = torch.tensor(user_feats_df.values, dtype=torch.float32)

    # Repositories
    repo_age = (cutoff - tables["Repositories"]["first_seen_at"]).dt.total_seconds() / 86400.0
    repo_is_org = tables["Repositories"]["org_id"].notna().astype(np.float32).values
    data["Repositories"].x = torch.tensor(
        _stack_f32(repo_age, repo_is_org), dtype=torch.float32
    )

    # PullRequests
    pr = tables["PullRequests"]
    pr_age = (cutoff - pr["first_seen_at"]).dt.total_seconds() / 86400.0
    pr_draft = pr["draft"].fillna(False).astype(np.float32).values
    pr_cross = pr["is_cross_repo"].fillna(False).astype(np.float32).values
    pr_branch_depth = pr["head_ref"].fillna("").astype(str).str.count("/").values
    data["PullRequests"].x = torch.tensor(
        _stack_f32(pr_age, pr_draft, pr_cross, pr_branch_depth), dtype=torch.float32
    )

    # Issues
    iss = tables["Issues"]
    iss_age = (cutoff - iss["first_seen_at"]).dt.total_seconds() / 86400.0
    iss_is_pr = iss["is_pull_request"].fillna(False).astype(np.float32).values
    iss_num_labels = iss["num_labels"].fillna(0).astype(np.float32).values
    iss_body = iss["body"].fillna("").astype(str)
    iss_log_body = np.log1p(iss_body.str.len().values)
    # Masking keeps punctuation, so detect real markdown: a heading at line
    # start, a fenced block, or inline code. A bare '#' is not a signal
    # because cross references such as '#123' are masked to '#000'.
    iss_has_md = iss_body.str.contains(
        r"(?m)^\s{0,3}#{1,6}\s|```|`[^`\n]+`", regex=True
    ).astype(np.float32).values
    # Titles are masked (letters -> a, digits -> 0), so match the shape of a
    # conventional prefix: scope word, optional (scope) or !, then a colon.
    iss_has_prefix = iss["title"].fillna("").astype(str).str.contains(
        r"^a+(?:\([^)]*\))?!?:", regex=True
    ).astype(np.float32).values
    data["Issues"].x = torch.tensor(
        _stack_f32(iss_age, iss_is_pr, iss_num_labels,
                   iss_log_body, iss_has_md, iss_has_prefix),
        dtype=torch.float32
    )

    # IssueComments
    ic = tables["IssueComments"]
    ic_age = (cutoff - ic["created_at"]).dt.total_seconds() / 86400.0
    ic_is_pr = ic["is_pull_request"].fillna(False).astype(np.float32).values
    data["IssueComments"].x = torch.tensor(
        _stack_f32(ic_age, ic_is_pr), dtype=torch.float32
    )

    # PRReviews
    rev = tables["PRReviews"]
    rev_age = (cutoff - rev["first_seen_at"]).dt.total_seconds() / 86400.0
    rev_approved = (rev["state"] == "approved").astype(np.float32).values
    rev_changes = (rev["state"] == "changes_requested").astype(np.float32).values
    data["PRReviews"].x = torch.tensor(
        _stack_f32(rev_age, rev_approved, rev_changes), dtype=torch.float32
    )

    # PRReviewComments
    prc = tables["PRReviewComments"]
    prc_age = (cutoff - prc["created_at"]).dt.total_seconds() / 86400.0
    prc_log_diff = np.log1p(prc["diff_hunk"].fillna("").astype(str).str.len().values)
    prc_path_depth = prc["path"].fillna("").astype(str).str.count("/").values
    data["PRReviewComments"].x = torch.tensor(
        _stack_f32(prc_age, prc_log_diff, prc_path_depth), dtype=torch.float32
    )

    # Pushes
    push = tables["Pushes"]
    push_age = (cutoff - push["created_at"]).dt.total_seconds() / 86400.0
    data["Pushes"].x = torch.tensor(_stack_f32(push_age), dtype=torch.float32)

    # ---------------- 3. Build Graph Edges (Forward + Reverse) ----------------
    def add_bidirected_edges(src_tbl, dst_tbl, rel_name, pairs, weights=None):
        """Add a forward edge and its typed reverse edge from (src, dst) index pairs.

        ``weights`` is one float per pair (the temporal decay).  Reverse edges
        carry the same weight because they describe the same event.  When no
        weights are supplied (static ownership), unit weights are stored so
        every edge type in the graph is self-describing.
        """
        if not pairs:
            return
        src_indices = [p[0] for p in pairs]
        dst_indices = [p[1] for p in pairs]
        if weights is None:
            weights = np.ones(len(pairs), dtype=np.float32)
        edge_weight = torch.as_tensor(weights, dtype=torch.float32)
        if edge_weight.numel() != len(pairs):
            raise ValueError(
                f"{src_tbl}->{dst_tbl}: {edge_weight.numel()} weights for {len(pairs)} edges"
            )
        data[src_tbl, rel_name, dst_tbl].edge_index = torch.tensor(
            [src_indices, dst_indices], dtype=torch.long
        )
        data[src_tbl, rel_name, dst_tbl].edge_weight = edge_weight
        # Reverse edge (bidirected flow)
        data[dst_tbl, f"rev_{rel_name}", src_tbl].edge_index = torch.tensor(
            [dst_indices, src_indices], dtype=torch.long
        )
        data[dst_tbl, f"rev_{rel_name}", src_tbl].edge_weight = edge_weight.clone()

    def add_edge_type(src_tbl, dst_tbl, rel_name, src_df, src_fk_col):
        if src_tbl not in tables or dst_tbl not in tables or src_df.empty:
            return

        valid = src_df.dropna(subset=[src_fk_col])
        src_map = id_maps[src_tbl]
        dst_map = id_maps[dst_tbl]
        time_col = "created_at" if "created_at" in valid.columns else "first_seen_at"
        if time_col not in valid.columns:
            raise ValueError(f"{src_tbl}: no event timestamp column for edge decay")

        activity = ACTIVITY_BY_EVENT_TABLE[src_tbl]
        tau_days = TEMPORAL_EDGE_TAU_DAYS[activity]

        pairs = []
        times = []
        for src_pk, dst_fk, stamp in zip(
            valid[pk_cols[src_tbl]], valid[src_fk_col], valid[time_col], strict=False
        ):
            if src_pk in src_map and dst_fk in dst_map:
                pairs.append((src_map[src_pk], dst_map[dst_fk]))
                times.append(stamp)

        weights = _decay_weights(
            times, cutoff, tau_days, max_decay_days, temporal_decay
        )
        add_bidirected_edges(src_tbl, dst_tbl, rel_name, pairs, weights)

    # Repositories -> Users "owner". Repositories has no owner_id FK: the owner
    # is stored as owner_login, so join it to Users.login. Organization-owned
    # repositories have no Users row, so they get no owner edge; the
    # `is_org_owned` flag in Repositories.x already records that case.
    repos = tables.get("Repositories")
    users = tables.get("Users")
    if repos is not None and users is not None and not repos.empty and not users.empty \
            and "owner_login" in repos.columns and "login" in users.columns:
        login_to_user_idx = {}
        user_map = id_maps["Users"]
        for user_pk, login in zip(users["user_id"].values, users["login"].values, strict=False):
            if user_pk in user_map and isinstance(login, str) and login not in login_to_user_idx:
                login_to_user_idx[login] = user_map[user_pk]

        repo_map = id_maps["Repositories"]
        owner_pairs = []
        for repo_pk, owner_login in zip(repos["repo_id"].values, repos["owner_login"].values, strict=False):
            user_idx = login_to_user_idx.get(owner_login)
            if user_idx is not None and repo_pk in repo_map:
                owner_pairs.append((repo_map[repo_pk], user_idx))
        # Ownership is static metadata, not an event: no temporal decay.
        add_bidirected_edges(
            "Repositories",
            "Users",
            "owner",
            owner_pairs,
            np.ones(len(owner_pairs), dtype=np.float32),
        )

    # Map edges based on foreign keys
    add_edge_type("PullRequests", "Users", "authored_by", tables["PullRequests"], "author_id")
    add_edge_type("PullRequests", "Repositories", "belongs_to", tables["PullRequests"], "repo_id")
    add_edge_type("Issues", "Users", "authored_by", tables["Issues"], "author_id")
    add_edge_type("Issues", "Repositories", "belongs_to", tables["Issues"], "repo_id")
    add_edge_type("IssueComments", "Users", "authored_by", tables["IssueComments"], "author_id")
    add_edge_type("IssueComments", "Issues", "comment_on", tables["IssueComments"], "issue_id")
    add_edge_type("PRReviews", "Users", "reviewed_by", tables["PRReviews"], "reviewer_id")
    add_edge_type("PRReviews", "PullRequests", "review_of", tables["PRReviews"], "pr_id")
    add_edge_type("PRReviewComments", "Users", "authored_by", tables["PRReviewComments"], "author_id")
    add_edge_type("PRReviewComments", "PullRequests", "comment_on", tables["PRReviewComments"], "pr_id")
    add_edge_type("Pushes", "Users", "pushed_by", tables["Pushes"], "user_id")
    add_edge_type("Pushes", "Repositories", "pushed_to", tables["Pushes"], "repo_id")

    return data


def mask_text_features(data: HeteroData, mode: str = "zero") -> HeteroData:
    """Remove the six structural/textual pattern features from the graph.

    Args:
        data: graph returned by :func:`build_hetero_graph`.
        mode: ``"zero"`` keeps the feature width and writes zeros (default),
            which keeps parameter counts identical to the full model;
            ``"drop"`` slices the columns out, so the encoder input width
            shrinks.  Both are valid ``no-text`` ablations; ``zero`` is the
            cleaner one-factor comparison.

    The function must run exactly once per graph.  It raises if a node feature
    matrix does not have its original width, which catches an accidental
    double-drop.
    """
    if mode not in {"zero", "drop"}:
        raise ValueError(f"mode must be 'zero' or 'drop', got {mode!r}")
    if N_TEXT_FEATURES != 6:
        raise AssertionError(f"expected 6 text features, mapped {N_TEXT_FEATURES}")

    for node_type, indices in TEXT_FEATURE_INDICES.items():
        if node_type not in data.node_types:
            continue
        x = data[node_type].x
        expected = ORIGINAL_FEATURE_DIMS[node_type]
        if x.shape[1] != expected:
            raise ValueError(
                f"{node_type}.x has {x.shape[1]} columns, expected {expected}; "
                "mask_text_features must run once, before any drop"
            )
        if mode == "zero":
            x[:, list(indices)] = 0.0
        else:
            drop = set(indices)
            keep = [i for i in range(x.shape[1]) if i not in drop]
            data[node_type].x = x[:, keep].contiguous()
    return data


def text_feature_sums(data: HeteroData) -> dict[str, dict[str, float]]:
    """Sum of ``abs`` per text feature, for a pre-mask integrity check."""
    out: dict[str, dict[str, float]] = {}
    for node_type, indices in TEXT_FEATURE_INDICES.items():
        x = data[node_type].x
        out[node_type] = {
            name: float(x[:, idx].abs().sum())
            for name, idx in zip(
                TEXT_FEATURE_COLUMNS[node_type], indices, strict=True
            )
        }
    return out


def edge_weight_stats(data: HeteroData) -> dict[str, dict[str, float | int]]:
    """Per-edge-type count/min/mean/max of ``edge_weight``."""
    out: dict[str, dict[str, float | int]] = {}
    for edge_type in data.edge_types:
        weight = getattr(data[edge_type], "edge_weight", None)
        key = "__".join(edge_type)
        if weight is None or weight.numel() == 0:
            out[key] = {"count": 0, "min": float("nan"), "mean": float("nan"), "max": float("nan")}
            continue
        out[key] = {
            "count": int(weight.numel()),
            "min": float(weight.min()),
            "mean": float(weight.mean()),
            "max": float(weight.max()),
        }
    return out


def check_edge_weights(data: HeteroData, *, expect_decay: bool) -> None:
    """Fail loudly if the temporal-edge ablation did not take effect.

    Every edge type must carry a finite weight in ``[0, 1]`` aligned with its
    ``edge_index``.  Event edges must show decay when ``expect_decay`` is true
    and exactly unit weights when it is false.  Static ``owner`` edges are
    excluded from the decay expectation because they have no event time.
    """
    event_means: list[float] = []
    for edge_type in data.edge_types:
        store = data[edge_type]
        weight = getattr(store, "edge_weight", None)
        if weight is None:
            raise RuntimeError(f"{edge_type} has no edge_weight")
        if weight.shape != (store.edge_index.shape[1],):
            raise RuntimeError(
                f"{edge_type}: edge_weight shape {tuple(weight.shape)} does not match "
                f"{store.edge_index.shape[1]} edges"
            )
        if weight.numel() == 0:
            continue
        if not bool(torch.isfinite(weight).all()):
            raise RuntimeError(f"{edge_type}: non-finite edge_weight")
        if float(weight.min()) < 0.0 or float(weight.max()) > 1.0 + 1e-6:
            raise RuntimeError(f"{edge_type}: edge_weight outside [0, 1]")
        if "owner" not in edge_type[1]:
            event_means.append(float(weight.mean()))

    if not event_means:
        raise RuntimeError("graph has no event edges to check")
    if expect_decay:
        if min(event_means) > 0.999:
            raise RuntimeError("decay requested but every event edge weight is ~1.0")
    else:
        if min(event_means) < 1.0 - 1e-6:
            raise RuntimeError("no-decay ablation still has non-unit edge weights")


if __name__ == "__main__":
    from src.data.split_io import load_tables
    from src.data.temporal_split import mask_future_leakage

    print("Testing Graph Assembly...")
    raw_tables = load_tables()
    cutoff = cast(pd.Timestamp, pd.Timestamp("2026-10-24 00:00:00"))
    masked_tables = mask_future_leakage(raw_tables, cutoff)

    graph = build_hetero_graph(masked_tables, cutoff)
    print("\nGraph successfully built:")
    print(graph)
    print("\nEdge weight summary:")
    for name, stats in edge_weight_stats(graph).items():
        print(f"  {name:<45} {stats}")
    print("\nNode Counts:")
    for ntype in graph.node_types:
        print(f"  {ntype:<18} nodes: {graph[ntype].num_nodes:<6} | feature_dim: {graph[ntype].x.shape[1]}")
