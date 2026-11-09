from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _stack_f32(*columns) -> np.ndarray:
    """Stack 1-D columns into a float32 feature matrix.

    All graph features are non-negative, so NaN becomes 0.0 and values clip at 0.
    """
    arrays = [
        np.nan_to_num(np.asarray(col, dtype=np.float64), nan=0.0).clip(min=0.0).astype(np.float32)
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


def build_hetero_graph(tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> HeteroData:
    """Build an 8-node HeteroData graph with forward and reverse edges."""
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
    def add_bidirected_edges(src_tbl, dst_tbl, rel_name, pairs):
        """Add a forward edge and its typed reverse edge from (src, dst) index pairs."""
        if not pairs:
            return
        src_indices = [p[0] for p in pairs]
        dst_indices = [p[1] for p in pairs]
        data[src_tbl, rel_name, dst_tbl].edge_index = torch.tensor(
            [src_indices, dst_indices], dtype=torch.long
        )
        # Reverse edge (bidirected flow)
        data[dst_tbl, f"rev_{rel_name}", src_tbl].edge_index = torch.tensor(
            [dst_indices, src_indices], dtype=torch.long
        )

    def add_edge_type(src_tbl, dst_tbl, rel_name, src_df, src_fk_col):
        if src_tbl not in tables or dst_tbl not in tables or src_df.empty:
            return

        valid = src_df.dropna(subset=[src_fk_col])
        src_map = id_maps[src_tbl]
        dst_map = id_maps[dst_tbl]

        pairs = []
        for src_pk, dst_fk in zip(valid[pk_cols[src_tbl]], valid[src_fk_col], strict=False):
            if src_pk in src_map and dst_fk in dst_map:
                pairs.append((src_map[src_pk], dst_map[dst_fk]))

        add_bidirected_edges(src_tbl, dst_tbl, rel_name, pairs)

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
        add_bidirected_edges("Repositories", "Users", "owner", owner_pairs)

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
    print("\nNode Counts:")
    for ntype in graph.node_types:
        print(f"  {ntype:<18} nodes: {graph[ntype].num_nodes:<6} | feature_dim: {graph[ntype].x.shape[1]}")
