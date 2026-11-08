import os
import pandas as pd
import numpy as np

def mask_future_leakage(tables: dict, cutoff_time: pd.Timestamp) -> dict:
    """Mask columns that become known only after cutoff_time."""
    tables_filtered = {}
    
    # 1. Filter rows by row_time < cutoff_time
    time_cols = {
        "Users": "first_seen_at",
        "Repositories": "first_seen_at",
        "PullRequests": "first_seen_at",
        "Issues": "first_seen_at",
        "IssueComments": "created_at",
        "PRReviews": "first_seen_at",
        "PRReviewComments": "created_at",
        "Pushes": "created_at"
    }
    
    for name, df in tables.items():
        time_col = time_cols.get(name)
        if time_col and time_col in df.columns:
            tables_filtered[name] = df[df[time_col] < cutoff_time].copy()
        else:
            tables_filtered[name] = df.copy()

    # 2. Mask PullRequests future columns
    if "PullRequests" in tables_filtered:
        pr = tables_filtered["PullRequests"]
        mask_merged = pr["merged_at"] >= cutoff_time
        pr.loc[mask_merged, "merged_at"] = pd.NaT
        pr.loc[mask_merged, "merged_by_id"] = np.nan
        
        mask_closed = pr["closed_at"] >= cutoff_time
        pr.loc[mask_closed, "closed_at"] = pd.NaT

    # 3. Mask Issues future columns
    if "Issues" in tables_filtered:
        issues = tables_filtered["Issues"]
        mask_closed_issue = issues["closed_at"] >= cutoff_time
        issues.loc[mask_closed_issue, "closed_at"] = pd.NaT
        issues.loc[mask_closed_issue, "state_reason"] = np.nan

    # 4. Mask PRReviews future columns
    if "PRReviews" in tables_filtered:
        reviews = tables_filtered["PRReviews"]
        mask_dismissed = reviews["dismissed_at"] >= cutoff_time
        reviews.loc[mask_dismissed, "dismissed_at"] = pd.NaT

    return tables_filtered


def extract_labels(tables: dict, start_time: pd.Timestamp, end_time: pd.Timestamp, users_df: pd.DataFrame) -> pd.DataFrame:
    """Generate ground truth binary labels for the 6 activities in [start_time, end_time)."""
    labels = pd.DataFrame()
    labels["user_id"] = users_df["user_id"]
    labels["UserID"] = users_df["github_user_id"]

    # 1. open_pr
    pr = tables["PullRequests"]
    pr_act = pr[(pr["created_at"] >= start_time) & (pr["created_at"] < end_time)]
    pr_users = set(pr_act["author_id"].dropna().astype(int))
    labels["open_pr"] = labels["user_id"].isin(pr_users).astype(int)

    # 2. open_issue (exclude PR conversation issues)
    issues = tables["Issues"]
    iss_act = issues[(issues["created_at"] >= start_time) & 
                     (issues["created_at"] < end_time) & 
                     (~issues["is_pull_request"])]
    iss_users = set(iss_act["author_id"].dropna().astype(int))
    labels["open_issue"] = labels["user_id"].isin(iss_users).astype(int)

    # 3. issue_comment
    ic = tables["IssueComments"]
    ic_act = ic[(ic["created_at"] >= start_time) & (ic["created_at"] < end_time)]
    ic_users = set(ic_act["author_id"].dropna().astype(int))
    labels["issue_comment"] = labels["user_id"].isin(ic_users).astype(int)

    # 4. pr_review
    rev = tables["PRReviews"]
    rev_act = rev[(rev["submitted_at"] >= start_time) & (rev["submitted_at"] < end_time)]
    rev_users = set(rev_act["reviewer_id"].dropna().astype(int))
    labels["pr_review"] = labels["user_id"].isin(rev_users).astype(int)

    # 5. pr_review_comment
    prc = tables["PRReviewComments"]
    prc_act = prc[(prc["created_at"] >= start_time) & (prc["created_at"] < end_time)]
    prc_users = set(prc_act["author_id"].dropna().astype(int))
    labels["pr_review_comment"] = labels["user_id"].isin(prc_users).astype(int)

    # 6. push
    pushes = tables["Pushes"]
    push_act = pushes[(pushes["created_at"] >= start_time) & (pushes["created_at"] < end_time)]
    push_users = set(push_act["user_id"].dropna().astype(int))
    labels["push"] = labels["user_id"].isin(push_users).astype(int)

    return labels
