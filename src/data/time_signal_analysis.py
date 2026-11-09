"""Empirical study of the temporal signal in the assignment data.

Reports, for the validation week encoded in splits/answer.csv:
  1. which 7-day window answer.csv corresponds to,
  2. the distribution of delta = t_cutoff - t_row over rows visible at cutoff,
  3. per-activity empirical P(active in target week | days since the last
     activity of that type) -- what tells us how to bucket a time feature
     instead of feeding one raw scalar,
  4. per-activity P(active in target week | activity count before cutoff).

Run:  python -m src.data.time_signal_analysis
"""

import numpy as np
import pandas as pd

TABLES = [
    "Users", "Repositories", "PullRequests", "Issues", "IssueComments",
    "PRReviews", "PRReviewComments", "Pushes",
]

# activity -> (table, actor column, event-time column defining the activity)
ACTIVITIES = {
    "open_pr": ("PullRequests", "author_id", "created_at"),
    "open_issue": ("Issues", "author_id", "created_at"),
    "issue_comment": ("IssueComments", "author_id", "created_at"),
    "pr_review": ("PRReviews", "reviewer_id", "submitted_at"),
    "pr_review_comment": ("PRReviewComments", "author_id", "created_at"),
    "push": ("Pushes", "user_id", "created_at"),
}

# row time: when the row becomes visible in the data. README: for a prediction
# at time t, use only rows whose row time is before t.
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

DB = "processed_tables/db"
DAY = 86400.0


def load() -> dict:
    return {t: pd.read_parquet(f"{DB}/{t}.parquet") for t in TABLES}


def visible(tables: dict, name: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows of `name` whose row time is strictly before cutoff."""
    df = tables[name]
    return df.loc[df[ROW_TIME[name]] < cutoff]


def actor_frame(tables: dict, name: str, actor: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """(u = actor row id, t = row time) for visible rows with a non-null actor."""
    df = visible(tables, name, cutoff)
    ids = pd.Series(pd.to_numeric(df[actor], errors="coerce"))
    keep = ids.notna().to_numpy()
    return pd.DataFrame({
        "u": ids.loc[keep].astype("int64").to_numpy(),
        "t": df[ROW_TIME[name]].loc[keep].to_numpy(),
    })


def labels_for(tables: dict, user_ids: np.ndarray, start: pd.Timestamp,
               end: pd.Timestamp) -> np.ndarray:
    """n_users x n_activities 0/1 matrix for events with event time in [start, end)."""
    out = np.zeros((len(user_ids), len(ACTIVITIES)), dtype=np.int64)
    for j, (tbl, actor, tcol) in enumerate(ACTIVITIES.values()):
        df = tables[tbl]
        hits = df.loc[(df[tcol] >= start) & (df[tcol] < end), actor].dropna()
        out[:, j] = np.isin(user_ids, hits.to_numpy(dtype="int64")).astype(np.int64)
    return out


def rate_by_bucket(values: pd.Series, edges: list, y: np.ndarray) -> list:
    """Empirical P(y=1) for users whose `values` fall in each [lo, hi) bucket."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = ((values >= lo) & (values < hi)).to_numpy()
        rows.append((int(sel.sum()), y[sel].mean() if sel.any() else np.nan))
    return rows


def print_row(label: str, rows: list) -> None:
    cells = "".join(
        (f"{r * 100:>8.1f}%({n:>4})" if r == r else f"{'--':>13}") for n, r in rows)
    print(f"  {label:>10}{cells}")


def main() -> None:
    tables = load()
    test = pd.read_csv("submission/test_input.csv")
    answer = pd.read_csv("splits/answer.csv").set_index("UserID")
    cohort = tables["Users"][tables["Users"]["is_cohort"]]
    users = cohort.loc[cohort["github_user_id"].isin(test["UserID"].to_numpy())]
    user_ids = users["user_id"].to_numpy(dtype="int64")
    gh_ids = users["github_user_id"].to_numpy(dtype="int64")
    acts = list(ACTIVITIES)

    truth = answer.loc[gh_ids, acts].to_numpy(dtype=np.int64)

    t0 = min(df[ROW_TIME[t]].min() for t, df in tables.items())
    t1 = max(df[ROW_TIME[t]].max() for t, df in tables.items())
    print(f"data span: {t0} -> {t1}")

    # 1. find the 7-day window that reproduces splits/answer.csv
    best_score, best_start = 0.0, None
    for start in pd.date_range(t0.normalize(), t1, freq="6h"):
        lab = labels_for(tables, user_ids, start, start + pd.Timedelta(days=7))
        score = float((lab == truth).mean())
        if score > best_score:
            best_score, best_start = score, start
        if score > 0.9999:
            break
    if best_start is None or best_score < 0.99:
        raise SystemExit(f"could not match answer.csv (best agreement {best_score:.4f})")
    cutoff = best_start
    end = cutoff + pd.Timedelta(days=7)
    print(f"validation window: {cutoff} -> {end}  (cell agreement with answer.csv "
          f"{best_score:.4f})")
    print("  per-activity agreement: "
          + "  ".join(f"{a}={float((labels_for(tables, user_ids, cutoff, end)[:, j] == truth[:, j]).mean()):.4f}"
                      for j, a in enumerate(acts)))

    y = truth
    print("\npositive rate in the target week (class imbalance):")
    for j, a in enumerate(acts):
        print(f"  {a:>18}: {y[:, j].mean():.3f}")

    # 2. distribution of delta = cutoff - row time, per table
    print("\ndelta = t_cutoff - t_row for visible rows, in HOURS")
    print(f"  {'table':>14} {'n':>7}" + "".join(f"{q:>10}" for q in
          ["p1", "p10", "p25", "p50", "p75", "p90", "p99"]))
    pooled = []
    for name in TABLES:
        d = (cutoff - visible(tables, name, cutoff)[ROW_TIME[name]]).dt.total_seconds() / 3600.0
        pooled.append(d)
        q = np.percentile(d, [1, 10, 25, 50, 75, 90, 99])
        print(f"  {name:>14} {len(d):>7}" + "".join(f"{x:>10.1f}" for x in q))
    allq = np.percentile(pd.concat(pooled), [1, 10, 25, 50, 75, 90, 99])
    print(f"  {'ALL ROWS':>14} {len(pd.concat(pooled)):>7}"
          + "".join(f"{x:>10.1f}" for x in allq))

    # 3. P(label | days since the last activity of that type before cutoff)
    recency, freq = {}, {}
    for a, (tbl, actor, _tcol) in ACTIVITIES.items():
        f = actor_frame(tables, tbl, actor, cutoff)
        last = pd.Series(f.groupby("u")["t"].max())
        deltas = cutoff - pd.DatetimeIndex(last.to_numpy())
        days = pd.Series(deltas.total_seconds() / DAY, index=last.index)
        recency[a] = days.reindex(user_ids)
        cnt = pd.Series(f.groupby("u").size())
        freq[a] = cnt.reindex(user_ids).fillna(0)

    print("\nP(active in target week | days since last activity of that type)")
    print("  (bucket [0,7) is exactly the copy-last-week rule)")
    edges = [0, 1, 2, 3, 5, 7, 10, 14, 21, 28]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        print_row(f"[{lo},{hi})",
                  [rate_by_bucket(recency[a], [lo, hi + 1e-9], y[:, j])[0]
                   for j, a in enumerate(acts)])
    print_row("never", [(int(recency[a].isna().sum()),
                        y[recency[a].isna().to_numpy(), j].mean()
                        if recency[a].isna().any() else np.nan)
                       for j, a in enumerate(acts)])

    # 4. P(label | total count of that activity visible before cutoff)
    print("\nP(active in target week | count of that activity visible before cutoff)")
    for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 10 ** 9)]:
        label = f"[{lo},{'+' if hi > 10**8 else str(hi)})"
        print_row(label,
                  [rate_by_bucket(freq[a], [lo, hi + 1e-9], y[:, j])[0]
                   for j, a in enumerate(acts)])


if __name__ == "__main__":
    main()
