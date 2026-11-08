#!/usr/bin/env python3
"""Data explorer for the 8 tables built by scripts/preprocess.py.

Throwaway-by-design: it does not touch the assignment pipeline, it only reads
processed_tables/db and prints/writes what is actually inside those parquet
files, so you can see the columns and the real rows instead of only the schema
graph (schema_graph_pipeline.png).

Sections:
  1. overview      rows / columns / file size / row-time window per table
  2. columns       every column of every table: dtype, nulls, distinct, stats, examples
  3. sample rows   a few real rows per table, printed one column per line
  4. keys          PK uniqueness + every FK's fill rate and dangling count
  5. activity      the 6 target activities: events, users, events per day
  6. gotchas       the things SCHEMA.md makes easy to get wrong
  7. charts        PNGs in --charts-dir (row counts, timeline, nulls, activity skew)

Usage:
  python scripts/explore_data.py                      # everything
  python scripts/explore_data.py --table Pushes       # one table only
  python scripts/explore_data.py --rows 3             # 3 sample rows per table
  python scripts/explore_data.py --rows 0 --no-charts # quick text-only pass
  python scripts/explore_data.py --db processed_tables/db --width 200
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd

# ---------------------------------------------------------------- config ----

TABLE_ORDER = [
    "Users",
    "Repositories",
    "PullRequests",
    "Issues",
    "IssueComments",
    "PRReviews",
    "PRReviewComments",
    "Pushes",
]

# the column whose value is the row time (README §3)
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

PK = {
    "Users": "user_id",
    "Repositories": "repo_id",
    "PullRequests": "pr_id",
    "Issues": "issue_id",
    "IssueComments": "comment_id",
    "PRReviews": "review_id",
    "PRReviewComments": "comment_id",
    "Pushes": "push_id",
}

# table -> (fk column, referenced table, referenced column); mirrors SCHEMA.md
FK: dict[str, list[tuple[str, str, str]]] = {
    "Users": [],
    "Repositories": [],
    "PullRequests": [("repo_id", "Repositories", "repo_id"),
                     ("author_id", "Users", "user_id"),
                     ("merged_by_id", "Users", "user_id")],
    "Issues": [("repo_id", "Repositories", "repo_id"),
               ("author_id", "Users", "user_id"),
               ("assignee_id", "Users", "user_id"),
               ("pull_request_id", "PullRequests", "pr_id")],
    "IssueComments": [("repo_id", "Repositories", "repo_id"),
                      ("author_id", "Users", "user_id"),
                      ("issue_id", "Issues", "issue_id")],
    "PRReviews": [("repo_id", "Repositories", "repo_id"),
                  ("reviewer_id", "Users", "user_id"),
                  ("pr_id", "PullRequests", "pr_id")],
    "PRReviewComments": [("repo_id", "Repositories", "repo_id"),
                         ("author_id", "Users", "user_id"),
                         ("pr_id", "PullRequests", "pr_id"),
                         ("pull_request_review_id", "PRReviews", "review_id"),
                         ("in_reply_to_id", "PRReviewComments", "comment_id")],
    "Pushes": [("repo_id", "Repositories", "repo_id"),
               ("user_id", "Users", "user_id")],
}


def _is_real_issue(df: pd.DataFrame) -> pd.Series:
    """Issues rows that are real issues, not pull-request conversations."""
    return cast(pd.Series, ~cast(pd.Series, df["is_pull_request"]).astype(bool))


# the 6 target activities: name -> (table, actor column, extra row filter)
ACTIVITIES: dict[str, tuple[str, str, Callable[[pd.DataFrame], pd.Series] | None]] = {
    "open_pr": ("PullRequests", "author_id", None),
    "open_issue": ("Issues", "author_id", _is_real_issue),
    "issue_comment": ("IssueComments", "author_id", None),
    "pr_review": ("PRReviews", "reviewer_id", None),
    "pr_review_comment": ("PRReviewComments", "author_id", None),
    "push": ("Pushes", "user_id", None),
}

# columns holding (masked) text: worth showing raw examples for
TEXT_COLS = {"body", "title", "diff_hunk", "labels", "name", "login", "ref",
             "head_ref", "base_ref", "path", "commit_id", "head_sha",
             "base_sha", "event_id", "owner_login", "org_login",
             "milestone_title", "issue_type", "state_reason", "via_app",
             "created_via_app", "subject_type", "file_extension", "branch"}

TRUNC = 60          # chars of a text value shown in stats / sample view
BOX = 78            # width of section banners


# ------------------------------------------------------------- helpers -----

def ser(df: pd.DataFrame, name: str) -> pd.Series:
    """df[name] as a Series (cast so the type checker agrees with pandas)."""
    return cast(pd.Series, df[name])


def where(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast(pd.DataFrame, df.loc[mask])


def is_na(v: object) -> bool:
    return bool(pd.isna(v)) if pd.api.types.is_scalar(v) or v is None else False


def banner(title: str) -> str:
    return f"\n{'=' * BOX}\n {title}\n{'=' * BOX}"


def sub(title: str) -> str:
    return f"\n--- {title} " + "-" * max(0, BOX - len(title) - 5)


def clip(v: object, n: int = TRUNC) -> str:
    if v is None or is_na(v):
        return "NULL"
    s = str(v).replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def fmt_num(x: object) -> str:
    if x is None or is_na(x):
        return "NA"
    xf = float(x)                                    # type: ignore[arg-type]
    return f"{xf:,.0f}" if xf == int(xf) else f"{xf:,.2f}"


def col_stats(series: pd.Series) -> dict[str, Any]:
    """Interesting facts about one column, dtype-aware."""
    n = len(series)
    nulls = int(series.isna().sum())
    st: dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_pct": 100.0 * nulls / n if n else 0.0,
        "distinct": int(series.nunique(dropna=True)),
        "note": "",
    }

    if pd.api.types.is_bool_dtype(series):
        t = int(series.sum())
        st["note"] = f"true {t} / false {n - t}"
    elif pd.api.types.is_datetime64_any_dtype(series):
        if n - nulls:
            st["note"] = f"{str(series.min())[:19]} .. {str(series.max())[:19]}"
    elif pd.api.types.is_numeric_dtype(series):
        if n - nulls:
            st["note"] = (f"min {fmt_num(series.min())}  max {fmt_num(series.max())}  "
                          f"mean {fmt_num(series.mean())}  median {fmt_num(series.median())}")
    else:
        non_null = series.dropna()
        if len(non_null):
            lens = non_null.astype(str).str.len()
            st["note"] = f"len {lens.min()}..{lens.max()} mean {lens.mean():.0f}"
            st["top"] = [(clip(k, 28), int(v)) for k, v in non_null.value_counts().head(3).items()]
    return st


def activity_frame(tables: dict[str, pd.DataFrame], act: str) -> pd.DataFrame:
    """Rows that count as one occurrence of `act`, with the actor column non-null."""
    table, actor, filt = ACTIVITIES[act]
    df = tables.get(table)
    if df is None or actor not in df.columns:
        return pd.DataFrame()
    rows = where(df, filt(df)) if filt is not None else df
    return where(rows, ser(rows, actor).notna())


def daily_counts(df: pd.DataFrame, time_col: str) -> pd.Series:
    t = ser(df, time_col)
    return cast(pd.Series, t.groupby(t.dt.floor("D")).size())


def cohort_ids(users: pd.DataFrame | None) -> list[int]:
    if users is None or "is_cohort" not in users.columns or "user_id" not in users.columns:
        return []
    return [int(x) for x in ser(users, "user_id")[ser(users, "is_cohort").astype(bool)].tolist()]


def load_tables(db: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for t in names:
        p = db / f"{t}.parquet"
        if not p.exists():
            print(f"!! missing {p}")
            continue
        out[t] = pd.read_parquet(p)
    return out


# ------------------------------------------------------------- sections -----

def section_overview(tables: dict[str, pd.DataFrame], db: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    print(banner("1. OVERVIEW — which tables exist and how big they are"))
    rows, bounds = [], []
    for t, df in tables.items():
        tc = ROW_TIME.get(t)
        size = (db / f"{t}.parquet").stat().st_size
        if tc and tc in df.columns:
            lo, hi = ser(df, tc).min(), ser(df, tc).max()
            bounds += [pd.Timestamp(lo), pd.Timestamp(hi)]
            lo_s, hi_s = str(lo)[:16], str(hi)[:16]
        else:
            lo_s = hi_s = "-"
        rows.append({"table": t, "rows": len(df), "cols": df.shape[1],
                     "size": f"{size / 1024:,.0f} KB", "row time col": tc or "-",
                     "earliest": lo_s, "latest": hi_s})
    print(pd.DataFrame(rows).to_string(index=False))

    gmin, gmax = min(bounds), max(bounds)
    print(f"\nWindow covered by these files (row time): {gmin} .. {gmax}")
    print(f"Span {(gmax - gmin).days} days | {len(tables)} tables | "
          f"{sum(len(d) for d in tables.values()):,} rows total")

    users = tables.get("Users")
    if users is not None:
        n_cohort = len(cohort_ids(users))
        n_bot = int(ser(users, "is_bot").sum()) if "is_bot" in users.columns else 0
        print(f"\nUsers: {len(users)} total | cohort (is_cohort=True) {n_cohort} | bots {n_bot}")
        print("NOTE: scripts/cohort.txt lists 9,000 cohort users. Far fewer here means you only "
              "downloaded a short time window.")
    print("\nTip: README §3 — for a prediction at time t, use only rows whose ROW TIME is < t.")
    return gmin, gmax


def section_columns(tables: dict[str, pd.DataFrame]) -> None:
    print(banner("2. COLUMNS — every column, with stats computed from the real data"))
    for t, df in tables.items():
        print(sub(f"{t}  ({len(df)} rows, {df.shape[1]} columns)"))
        fk_map = {c: f"{rt}.{rc}" for c, rt, rc in FK.get(t, [])}
        recs = []
        for c in df.columns:
            s = col_stats(ser(df, c))
            role = "PK" if c == PK.get(t) else ("FK → " + fk_map[c] if c in fk_map else "")
            if c == ROW_TIME.get(t):
                role = (role + " | row time").lstrip(" |")
            recs.append({"column": c, "role": role, "dtype": s["dtype"],
                         "null%": f"{s['null_pct']:.1f}", "distinct": s["distinct"],
                         "stats": s["note"]})
        print(pd.DataFrame(recs).to_string(index=False))

        for c in df.columns:
            if c in TEXT_COLS and not pd.api.types.is_numeric_dtype(ser(df, c)):
                vals = ser(df, c).dropna().unique()[:3]
                if len(vals):
                    print(f"    {c} examples: " + " | ".join(clip(v, 42) for v in vals))
        for c in df.columns:
            s = col_stats(ser(df, c))
            if s.get("top"):
                non_null = int(ser(df, c).notna().sum())
                if s["distinct"] < non_null:          # skip columns where every value is unique
                    print(f"    {c} most common: "
                          + " | ".join(f"{k} ({v})" for k, v in s["top"]))  # type: ignore[union-attr]
                else:
                    print(f"    {c}: all {non_null} values unique (an id/code column)")


def section_samples(tables: dict[str, pd.DataFrame], n: int) -> None:
    print(banner(f"3. SAMPLE ROWS — {n} real row(s) per table, one column per line"))
    for t, df in tables.items():
        print(sub(t))
        for i in range(min(n, len(df))):
            print(f"  row {i} " + "-" * (BOX - 7))
            for c in df.columns:
                print(f"    {c:<24} {clip(df.iloc[i][c], 90)}")


def section_keys(tables: dict[str, pd.DataFrame]) -> None:
    print(banner("4. KEYS — PK uniqueness and whether the FK graph really exists in the data"))
    recs = []
    for t, df in tables.items():
        pk = PK.get(t)
        if not pk or pk not in df.columns:
            continue
        c = ser(df, pk)
        dup = int(c.duplicated().sum())
        recs.append({"table": t, "pk": pk, "unique": f"{len(df) - dup}/{len(df)}",
                     "min..max": f"{c.min()}..{c.max()}", "dups": dup})
    print(pd.DataFrame(recs).to_string(index=False))

    print(sub("foreign keys: fill = non-null share, resolve = share pointing at a row you have"))
    recs = []
    for t, df in tables.items():
        for c, rt, rc in FK.get(t, []):
            ref = tables.get(rt)
            if ref is None or c not in df.columns or rc not in ref.columns:
                continue
            known = ser(df, c).dropna()
            filled = len(known)
            valid = int(known.isin(set(ser(ref, rc).tolist())).sum())
            recs.append({"table": t, "fk": c, "references": f"{rt}.{rc}",
                         "non-null": filled,
                         "fill %": f"{100 * filled / len(df):.1f}",
                         "resolve %": f"{100 * valid / filled:.1f}" if filled else "-",
                         "dangling": filled - valid})
    print(pd.DataFrame(recs).to_string(index=False))


def section_activity(tables: dict[str, pd.DataFrame], gmin: pd.Timestamp,
                     gmax: pd.Timestamp) -> pd.DataFrame:
    print(banner("5. ACTIVITY — the 6 things you have to predict"))
    cohort = set(cohort_ids(tables.get("Users")))

    w_end = gmin.normalize() + pd.Timedelta(days=(gmax - gmin).days // 7 * 7)
    w_start = w_end - pd.Timedelta(days=7)
    print(f"Stand-in validation week used below: {w_start} .. {w_end}")

    recs, daily = [], {}
    for act, (table, actor, _) in ACTIVITIES.items():
        rows = activity_frame(tables, act)
        if rows.empty:
            continue
        tc = ROW_TIME[table]
        if tc not in rows.columns:
            continue
        tcol = ser(rows, tc)
        a = ser(rows, actor)
        daily[act] = tcol.groupby(tcol.dt.floor("D")).size()

        per_user = cast(pd.Series, rows.groupby(a).size())
        in_week = where(rows, (tcol >= w_start) & (tcol < w_end))
        week_users = {int(x) for x in ser(in_week, actor).unique().tolist()}
        recs.append({
            "activity": act, "table": table, "actor col": actor, "events": len(rows),
            "users >=1": len(per_user),
            "cohort among them": len({int(x) for x in per_user.index.tolist()} & cohort),
            "max by 1 user": int(per_user.max()) if len(per_user) else 0,
            f"positives {w_start.date()} wk": len(week_users & cohort) if cohort else len(week_users),
        })
    print(pd.DataFrame(recs).to_string(index=False))

    daily_df = pd.DataFrame(daily).sort_index().fillna(0).astype(int)
    if not daily_df.empty:
        print(sub("events per day, by activity (row time)"))
        print("          " + " ".join(f"{c:>16}" for c in daily_df.columns))
        for d, r in daily_df.iterrows():
            print(f"  {str(d)[:10]} " + " ".join(f"{v:>16,}" for v in r.tolist()))

    print(sub("rows per day, by table (row time)"))
    per_table = {t: daily_counts(df, ROW_TIME[t])
                 for t, df in tables.items() if ROW_TIME.get(t) in df.columns}
    print(pd.DataFrame(per_table).sort_index().fillna(0).astype(int).T.to_string())
    print("\nBlank days = GH Archive gave you no data for that day (missing hours).")
    return daily_df


def section_gotchas(tables: dict[str, pd.DataFrame]) -> None:
    print(banner("6. GOTCHAS worth knowing before you model"))
    pr = tables.get("PullRequests")
    if pr is not None and "created_at" in pr.columns:
        n = int(ser(pr, "created_at").isna().sum())
        print(f"- PullRequests: created_at is NULL on {n}/{len(pr)} rows → that PR opened before your "
              f"data starts, so it cannot give you an 'open_pr' label (SCHEMA.md).")
    iss = tables.get("Issues")
    if iss is not None and "is_pull_request" in iss.columns:
        n_pr = int(ser(iss, "is_pull_request").sum())
        print(f"- Issues: {n_pr}/{len(iss)} rows are PR conversations, {len(iss) - n_pr} are real issues "
              f"→ open_issue must exclude is_pull_request=True.")
        if "title" in iss.columns:
            print(f"  Issues.title non-null on {int(ser(iss, 'title').notna().sum())}/{len(iss)} rows.")
    if pr is not None and "title" in pr.columns:
        print(f"- PullRequests.title non-null on {int(ser(pr, 'title').notna().sum())}/{len(pr)} rows "
              f"(only known once the PR conversation was seen).")
    users = tables.get("Users")
    if users is not None:
        n_non = len(users) - len(cohort_ids(users))
        print(f"- Users: {n_non} non-cohort users appear only through someone else's activity, so "
              f"their history is incomplete — the test list is cohort users only.")
    print("- Text columns (body / title / diff_hunk) are MASKED: letters→'a', digits→'0'. "
          "You get shape and length, not meaning.")
    print("- first_seen_at (row time) ≠ created_at. README §3: do not filter PRs, issues or reviews "
          "by created_at/submitted_at — they enter the data late.")


# --------------------------------------------------------------- charts -----

def make_charts(tables: dict[str, pd.DataFrame], out_dir: Path,
                daily_df: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, name: str) -> None:
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    # a. table sizes
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(tables)
    counts = [len(tables[t]) for t in names]
    ax.bar(names, counts, color="#2563eb")
    ax.set_yscale("log")
    ax.set_ylabel("rows (log scale)")
    ax.set_title("processed_tables: rows per table")
    for i, c in enumerate(counts):
        ax.text(i, c * 1.1, f"{c:,}", ha="center", fontsize=8)
    save(fig, "a_table_sizes.png")

    # b. daily timeline of each activity
    if not daily_df.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for c in daily_df.columns:
            ax.plot(daily_df.index, ser(daily_df, c).to_numpy(), marker=".", label=c)
        ax.set_title("events per day by activity (row time)")
        ax.set_ylabel("events")
        ax.legend(ncol=3, fontsize=8)
        fig.autofmt_xdate()
        save(fig, "b_daily_activity.png")

    # c. null-rate heatmap over every column of every table
    nulls = pd.DataFrame({t: {c: float(ser(tables[t], c).isna().mean())
                              for c in tables[t].columns} for t in tables})
    fig, ax = plt.subplots(figsize=(4.5 + 1.1 * len(tables), 0.24 * len(nulls) + 2))
    ax.imshow(nulls.to_numpy() * 100, cmap="viridis_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(tables)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(nulls)), list(nulls.index), fontsize=7)
    ax.set_title("% null per column (blank cell = column not in that table)")
    for i in range(nulls.shape[0]):
        for j in range(nulls.shape[1]):
            raw = nulls.iat[i, j]
            if raw is None or pd.isna(raw):            # column not in this table -> leave blank
                continue
            v = float(raw) * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if v < 55 else "black")
    save(fig, "c_null_heatmap.png")

    cohort = set(cohort_ids(tables.get("Users")))

    # d. how skewed is each activity per cohort user
    if cohort:
        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        for ax, act in zip(axes.ravel(), ACTIVITIES, strict=True):
            rows = activity_frame(tables, act)
            if rows.empty:
                continue
            a = ser(rows, ACTIVITIES[act][1])
            per_user = cast(pd.Series, rows.groupby(a).size())
            coh = cast(pd.Series, per_user[per_user.index.isin(cohort)])
            if coh.empty:
                continue
            ax.hist(coh.to_numpy(), bins=30, color="#059669")
            ax.set_title(f"{act}: events per cohort user", fontsize=9)
            ax.set_xscale("log")
            ax.axvline(1, color="red", lw=0.8, ls="--")
        save(fig, "d_activity_per_user.png")

    # e. breadth: how many of the 6 activities does a cohort user ever do
    if cohort:
        breadth = pd.Series(0, index=sorted(cohort), dtype=int)
        for act in ACTIVITIES:
            rows = activity_frame(tables, act)
            if rows.empty:
                continue
            a = ser(rows, ACTIVITIES[act][1])
            per_user = cast(pd.Series, rows.groupby(a).size())
            hits = cast(pd.Series, per_user[per_user.index.isin(cohort)])
            one = (hits > 0).astype(int).reindex(breadth.index).fillna(0)
            breadth = (breadth + one).astype(int)
        fig, ax = plt.subplots(figsize=(7, 4))
        cast(pd.Series, breadth.value_counts().sort_index()).plot.bar(ax=ax, color="#7c3aed")
        ax.set_xlabel("how many of the 6 activities appear in your whole window")
        ax.set_ylabel("cohort users")
        ax.set_title("cohort user activity breadth")
        save(fig, "e_cohort_breadth.png")

    # f. which repositories carry the activity
    repo_counts = {}
    for t, df in tables.items():
        if "repo_id" in df.columns:
            r = ser(df, "repo_id")
            repo_counts[t] = cast(pd.Series, r.groupby(r).size())
    if repo_counts:
        rc = pd.DataFrame(repo_counts).fillna(0).astype(int)
        top = cast(pd.Series, rc.sum(axis=1).sort_values(ascending=False)).head(20)
        fig, ax = plt.subplots(figsize=(9, 5))
        vals = cast(pd.Series, rc.loc[cast(pd.Series, top).index].sum(axis=1)).sort_values()
        vals.plot.barh(ax=ax, color="#0891b2")
        ax.set_xlabel("events")
        ax.set_ylabel("repo_id")
        ax.set_title("top 20 repositories by total activity")
        save(fig, "f_top_repos.png")

    # g. total activity per cohort user (class imbalance at a glance)
    if cohort:
        totals = pd.Series(0.0, index=sorted(cohort))
        for act in ACTIVITIES:
            rows = activity_frame(tables, act)
            if rows.empty:
                continue
            a = ser(rows, ACTIVITIES[act][1])
            per_user = cast(pd.Series, rows.groupby(a).size())
            per_user = cast(pd.Series, per_user[per_user.index.isin(cohort)])
            totals = totals.add(per_user.reindex(totals.index).fillna(0), fill_value=0)
        active = cast(pd.Series, totals[totals > 0])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(active.to_numpy(), bins=40, color="#ca8a04")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("total events over all 6 activities (whole window)")
        ax.set_ylabel("cohort users (log scale)")
        ax.set_title(f"cohort user total activity: {len(active)} of {len(totals)} have any")
        save(fig, "g_cohort_total_activity.png")

    return written


# ----------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser(description="explore the parquet tables in processed_tables/db")
    ap.add_argument("--db", default="processed_tables/db", help="folder with the .parquet files")
    ap.add_argument("--table", action="append", help="only these tables (repeatable)")
    ap.add_argument("--rows", type=int, default=2, help="sample rows per table (0 = skip section 3)")
    ap.add_argument("--no-charts", action="store_true", help="text only")
    ap.add_argument("--charts-dir", default="explore_output")
    ap.add_argument("--width", type=int, default=BOX, help="text output width")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.is_dir():
        print(f"no such folder: {db} (run scripts/preprocess.py first)")
        return 1

    names = args.table or [t for t in TABLE_ORDER if (db / f"{t}.parquet").exists()]
    extra = [p.stem for p in sorted(db.glob("*.parquet")) if p.stem not in TABLE_ORDER]
    if extra:
        print(f"note: other parquet files present, skipped: {extra}")

    tables = load_tables(db, names)
    if not tables:
        print(f"no tables found in {db}")
        return 1

    pd.set_option("display.width", args.width)
    pd.set_option("display.max_columns", 50)
    print(f"reading {db}: {', '.join(tables)}")

    gmin, gmax = section_overview(tables, db)
    section_columns(tables)
    if args.rows > 0:
        section_samples(tables, args.rows)
    section_keys(tables)
    daily = section_activity(tables, gmin, gmax)
    section_gotchas(tables)

    if not args.no_charts:
        print(banner("7. CHARTS"))
        try:
            for p in make_charts(tables, Path(args.charts_dir), daily):
                print(f"  wrote {p}")
        except Exception as e:                                  # noqa: BLE001
            print(f"  charts skipped: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
