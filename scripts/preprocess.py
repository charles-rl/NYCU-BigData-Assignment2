"""Build the 8 anonymized tables of Assignment 2 from GH Archive hour files.

You download the GH Archive hour files yourself (https://www.gharchive.org) and put them in
one folder. This script reads all of them and writes the 8 tables:

    <out>/db/Users.parquet, Repositories.parquet, PullRequests.parquet, Issues.parquet,
             IssueComments.parquet, PRReviews.parquet, PRReviewComments.parquet, Pushes.parquet

Usage:

    python scripts/preprocess.py --raw RAW_DATA_FOLDER --out YOUR_OUTPUT_FOLDER

    RAW_DATA_FOLDER     the folder with your GH Archive hour files, e.g. gharchive
    YOUR_OUTPUT_FOLDER  the folder the tables are written to (any name), e.g. data

How it works:

    Step 1  find the hour files         (YYYY-MM-DD-H.json.gz, as named by GH Archive)
    Step 2  keep the cohort's events    (only the 9,000 users in cohort.txt)
    Step 3  build the 8 tables          (one row per user, repository, PR, issue, comment, ...)
    Step 4  anonymize                   (ids and names -> codes; text masked)
    Step 5  write the Parquet files

Anonymization: every column of the tables is kept, but
  * GitHub ids (users, repositories, PRs, ...), commit SHAs, GH Archive event ids and
    user / organization names are replaced by codes. A code is a keyed hash of the original
    value: the same user always gets the same code, whoever runs this script and for
    whatever period, so UserID in test_input.csv always matches Users.github_user_id;
  * text (titles, bodies, comments, file paths, diffs, branch names, labels) is masked: every
    letter becomes "a" and every digit "0", while spaces, line breaks and punctuation stay,
    so "Fix #12 for @alice" becomes "aaa #00 aaa @aaaaa". The length and layout of the text
    are kept, but not its content. Issue labels, issue types, GitHub App names and file
    extensions are masked in the same way.
See COLUMN RULES below for the exact list.

Requirements: Python 3.9+, pandas, pyarrow.
"""
import argparse
import gzip
import hashlib
import json
import re
import unicodedata
import zlib
from datetime import datetime, timezone
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Settings

DATA_START = datetime(2025, 12, 1, tzinfo=timezone.utc)  # files before this are ignored

HASH_KEY = b"NYCU-BigData-2026-Assignment2"  # fixed, so codes are the same for everyone

# The six GitHub event types that become rows in the tables.
EVENT_TYPES = {
    "PushEvent",
    "PullRequestEvent",
    "IssuesEvent",
    "IssueCommentEvent",
    "PullRequestReviewEvent",
    "PullRequestReviewCommentEvent",
}

HOUR_FILE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{1,2})\.json\.gz$")     # GH Archive file
WORKERS = 4  # hour files read at the same time

# GH Archive writes the keys in the order id, type, actor, ..., so the first match of this
# pattern on a line is the actor. Checking it first avoids parsing the JSON of other users.
ACTOR_ID_RE = re.compile(rb'"actor":\{"id":(\d+)')

# Table layout: primary key, GitHub-id column, row-time column, columns, foreign keys
# Types: int, str, bool, time (UTC). Every primary key is the row number 0..n-1 of its
# table (rows sorted by the row-time column) and every foreign key holds the row number of
# the row it points to. Column meanings are in SCHEMA.md.

TABLES = {
    "Users": {
        "pk": "user_id", "github": "github_user_id", "time": "first_seen_at",
        "columns": [("user_id", "int"), ("github_user_id", "int"), ("login", "str"),
                    ("is_bot", "bool"), ("is_cohort", "bool"), ("first_seen_at", "time")],
        "fks": {},
    },
    "Repositories": {
        "pk": "repo_id", "github": "github_repo_id", "time": "first_seen_at",
        "columns": [("repo_id", "int"), ("github_repo_id", "int"), ("name", "str"),
                    ("owner_login", "str"), ("org_id", "int"), ("org_login", "str"),
                    ("first_seen_at", "time")],
        "fks": {},
    },
    "PullRequests": {
        "pk": "pr_id", "github": "github_pr_id", "time": "first_seen_at",
        "columns": [("pr_id", "int"), ("github_pr_id", "int"), ("repo_id", "int"),
                    ("number", "int"), ("author_id", "int"), ("title", "str"), ("draft", "bool"),
                    ("head_ref", "str"), ("base_ref", "str"), ("head_sha", "str"),
                    ("base_sha", "str"), ("head_repo_id", "int"), ("is_cross_repo", "bool"),
                    ("created_at", "time"), ("merged_at", "time"), ("merged_by_id", "int"),
                    ("closed_at", "time"), ("first_seen_at", "time")],
        "fks": {"repo_id": "Repositories", "author_id": "Users", "merged_by_id": "Users"},
    },
    "Issues": {
        "pk": "issue_id", "github": "github_issue_id", "time": "first_seen_at",
        "columns": [("issue_id", "int"), ("github_issue_id", "int"), ("repo_id", "int"),
                    ("number", "int"), ("author_id", "int"), ("is_pull_request", "bool"),
                    ("pull_request_id", "int"), ("title", "str"), ("body", "str"),
                    ("labels", "str"), ("num_labels", "int"), ("assignee_id", "int"),
                    ("milestone_title", "str"), ("issue_type", "str"), ("locked", "bool"),
                    ("created_via_app", "str"), ("created_at", "time"), ("closed_at", "time"),
                    ("state_reason", "str"), ("first_seen_at", "time")],
        "fks": {"repo_id": "Repositories", "author_id": "Users",
                "pull_request_id": "PullRequests", "assignee_id": "Users"},
    },
    "IssueComments": {
        "pk": "comment_id", "github": "github_comment_id", "time": "created_at",
        "columns": [("comment_id", "int"), ("github_comment_id", "int"), ("issue_id", "int"),
                    ("repo_id", "int"), ("author_id", "int"), ("is_pull_request", "bool"),
                    ("body", "str"), ("via_app", "str"), ("created_at", "time"),
                    ("event_id", "str")],
        "fks": {"issue_id": "Issues", "repo_id": "Repositories", "author_id": "Users"},
    },
    "PRReviews": {
        "pk": "review_id", "github": "github_review_id", "time": "first_seen_at",
        "columns": [("review_id", "int"), ("github_review_id", "int"), ("pr_id", "int"),
                    ("repo_id", "int"), ("reviewer_id", "int"), ("state", "str"),
                    ("body", "str"), ("commit_id", "str"), ("submitted_at", "time"),
                    ("dismissed_at", "time"), ("first_seen_at", "time"), ("event_id", "str")],
        "fks": {"pr_id": "PullRequests", "repo_id": "Repositories", "reviewer_id": "Users"},
    },
    "PRReviewComments": {
        "pk": "comment_id", "github": "github_comment_id", "time": "created_at",
        "columns": [("comment_id", "int"), ("github_comment_id", "int"), ("pr_id", "int"),
                    ("repo_id", "int"), ("author_id", "int"), ("pull_request_review_id", "int"),
                    ("in_reply_to_id", "int"), ("path", "str"), ("file_extension", "str"),
                    ("subject_type", "str"), ("position", "int"), ("original_position", "int"),
                    ("commit_id", "str"), ("diff_hunk", "str"), ("body", "str"),
                    ("created_at", "time"), ("event_id", "str")],
        "fks": {"pr_id": "PullRequests", "repo_id": "Repositories", "author_id": "Users",
                "pull_request_review_id": "PRReviews", "in_reply_to_id": "PRReviewComments"},
    },
    "Pushes": {
        "pk": "push_id", "github": "github_push_id", "time": "created_at",
        "columns": [("push_id", "int"), ("github_push_id", "int"), ("user_id", "int"),
                    ("repo_id", "int"), ("ref", "str"), ("ref_type", "str"), ("branch", "str"),
                    ("is_new_ref", "bool"), ("created_at", "time"), ("event_id", "str")],
        "fks": {"user_id": "Users", "repo_id": "Repositories"},
    },
}

# COLUMN RULES (anonymization)

# Columns whose values are replaced by codes. Equal values get equal codes, so the column
# can still be compared and joined, e.g. PullRequests.head_repo_id with
# Repositories.github_repo_id, PRReviews.commit_id with PullRequests.head_sha, or
# Repositories.owner_login with Users.login.
# (table, column) -> kind of value; the kind is part of the hash, so a user and a
# repository with the same GitHub number get different codes.
CODED_COLUMNS = {
    ("Users", "github_user_id"): "user",
    ("Users", "login"): "login",
    ("Repositories", "github_repo_id"): "repo",
    ("Repositories", "owner_login"): "login",
    ("Repositories", "org_id"): "org",
    ("Repositories", "org_login"): "login",
    ("PullRequests", "github_pr_id"): "pull_request",
    ("PullRequests", "head_repo_id"): "repo",
    ("PullRequests", "head_sha"): "commit",
    ("PullRequests", "base_sha"): "commit",
    ("Issues", "github_issue_id"): "issue",
    ("IssueComments", "github_comment_id"): "issue_comment",
    ("IssueComments", "event_id"): "event",
    ("PRReviews", "github_review_id"): "review",
    ("PRReviews", "commit_id"): "commit",
    ("PRReviews", "event_id"): "event",
    ("PRReviewComments", "github_comment_id"): "review_comment",
    ("PRReviewComments", "commit_id"): "commit",
    ("PRReviewComments", "event_id"): "event",
    ("Pushes", "github_push_id"): "push",
    ("Pushes", "event_id"): "event",
}

# Text columns that are masked (see mask()). Repositories.name ("owner/repo") becomes
# "<code of owner>/<masked repo name>".
MASKED_COLUMNS = {
    "PullRequests": ["title", "head_ref", "base_ref"],
    "Issues": ["title", "body", "labels", "milestone_title", "issue_type", "created_via_app"],
    "IssueComments": ["body", "via_app"],
    "PRReviews": ["body"],
    "PRReviewComments": ["path", "file_extension", "diff_hunk", "body"],
    "Pushes": ["ref", "branch"],
}

DIGIT_RE = re.compile(r"\d")
LETTER_RE = re.compile(r"[^\W\d_]")  # any letter, in any language


# Codes (keyed hashes)

def _digest(kind, value):
    data = f"{kind}:{value}".encode()
    return hashlib.blake2b(data, key=HASH_KEY, digest_size=8).digest()


def code_int(kind, value):
    """A 15-digit code for a GitHub id, e.g. code_int("user", 6470068)."""
    if value is None:
        return None
    return 10**14 + int.from_bytes(_digest(kind, value), "big") % (9 * 10**14)


def code_str(kind, value):
    """A 16-character hex code for a string value such as a commit SHA or a login.
    Logins are not case-sensitive on GitHub, so they are lower-cased first."""
    if value is None:
        return None
    if kind == "login":
        value = value.lower()
    return _digest(kind, value).hex()


@lru_cache(maxsize=None)
def _mask_char(ch):
    """Mask one character of a non-ASCII text: numbers of any kind (7, ², ①, Ⅻ) -> "0",
    letters and letter-like symbols (circled letters, regional indicators, ...) -> "a", and
    accents and vowel signs, which belong to the letter before them, are dropped."""
    cat = unicodedata.category(ch)
    if cat.startswith("N"):
        return "0"
    if cat.startswith("L") or " LETTER " in unicodedata.name(ch, ""):
        return "a"
    if cat.startswith("M"):
        return ""
    return ch


def mask(text):
    """Every letter -> "a", every digit -> "0"; spaces, line breaks and punctuation stay.
    mask("Fix #12 for @alice") == "aaa #00 aaa @aaaaa"."""
    if not isinstance(text, str):
        return None
    if text.isascii():
        return LETTER_RE.sub("a", DIGIT_RE.sub("0", text))
    return "".join(map(_mask_char, text))


# Step 1: find the hour files

def file_hour(match):
    """UTC datetime of the hour in a file name matched by HOUR_FILE_RE."""
    year, month, day, hour = map(int, match.groups())
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def find_hour_files(raw_dir):
    """{hour (UTC datetime): path} for every GH Archive file in raw_dir."""
    files, too_early = {}, 0
    for path in sorted(Path(raw_dir).rglob("*.json.gz")):
        m = HOUR_FILE_RE.match(path.name)
        if not m:
            continue
        h = file_hour(m)
        if h < DATA_START:
            too_early += 1
        else:
            files[h] = path
    if too_early:
        print(f"Ignored {too_early} files before {DATA_START:%Y-%m-%d} (not allowed).")
    return dict(sorted(files.items()))


# Step 2: keep the events of the cohort users

COHORT = set()  # cohort user codes; filled in each worker process by _init_worker


def _init_worker(cohort):
    global COHORT
    COHORT = cohort


def read_gzip_lines(path):
    """Lines of a .json.gz file. A few GH Archive files have a damaged end; the lines
    before the damage are kept."""
    with gzip.open(path, "rb") as f:
        try:
            yield from f
        except (EOFError, gzip.BadGzipFile, zlib.error):
            return


def extract_hour(raw_path):
    """The cohort's events of one hour file, as raw JSON lines."""
    kept, checked = [], {}
    for line in read_gzip_lines(raw_path):
        m = ACTOR_ID_RE.search(line)
        if not m:
            continue
        actor = int(m.group(1))
        if actor not in checked:                       # hash each user only once per file
            checked[actor] = code_int("user", actor) in COHORT
        if not checked[actor]:
            continue
        try:
            event = json.loads(line)
        except ValueError:                             # damaged last line
            continue
        if event["type"] in EVENT_TYPES and not event["actor"]["login"].endswith("[bot]"):
            kept.append(line.rstrip(b"\n"))
    return kept


def extract_all(files, cohort):
    """Yield the cohort's events of every hour file, hour by hour in time order."""
    with Pool(WORKERS, initializer=_init_worker, initargs=(cohort,)) as pool:
        for i, lines in enumerate(pool.imap(extract_hour, files.values()), 1):
            if i % 24 == 0 or i == len(files):
                print(f"  {i}/{len(files)} hours read")
            yield lines


def hour_key(h):
    return f"{h:%Y-%m-%d}-{h.hour}"


# Step 3: build the tables

class TableBuilder:
    """Turns events (in time order) into rows. Rows are keyed by the original GitHub id
    and hold the original values until Step 4 replaces ids and names and masks the text."""

    def __init__(self):
        self.rows = {table: {} for table in TABLES}   # table -> GitHub id -> row (dict)

    def upsert(self, table, key, first=None, latest=None, earliest=None):
        """Create or update one row.
        first:    set a column only if it has no value yet
        latest:   always overwrite (the newest event wins)
        earliest: keep the earliest time"""
        row = self.rows[table].setdefault(key, {TABLES[table]["pk"]: key})
        for col, value in (first or {}).items():
            if row.get(col) is None and value is not None:
                row[col] = value
        for col, value in (latest or {}).items():
            row[col] = value
        for col, value in (earliest or {}).items():
            if value is not None and (row.get(col) is None or value < row[col]):
                row[col] = value
        return row

    # ---- objects that appear inside events ---------------------------------
    def user(self, u, t):
        """Register a user object ({"id", "login", ...}); returns its id."""
        if not u or u.get("id") is None:
            return None
        self.upsert("Users", u["id"],
                    latest={"login": u["login"], "is_bot": u["login"].endswith("[bot]")},
                    earliest={"first_seen_at": t})
        return u["id"]

    def repository(self, e, t):
        repo, org = e["repo"], e.get("org") or {}
        self.upsert("Repositories", repo["id"],
                    first={"org_id": org.get("id"), "org_login": org.get("login")},
                    latest={"name": repo["name"], "owner_login": repo["name"].split("/")[0]},
                    earliest={"first_seen_at": t})

    def pull_request(self, e, t):
        pr = e["payload"]["pull_request"]
        head_repo = (pr.get("head") or {}).get("repo") or {}
        self.upsert("PullRequests", pr["id"],
                    first={"repo_id": e["repo"]["id"], "number": pr.get("number"),
                           "head_ref": pr["head"]["ref"], "base_ref": pr["base"]["ref"],
                           "head_sha": pr["head"]["sha"], "base_sha": pr["base"]["sha"],
                           "head_repo_id": head_repo.get("id")},
                    earliest={"first_seen_at": t})
        return pr["id"]

    def issue(self, e, t):
        iss = e["payload"]["issue"]
        is_new = iss["id"] not in self.rows["Issues"]
        assignee = self.user(iss.get("assignee"), t)
        app = iss.get("performed_via_github_app")
        self.upsert("Issues", iss["id"],
                    first={"repo_id": e["repo"]["id"], "number": iss["number"],
                           "author_id": self.user(iss.get("user"), t),
                           "is_pull_request": "pull_request" in iss,
                           "created_via_app": app.get("slug") if app else None,
                           "created_at": iss.get("created_at")},
                    latest={"closed_at": iss.get("closed_at"),
                            "state_reason": iss.get("state_reason")},
                    earliest={"first_seen_at": t})
        if is_new:
            # Attributes that can change later (title, labels, assignee, ...) come from the
            # first event about the issue only, so a row never shows a value set after it
            # appeared.
            labels = [label["name"] for label in iss.get("labels") or []]
            issue_type = iss.get("type")
            self.rows["Issues"][iss["id"]].update({
                "title": iss.get("title"), "body": iss.get("body"),
                "milestone_title": (iss.get("milestone") or {}).get("title"),
                "labels": "|".join(labels), "num_labels": len(labels), "assignee_id": assignee,
                "issue_type": issue_type.get("name") if isinstance(issue_type, dict) else None,
                "locked": iss.get("locked"), "_draft": iss.get("draft")})
        return iss

    # ---- one event ----------------------------------------------------------
    def add(self, e):
        t, kind, p = e["created_at"], e["type"], e["payload"]
        actor = self.user(e["actor"], t)
        repo = e["repo"]["id"]
        self.repository(e, t)

        if kind == "PushEvent":
            ref = p.get("ref") or ""
            self.upsert("Pushes", p["push_id"], first={
                "user_id": actor, "repo_id": repo, "ref": ref,
                "ref_type": "tag" if ref.startswith("refs/tags/") else "branch",
                "branch": ref.removeprefix("refs/heads/").removeprefix("refs/tags/"),
                "is_new_ref": p.get("before") == "0" * 40, "created_at": t, "event_id": e["id"]})

        elif kind == "PullRequestEvent":
            pr = self.pull_request(e, t)
            action = p.get("action")
            if action == "opened":
                self.upsert("PullRequests", pr, first={"author_id": actor, "created_at": t})
            elif action == "merged":
                self.upsert("PullRequests", pr,
                            first={"merged_at": t, "merged_by_id": actor, "closed_at": t})
            elif action == "closed":
                self.upsert("PullRequests", pr, first={"closed_at": t})
            elif action == "reopened":
                self.upsert("PullRequests", pr,
                            latest={"closed_at": None, "merged_at": None, "merged_by_id": None})

        elif kind == "IssuesEvent":
            self.issue(e, t)

        elif kind == "IssueCommentEvent":
            iss = self.issue(e, t)
            c = p["comment"]
            app = c.get("performed_via_github_app")
            self.upsert("IssueComments", c["id"], first={
                "issue_id": iss["id"], "repo_id": repo, "author_id": self.user(c.get("user"), t),
                "is_pull_request": "pull_request" in iss, "body": c.get("body"),
                "via_app": app.get("slug") if app else None,
                "created_at": c.get("created_at") or t, "event_id": e["id"]})

        elif kind == "PullRequestReviewEvent":
            pr = self.pull_request(e, t)
            r = p["review"]
            self.upsert("PRReviews", r["id"], first={
                "pr_id": pr, "repo_id": repo, "reviewer_id": self.user(r.get("user"), t),
                "state": (r.get("state") or "").lower() or None, "body": r.get("body"),
                "commit_id": r.get("commit_id"), "submitted_at": r.get("submitted_at") or t,
                "event_id": e["id"]},
                earliest={"first_seen_at": t})
            if p.get("action") == "dismissed":
                self.upsert("PRReviews", r["id"], first={"dismissed_at": t})

        elif kind == "PullRequestReviewCommentEvent":
            pr = self.pull_request(e, t)
            c = p["comment"]
            path = c.get("path") or ""
            file_name = path.rsplit("/", 1)[-1]
            self.upsert("PRReviewComments", c["id"], first={
                "pr_id": pr, "repo_id": repo, "author_id": self.user(c.get("user"), t),
                "pull_request_review_id": c.get("pull_request_review_id"),
                "in_reply_to_id": c.get("in_reply_to_id"), "path": path or None,
                "file_extension": file_name.rsplit(".", 1)[-1].lower() if "." in file_name else None,
                "subject_type": c.get("subject_type"), "position": c.get("position"),
                "original_position": c.get("original_position"), "commit_id": c.get("commit_id"),
                "diff_hunk": c.get("diff_hunk"), "body": c.get("body"),
                "created_at": c.get("created_at") or t, "event_id": e["id"]})

    # ---- after the last event -----------------------------------------------
    def finish(self):
        prs, issues = self.rows["PullRequests"], self.rows["Issues"]

        # Link each PR conversation (an issue with is_pull_request) to its PR. Since late
        # 2025 PullRequestEvent no longer carries the author or title, so they are taken
        # from the conversation issue when there is one; the PR row then starts no earlier
        # than that issue was first seen.
        pr_by_number = {(r["repo_id"], r["number"]): key for key, r in prs.items()}
        for iss in issues.values():
            draft = iss.pop("_draft", None)
            if not iss.get("is_pull_request"):
                continue
            pr_key = pr_by_number.get((iss["repo_id"], iss["number"]))
            iss["pull_request_id"] = pr_key
            if pr_key is None:
                continue
            pr, borrowed = prs[pr_key], False
            for col, value in (("author_id", iss.get("author_id")), ("title", iss.get("title")),
                               ("draft", draft), ("created_at", iss.get("created_at"))):
                if pr.get(col) is None and value is not None:
                    pr[col] = value
                    borrowed = True
            if borrowed and iss["first_seen_at"] > pr["first_seen_at"]:
                pr["first_seen_at"] = iss["first_seen_at"]

        for pr in prs.values():
            head = pr.get("head_repo_id")
            pr["is_cross_repo"] = None if head is None else head != pr["repo_id"]

        # A foreign key to a row that never appears in the data becomes null, so every
        # non-null foreign key points to an existing row.
        for table, spec in TABLES.items():
            for col, ref in spec["fks"].items():
                existing = self.rows[ref]
                for row in self.rows[table].values():
                    if row.get(col) is not None and row[col] not in existing:
                        row[col] = None

    def to_frames(self, cohort):
        """One DataFrame per table, sorted by its row-time column."""
        frames = {}
        for table, spec in TABLES.items():
            names = [name for name, _ in spec["columns"]]
            df = pd.DataFrame(list(self.rows[table].values()), columns=names)
            df = df.sort_values([spec["time"], spec["pk"]], kind="stable", ignore_index=True)
            frames[table] = df
        users = frames["Users"]
        users["is_cohort"] = [code_int("user", uid) in cohort for uid in users["user_id"]]
        return frames


def build_tables(hour_lines, cohort):
    """Step 3: the cohort's events, hour by hour in time order, into a TableBuilder."""
    builder, seen, n_events = TableBuilder(), set(), 0
    for lines in hour_lines:
        events = []
        for line in lines:
            e = json.loads(line)
            if e["id"] not in seen:
                seen.add(e["id"])
                events.append(e)
        events.sort(key=lambda e: (e["created_at"], int(e["id"])))
        for e in events:
            builder.add(e)
        n_events += len(events)
    builder.finish()
    print(f"Step 3: {n_events:,} events of the cohort users.")
    return builder.to_frames(cohort)


# Step 4: anonymize, then number the rows

def anonymize(frames):
    """Apply the COLUMN RULES and turn GitHub-id keys into row numbers."""
    repos = frames["Repositories"]
    # "owner/repo" -> "<code of owner>/<masked repo>", computed before owner_login is coded
    repo_names = [None if not isinstance(n, str) else
                  code_str("login", n.split("/", 1)[0]) + "/" + mask(n.split("/", 1)[-1])
                  for n in repos["name"]]
    # Row number of every row, looked up by its original GitHub id.
    row_number = {table: pd.Series(np.arange(len(df)), index=df[TABLES[table]["pk"]].to_numpy())
                  for table, df in frames.items()}
    out = {}
    for table, df in frames.items():
        spec = TABLES[table]
        df = df.copy()
        pk = spec["pk"]
        df[spec["github"]] = df[pk]                       # original GitHub id, coded below
        df[pk] = np.arange(len(df))
        for col, ref in spec["fks"].items():
            df[col] = df[col].map(row_number[ref])
        for (t, col), kind in CODED_COLUMNS.items():
            if t == table:
                code = code_int if dict(spec["columns"])[col] == "int" else code_str
                df[col] = [code(kind, v) if pd.notna(v) else None for v in df[col]]
        for col in MASKED_COLUMNS.get(table, []):
            df[col] = [mask(v) for v in df[col]]
        if table == "Repositories":
            df["name"] = repo_names
        out[table] = set_types(df, spec)
    return out


def set_types(df, spec):
    """Give every column its type; times become UTC without a time zone."""
    for name, kind in spec["columns"]:
        if name == spec["pk"]:
            df[name] = df[name].astype("int64")
        elif kind == "int":
            df[name] = pd.array(df[name].astype(object), dtype="Int64")
        elif kind == "bool":
            df[name] = pd.array(df[name].astype(object), dtype="boolean")
        elif kind == "str":
            df[name] = df[name].astype(object).where(df[name].notna(), None)
        elif kind == "time":
            df[name] = (pd.to_datetime(df[name], utc=True, format="ISO8601")
                        .dt.tz_localize(None).astype("datetime64[ns]"))
    return df[[name for name, _ in spec["columns"]]]


# Step 5: write the files

PARQUET_TYPES = {"int": pa.int64(), "str": pa.string(), "bool": pa.bool_(), "time": pa.timestamp("ns")}


def write_output(out_dir, tables):
    db = out_dir / "db"
    db.mkdir(parents=True, exist_ok=True)
    for table, df in tables.items():
        # An explicit schema keeps the column types even when a column is all empty.
        schema = pa.schema([(name, PARQUET_TYPES[kind]) for name, kind in TABLES[table]["columns"]])
        pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False),
                       db / f"{table}.parquet")
        print(f"  {table:18s}{len(df):>10,} rows")


# Main

def load_cohort(path):
    """The cohort user codes, one per line."""
    return {int(line) for line in Path(path).read_text().split() if line.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, help="folder with the GH Archive hour files")
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args()

    out_dir = Path(args.out)
    cohort = load_cohort(Path(__file__).with_name("cohort.txt"))

    files = find_hour_files(args.raw)
    if not files:
        raise SystemExit(f"No GH Archive hour files (YYYY-MM-DD-H.json.gz) in {args.raw}.")
    hours = list(files)
    print(f"Step 1: {len(files)} GH Archive hour files, {hour_key(hours[0])} to {hour_key(hours[-1])} (UTC).")
    print("Step 2: keeping the cohort users' events.")
    frames = build_tables(extract_all(files, cohort), cohort)
    print("Step 4: anonymizing.")
    tables = anonymize(frames)
    print("Step 5: writing", out_dir / "db")
    write_output(out_dir, tables)


if __name__ == "__main__":
    main()
