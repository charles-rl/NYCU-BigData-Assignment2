# Schema

Every column of the 8 tables written by `preprocess.py`.

**Keys.** Every primary key (PK) is the row number 0, 1, 2, ..., and every foreign key (FK) holds the row number of the row it points to.

**Anonymization.** The preprocessing includes anonymization. The columns marked "code" or "masked" below are anonymized.

## Users

| column | type | key | description |
|---|---|---|---|
| `user_id` | int | PK | Row number 0..n-1 in time order; FKs pointing here hold this number (depends on the period you process) |
| `github_user_id` | int | code | Code of the original GitHub id of the user; the same in every period; this is `UserID` in test_input.csv |
| `login` | str | code | Code of the username (latest seen; users can rename). The same code as Repositories.owner_login for the repositories this user owns |
| `is_bot` | bool |  | The username ends with [bot] |
| `is_cohort` | bool |  | One of the 9,000 cohort users whose own activity is collected. Other users only appear because a cohort user's activity references them (e.g. the author of a PR a cohort user reviewed), so their activity is incomplete; the test list contains cohort users only |
| `first_seen_at` | time | row time | Earliest event in your data that references this user |

## Repositories

| column | type | key | description |
|---|---|---|---|
| `repo_id` | int | PK | Row number 0..n-1 in time order; FKs pointing here hold this number (depends on the period you process) |
| `github_repo_id` | int | code | Code of the original GitHub id of the repository; the same in every period |
| `name` | str | code, masked | owner/repo, written as `<code of the owner>/<masked repository name>`, e.g. `3fa2c1d09b7e4a55/aaaa-aaaaaaa`. Masked: letters → a, digits → 0 |
| `owner_login` | str | code | Code of the owner part of owner/repo (a user or an organization); compare it with Users.login and org_login |
| `org_id` | int | code | Code of the GitHub organization id; null for user-owned repositories (not a row number) |
| `org_login` | str | code | Code of the organization login; null for user-owned repositories. Equals owner_login when the organization owns the repository |
| `first_seen_at` | time | row time | Earliest event in your data on this repository |

## PullRequests

| column | type | key | description |
|---|---|---|---|
| `pr_id` | int | PK | Row number 0..n-1 in time order; FKs pointing here hold this number (depends on the period you process) |
| `github_pr_id` | int | code | Code of the original GitHub id of the pull request; the same in every period |
| `repo_id` | int | FK → Repositories | Target repository |
| `number` | int |  | PR number within the repo (#123) |
| `author_id` | int | FK → Users | PR author; null if the PR was opened before your data starts and nobody commented on it |
| `title` | str | masked | PR title; only known if the PR has an IssueCommentEvent. Masked: letters → a, digits → 0 |
| `draft` | bool |  | Draft PR; only known if someone commented on the PR conversation |
| `head_ref` | str | masked | Source branch. Masked: letters → a, digits → 0 |
| `base_ref` | str | masked | Target branch. Masked: letters → a, digits → 0 |
| `head_sha` | str | code | Code of the head commit when first seen |
| `base_sha` | str | code | Code of the base commit when first seen |
| `head_repo_id` | int | code | Code of the repository the branch lives in (a fork for external PRs); compare it with Repositories.github_repo_id. Not an FK, not a row number |
| `is_cross_repo` | bool |  | head_repo_id != repo_id, i.e. PR from a fork |
| `created_at` | time |  | When the PR was opened (may be before first_seen_at); null when the opening was never seen, e.g. a PR opened before your data starts |
| `merged_at` | time | known only later | When it was merged |
| `merged_by_id` | int | FK → Users, known only later | Who merged it |
| `closed_at` | time | known only later | When it was closed |
| `first_seen_at` | time | row time | When the PR row appears in your data, so it is the row time: its earliest pull-request or review event; if author, title or opening time come from the PR conversation issue seen later, the time that issue was first seen. Comments on the PR conversation update the issue row (Issues.first_seen_at), not this |

## Issues

| column | type | key | description |
|---|---|---|---|
| `issue_id` | int | PK | Row number 0..n-1 in time order; FKs pointing here hold this number (depends on the period you process) |
| `github_issue_id` | int | code | Code of the original GitHub id of the issue; the same in every period |
| `repo_id` | int | FK → Repositories | Repository |
| `number` | int |  | Issue number within the repo |
| `author_id` | int | FK → Users | Issue author |
| `is_pull_request` | bool |  | This issue is the conversation of a PR |
| `pull_request_id` | int | FK → PullRequests | The matching PR when is_pull_request and the PR is in PullRequests |
| `title` | str | masked | Title, from the first event about the issue. Masked: letters → a, digits → 0 |
| `body` | str | masked | Body text, from the first event about the issue. Masked: letters → a, digits → 0 |
| `labels` | str | masked | Label names joined by \|, from the first event about the issue (labels added later are not included). Masked: letters → a, digits → 0 |
| `num_labels` | int |  | Number of labels, from the first event about the issue |
| `assignee_id` | int | FK → Users | Assignee, from the first event about the issue (null if not set then, even when set later) |
| `milestone_title` | str | masked | Milestone, from the first event about the issue (null if not set then, even when set later). Masked: letters → a, digits → 0 |
| `issue_type` | str | masked | Issue type (Bug / Feature / Task ...) if the org uses them, from the first event about the issue (null if not set then, even when set later). Masked: letters → a, digits → 0 |
| `locked` | bool |  | Conversation locked, from the first event about the issue (null if not set then, even when set later) |
| `created_via_app` | str | masked | GitHub App that created it, if any. Masked: letters → a, digits → 0 |
| `created_at` | time |  | When the issue was opened (may be long before first_seen_at, even before your data starts) |
| `closed_at` | time | known only later | When it was closed; null if open at the end of your data |
| `state_reason` | str | known only later | completed / not_planned / reopened / duplicate |
| `first_seen_at` | time | row time | Earliest event in your data about this issue: the row exists from this time on, so it is the row time |

## IssueComments

| column | type | key | description |
|---|---|---|---|
| `comment_id` | int | PK | Row number 0..n-1 in time order; FKs pointing here hold this number (depends on the period you process) |
| `github_comment_id` | int | code | Code of the original GitHub id of the comment; the same in every period |
| `issue_id` | int | FK → Issues | Issue (or PR conversation) commented on |
| `repo_id` | int | FK → Repositories | Repository |
| `author_id` | int | FK → Users | Commenter |
| `is_pull_request` | bool |  | Comment is on a PR conversation |
| `body` | str | masked | Comment text. Masked: letters → a, digits → 0 |
| `via_app` | str | masked | GitHub App that posted it, if any. Masked: letters → a, digits → 0 |
| `created_at` | time | row time | When it was posted |
| `event_id` | str | code | Code of the GH Archive event id |

## PRReviews

| column | type | key | description |
|---|---|---|---|
| `review_id` | int | PK | Row number 0..n-1 in time order; FKs pointing here hold this number (depends on the period you process) |
| `github_review_id` | int | code | Code of the original GitHub id of the review; the same in every period |
| `pr_id` | int | FK → PullRequests | Reviewed PR |
| `repo_id` | int | FK → Repositories | Repository |
| `reviewer_id` | int | FK → Users | Reviewer |
| `state` | str |  | approved / commented / changes_requested (as submitted) |
| `body` | str | masked | Review summary text. Masked: letters → a, digits → 0 |
| `commit_id` | str | code | Code of the commit the review was made on (compare with PullRequests.head_sha) |
| `submitted_at` | time |  | When the review was submitted (can be before first_seen_at, even before your data starts) |
| `dismissed_at` | time | known only later | When the review was dismissed |
| `first_seen_at` | time | row time | Earliest event in your data about this review: the row exists from this time on, so it is the row time |
| `event_id` | str | code | Code of the GH Archive event id |

## PRReviewComments

| column | type | key | description |
|---|---|---|---|
| `comment_id` | int | PK | Row number 0..n-1 in time order; FKs pointing here hold this number (depends on the period you process) |
| `github_comment_id` | int | code | Code of the original GitHub id of the review comment; the same in every period |
| `pr_id` | int | FK → PullRequests | PR the comment is on |
| `repo_id` | int | FK → Repositories | Repository |
| `author_id` | int | FK → Users | Commenter |
| `pull_request_review_id` | int | FK → PRReviews | Review this comment belongs to (null if not in PRReviews) |
| `in_reply_to_id` | int | FK → PRReviewComments | Parent review comment in a thread (null if none / not in table) |
| `path` | str | masked | File the comment is on (its extension is also in file_extension). Masked: letters → a, digits → 0 |
| `file_extension` | str | masked | Extension of the file name (py, ts, md, ...). Masked: letters → a, digits → 0, so py becomes aa |
| `subject_type` | str |  | line / file |
| `position` | int |  | Line index in the diff (null if outdated) |
| `original_position` | int |  | Line index in the original diff |
| `commit_id` | str | code | Code of the commit the comment refers to |
| `diff_hunk` | str | masked | Diff context. Masked: letters → a, digits → 0 |
| `body` | str | masked | Comment text. Masked: letters → a, digits → 0 |
| `created_at` | time | row time | When it was posted |
| `event_id` | str | code | Code of the GH Archive event id |

## Pushes

| column | type | key | description |
|---|---|---|---|
| `push_id` | int | PK | Row number 0..n-1 in time order; FKs pointing here hold this number (depends on the period you process) |
| `github_push_id` | int | code | Code of the original GitHub id of the push; the same in every period |
| `user_id` | int | FK → Users | Pusher |
| `repo_id` | int | FK → Repositories | Repository |
| `ref` | str | masked | Full ref, e.g. refs/heads/main, stored as aaaa/aaaaa/aaaa; whether it is a branch or a tag is in ref_type. Masked: letters → a, digits → 0 |
| `ref_type` | str |  | branch / tag |
| `branch` | str | masked | Ref name without refs/heads/ or refs/tags/. Masked: letters → a, digits → 0 |
| `is_new_ref` | bool |  | The push created the branch or tag |
| `created_at` | time | row time | When the push happened |
| `event_id` | str | code | Code of the GH Archive event id |
