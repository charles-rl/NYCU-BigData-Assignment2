# Data Notes

After inspecting 
```
Table                Total Rows   Unique PKs  
----------------------------------------------
IssueComments        2,595        2,595       
Issues               6,685        6,685       
PRReviewComments     1,475        1,475       
PRReviews            2,441        2,441       
PullRequests         10,075       10,075      
Pushes               42,116       42,116      
Repositories         11,475       11,475      
Users                5,338        5,338       
----------------------------------------------
Total Database Rows: 82,200
```
It seems that the number of database rows will fit inside my GPU and this is for October 1 to October 31 data.

> **Which data to train and validate on: see [`docs/VALIDATION_DATA.md`](docs/VALIDATION_DATA.md).**
> It gives the approved validation cutoff (`2026-02-15`), the rule for excluding damaged
> days per activity, and why November 2026 must never be scored.

## Data completeness (GH Archive drops event types, not files)

The raw downloads are byte-for-byte what `data.gharchive.org` serves, so re-downloading
does not help: several hours are legitimately incomplete at the source, and the missing
event type is almost always `PushEvent`. Confirmed by re-fetching three files
(`2026-10-28-12`, `2026-10-20-12`, `2026-11-02-12`) and comparing md5s — identical.

Failure modes seen, all of which pass `get_raw_data.py`'s `size > 1024` skip rule:

| pattern | example | effect |
| --- | --- | --- |
| `PushEvent` vanishes, rest of the hour is normal | `2026-11-01` 115k events, 0 push | push looks dead |
| `PushEvent` vanishes *and* the hour balloons | `2026-09-04` 1,499k events, 0 push (2.4× normal) | redistribution, not activity |
| one hour is a 404 body saved as `.gz` | `raw_data/2026-10-01-00.json.gz`, 127 bytes, starts `<?xml` | silently skipped by the loader |

Measured with `scripts/verify_raw_data.py` → `raw_data/coverage.csv` plus
`raw_data/coverage_activity.csv` (per day and activity):

- **8232 hour files, 5051 sane for all six activities at once (61.4%)**, 1 unreadable.
- 343 days, **181 usable for all six activities**, 121-166 fully clean per activity.
- Whole validation week `2026-11-01 .. 11-08` is bad: **0 usable days for every
  activity**, so the test week carries no trustworthy labels at all.

The loss is *activity-specific*, which is why the audit judges each of the six types
separately instead of checking `PushEvent` alone:

| activity | sane hours | clean days | usable days |
| --- | --- | --- | --- |
| `open_pr` | 6613 | 139 | 263 |
| `open_issue` | 6424 | 135 | 248 |
| `issue_comment` | 6700 | 161 | 269 |
| `pr_review` | 6781 | 166 | 275 |
| `pr_review_comment` | 6394 | 121 | 257 |
| `push` | 6634 | 240 | 257 |

Two calibration decisions that were wrong at first and are worth recording:

1. **The reference density must be local, not global.** Density of the five PR-ish
   types falls monotonically from Dec 2025 to Nov 2026 (18,289 `PullRequestEvent`s/hr
   in Dec 2025 vs 1,191-2,090 in Sep-Oct 2026), so a corpus-wide baseline marks an
   entire sparse month as broken and rejected 5 of 6 activities outright. The audit
   now compares each day against the median of its own ±7-day neighbourhood, so a
   regime shift cancels and a zero-push day does not.
2. **The band must be two-sided.** `2026-09-04 .. 09-06` carries ~1400
   `PullRequestEvent`s/day against a ~110/day neighbourhood while pushes vanish — a
   lower-bound-only test accepts that dump and feeds it into the curve.

Consequence for the label split: push prevalence in the `2026-11-01` anchor is 5.4%
instead of the ~51% a clean week gives, so `splits/answer.csv` under-counts push
positives by ~7-9×. The other five activities are unaffected (0.072 vs 0.079 for
`open_pr`, etc.). This is consistent with README §4: an activity whose data GH
Archive lost during the test week is simply not scored.

How the analysis handles it: `activity_decay_analysis.py --clean-days {off,label,strict}`
filters **per (anchor, activity)**, not per day — an anchor whose push stream survived
but whose review stream collapsed is a valid push anchor and an invalid review anchor,
and the damaged cell contributes no rows. `clean` demands all 24 hours; `usable`
(the default) allows ≥75% of hours plus the local density band, because requiring
24/24 on a typical 19/24 day has probability ~0.4% and makes whole months unusable.
Both the filtered and the unfiltered pass are reported side by side, because
*prevalence* reveals the contamination directly while *tau* does not.

---

## Periodicity: why there is no "cycle" column from a periodogram

`bump@7` is the primary periodicity read-out and it is negative (-0.19 .. -0.04) on
every one of the 12 curves, so the curves are decay-to-a-floor, not weekly resonance.
The spectral cross-check was replaced after it kept inventing cycles:

- Detrending the log curve with a **quadratic** leaves a sawtooth on a discrete
exponential; its leakage lands at the band edge and reports an **11.3-day "cycle"**.
A noiseless synthetic `floor + amp*exp(-dt/tau)` reproduces exactly 11.27 d, so the
number was a property of the detrend, not the data. The residual is now taken against
the fitted decay model, which makes an exponential score zero by construction.
- A **periodogram peak's share of in-band power does not separate signal from null** on
a 25-point curve: a true 50%-amplitude weekly cycle scores 0.25 while a noiseless pure
exponential scores 0.26. Replaced with a least-squares scan over candidate periods,
scoring the variance a sinusoid explains (`cycle_r2`) plus its amplitude as a fraction
of the curve level.
- Gates (`PERIOD_R2_GATE = 0.40`, `PERIOD_AMP_GATE = 0.10`, `PERIOD_MIN_SUPPORT = 500`)
were set from a Poisson-sampled null sweep: at n ≥ 1500 the scan stays under r2 0.47
and amplitude 0.26, while a 30% weekly sine on a tau = 4 d decay scores r2 0.44-0.48
and is detected 20/20 times at n = 2000. Support below 500 rows/bin is refused rather
than tested, because the scan's false-positive rate climbs to 0.7% there.

---

Full 8 node heterogeneous graph is used because it matches with the schema graph and is **bidirectional** which is what I want.

---

## Temporal Design

The research finding: A 2025 paper showed that a simple 2 to 8-dimensional learnable linear projection beats complex trigonometric/sinusoidal encodings across almost all benchmarks.

So SOTA models use two distinct time channels which is the;
- Global anchor: How long before the cutoff did this specific event occur?
- User-level recency: How many days has it been since this user performed this activity?

The time needs to be normalized as well with mean and sd so that it doesn't mess up the network as usual in ML.

### TODO

**Not sure what this means yet and needs further learning**

Optimize decision thresholds per activity

- pr_review_comment only occurs in ~1.9% of user records.
- A standard 0.5 decision threshold will predict all zeros for rare activities. You must search for optimal per-label decision thresholds (e.g., 0.15 for reviews, 0.45 for pushes) to maximize F1.

Measured on the all-users denominator (`temporal_output/official_window_weights_clean.csv`,
mean over the 8 fully-covered anchors) the rarity ordering is open_pr 10.2%, pr_review
7.2%, issue_comment 7.6%, open_issue 6.0%, pr_review_comment 5.1%, push 50.7%.
So `pr_review_comment` is ~5.1%, not 1.9% — the 1.9% figure came from the lossy
2026-11-01 anchor week. Threshold search still matters most for `pr_review_comment`
(~5 positives per 100 users) and least for `push`, which wants `pos_weight < 1`
(0.97) rather than a boost.

---

## Feature Design

I noticed these text patterns when printing the database, despite them all getting masked with 'a'. There is still some patterns like;
`title            str  98.6%     140 ['aaaa(aaaaa): aaaaaaaaaa aaaaaa...` These aren't useful for NLPs or Language Embedding models but maybe we can extract some statistic from these text and get a few features that would help the model learn in the future. Maybe an ablation could be done to compare if it is actually useful.

```
================================================================================
TABLE: Issues (Total Rows: 6,685)
================================================================================
         Column          Dtype Null %  Unique                              Sample
       issue_id          int64   0.0%    6685                              [0, 1]
github_issue_id          Int64   0.0%    6685  [902409756721287, 748759002810787]
        repo_id          Int64   0.0%    1959                             [2, 10]
         number          Int64   0.0%    3383                        [5588, 1173]
      author_id          Int64   0.0%    1915                             [2, 10]
is_pull_request        boolean   0.0%       2                      [False, False]
pull_request_id          Int64  97.9%     140                            [86, 96]
          title            str   0.0%    6405 ['aaaa aaa aaaaa aaaaaaaa aaaaaa...
           body            str   2.3%    6490 ['### aaaaaaa aaaaaaaaaa\n\naaaa...
```

Doing a text pattern analysis, it is clear that there are patterns that I can extract and use for the model even if it is just small.
```
Raw Text Pattern                   What It Means                        Extracted Numeric Feature
──────────────────────────────────────────────────────────────────────────────────────────────────────────
PullRequests.head_ref              74% contain '/'                      • branch_depth (count of '/')
"aaa/000-aaaaaa-aaaa"              (e.g., feature/branch vs main)       • is_nested_branch (0 or 1)

PullRequests.base_ref              Median length = 4                    • is_main_branch (len <= 5)
"aaaa"                             (target is main/master)

Issues.title                       47% contain ':'                      • has_conventional_prefix (0/1)
"aa(aaaa): aaaa..."                28.6% contain '(' and ')'            • title_length

Issues.body                        Median length = 1,595 chars          • log_body_length
"## aaaaaaa ... ```"               82.5% contain '#' (markdown header)  • has_markdown_header (0/1)
                                   79.0% contain '`' (code blocks)      • has_code_block (0/1)

PRReviewComments.diff_hunk         98.6% contain '@', '+', '-'          • log_diff_length
"@@ -00,00 +00,00 @@"              Unified git diff snippet             • diff_line_count (count of '\n')

PRReviewComments.path              Median length = 41 chars             • file_depth (count of '/')
"aa/aaa/aaa/aaaa.aa"               96.2% contain '/'
```

If extracted, these are the only 6 features with enough non-null data to be viable:
1. **`Issues.body` → `log_issue_len`:** $\log(1 + \text{char\_length})$ (Distinguishes one-line issues from detailed reports).
2. **`Issues.body` → `has_markdown_structure`:** Binary flag for `#` or ` ``` ` (Distinguishes template bug reports from casual comments).
3. **`Issues.title` → `has_prefix`:** Binary flag for `:` or `(` (Captures conventional commit/issue prefixes like `feat:` or `fix:`).
4. **`PullRequests.head_ref` → `branch_depth`:** Count of `/` (Captures branch hierarchy, e.g. `feature/user/branch` vs `main`).
5. **`PRReviewComments.diff_hunk` → `log_diff_len`:** $\log(1 + \text{char\_length})$ (Captures diff inspection depth).
6. **`PRReviewComments.path` → `path_depth`:** Count of `/` (Captures directory depth of the modified file).

## Some helpful coefficients

Before launching the training loop, configure your pipeline with these five concrete action items:

---

#### Action Item 1: Adopt the Final 252-Anchor Config

Replace any preliminary or temporary weights in your code. Update your model and loss configuration with the final parameters derived from the full 343-day corpus:

```python
import torch

# 1. Edge decay parameters for HGT message passing: w = exp(-dt_days / tau)
# Push uses the clean-anchor fit (7.50d); the other 5 use the 252-anchor fit
TEMPORAL_EDGE_TAU_DAYS = {
    'open_pr': 5.96,
    'open_issue': 5.37,
    'issue_comment': 6.02,
    'pr_review': 6.73,
    'pr_review_comment': 6.91,
    'push': 7.50,
}

# 2. Loss pos_weight for nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHTS)
# Based on the true all-users clean-anchor denominator:
POS_WEIGHTS = torch.tensor([
    7.28,   # open_pr (~12.1% prevalence)
    18.08,  # open_issue (~5.2% prevalence)
    11.05,  # issue_comment (~8.3% prevalence)
    12.83,  # pr_review (~7.2% prevalence)
    19.29,  # pr_review_comment (~4.9% prevalence)
    1.32    # push (~43.2% prevalence)
])

# Alternative: Square-root pos_weights (recommended if Micro-F1 degrades)
SQRT_POS_WEIGHTS = torch.tensor([2.70, 4.25, 3.32, 3.58, 4.39, 1.15])
```

---

#### Action Item 2: Fix the Validation Split Anchor

* **The Problem:** Your current `splits/answer.csv` is anchored at `2026-11-01`. As the audit proved, the week of `2026-11-01` had **0 usable days** in GH Archive (severe missing data). Evaluating against this week creates an artificial undercount (only 283 pushes instead of ~2,300), which will distort your validation metrics.
* **The Fix:** Point your validation set to a **clean historical window** where GH Archive was fully operational:
  * Select a cutoff within the clean stretch, such as `2026-09-23` or `2026-09-28` (or another anchor from the 252 clean anchor list).
  * Generate `splits/val_features/` and `splits/val_answer.csv` at this clean cutoff.
  * Your offline validation Macro-F1 and Micro-F1 will now match the true distribution.

---

#### Action Item 3: Build the Multi-Anchor Training Set

With 252 clean anchors available across nearly a year of history, **do not train on only one cutoff slice**:
* Select **8 to 16 diverse training anchors** spaced 14 to 28 days apart (e.g., one or two per month from January 2026 through August 2026).
* By training across multiple anchors, your model encounters different levels of project activity, seasonality, and developer turnover without overfitting to a single week.

---

#### Action Item 4: Implement Per-Activity Threshold Tuning

The competition evaluates:
$$\text{Score} = \frac{\text{Macro-F1} + \text{Micro-F1}}{2}$$

* A fixed threshold of $0.5$ across all activities will underperform because class prevalences vary from 4.9% (`pr_review_comment`) to 43.2% (`push`).
* Set up a grid search on your validation probabilities:
  ```python
  import numpy as np
  from sklearn.metrics import f1_score

  def optimize_thresholds(val_probs, val_targets):
      best_thresholds = []
      for col_idx in range(6):
          best_th = 0.5
          best_f1 = 0.0
          for th in np.linspace(0.1, 0.9, 81):
              preds = (val_probs[:, col_idx] >= th).astype(int)
              f1 = f1_score(val_targets[:, col_idx], preds, zero_division=0)
              if f1 > best_f1:
                  best_f1 = f1
                  best_th = th
          best_thresholds.append(best_th)
      return best_thresholds
  ```

---

#### Action Item 5: Save Empirical Statistics for Slides and Video

Save these exact data points from this 343-day run for your slides:
1. **Question 1 & 2 (Relational Modeling):** Note that your graph contains 8 tables, 13 relations, and processes **1,676,688 pushes, 308,761 PRs, and 176,651 issues** spanning 9,000 cohort developers.
2. **Question 3 (Temporal Modeling):** Show that across 8,232 hours and 252 clean anchors:
   * Recency decay is monotonic (exponential drop to habit floor); no weekly periodicity exists (`bump@7 < 0`, sinusoid scan failed).
   * General engagement decays rapidly ($\Delta t_{\text{any}} \tau \approx 2.2\text{--}3.5\text{d}$), whereas specific activity habits persist longer ($\Delta t_{\text{same}} \tau \approx 5.3\text{--}7.5\text{d}$).
3. **Question 4 (Pipeline & Loss):** Explain why `push` was assigned `pos_weight = 1.32` (derived from clean data where prevalence is ~43%) rather than the distorted ~30.8 from corrupted archive files.

### 3. The 2026-11-01 Validation Week Was Corrupted
* The official test anchor in `splits/answer.csv` (`2026-11-01`) happened during a severe GH Archive outage with **0 usable days**.
* **The Distortion:**
  * In that broken week, `push` showed only 283 positives (**3.14% prevalence**), which falsely suggested a `pos_weight` of **30.80**.
  * On the 252 clean anchors across the year, `push` has an average prevalence of **43.17%**, meaning the true clean `pos_weight` is only **1.32**.
* **The Hazard Prevented:** If you had trained with a `pos_weight` of 30.8, the model would have assumed pushes were extremely rare, flooding your test predictions with false positives and ruining your score.

---

# Model Architecture part

## My decisions and why
- I pick full-graph batching here because my hardware can handle the massive data.
- I also picked a layernorm + mish to map the message in a latent space
- I choose HGTConv but I have to refer a paper to explain this decision. 
  - Relational Deep Learning (RDL): Fey et al., "Relational Deep Learning: Graph Representation Learning on Relational Databases" (arXiv:2312.04615, 2023). RelBench uses HGT as its primary heterogeneous graph baseline.
  - Original HGT Paper: Hu et al., "Heterogeneous Graph Transformer" (WWW 2020).
- Concatenating is good here to prevent the message passing diluting the user specific feature of time. We don't want this diluted with other features.
- Task-specific heads linear heads branching off the shared representation so that their losses won't affect each other from experience. Weighted BCE of course because of the rarity of some predictions.
- After data analysis of the temporal design, it seems the user has no weekly periodicity. So instead of a linear projection of the temporal difference, I just use Temporal Attention Bias (Continuous Decay) to enforce that older events have less predictive value. For next week's activity.


