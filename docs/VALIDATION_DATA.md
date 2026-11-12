# Which data to train on and which to validate on

**Read this before any training run.** It records what is wrong with the recent data,
which weeks are safe to score against, and the exact cutoffs to use. Everything here is
measured from `raw_data/` by `scripts/verify_raw_data.py`, not assumed.

Audience: a training agent (or a human) picking a training window and a validation
cutoff. Follow the decision box, then run the checklist in §6 before believing any score.

---

## 1. Decisions (the short version)

| Question | Answer |
| --- | --- |
| Which validation cutoff? | **`--cutoff 2026-02-15`** (label window 2026-02-15 .. 2026-02-22) |
| Second choice | `--cutoff 2026-03-08` (window 2026-03-08 .. 2026-03-15) |
| Latest cutoff that still has a fully usable 7-day window for all six activities | **2026-03-14** |
| Any cutoff after that? | **No 7-day cutoff exists.** The only recent window usable for all six is 2026-10-01 .. 10-05 — five days, sanity checks only (§4.2). Never November. |
| Which months to train on? | All of 2025-12-01 .. 2026-10-31, with the per-activity coverage flags applied (§5). Do **not** drop whole months — see §7. |
| Which data must never produce a *label*? | 2026-10-22 .. 2026-10-31, and any November week |
| Is the test week predictable from the current corpus? | **No.** The test week needs features through ~2026-11-21, which are not on disk (§8). |
| Is `splits/answer.csv` usable? | **No — stale and invalid.** It is anchored at 2026-11-01, a window built from data that has since been deleted (§9). |

If a training run reports a validation score from a cutoff outside this list, treat the score
as unmeasured rather than as a bad score.

---

## 2. Why November 2026 cannot be validation data

The failure is not corrupt files. `data.gharchive.org` serves those hours incomplete, so a
re-download reproduces them exactly (verified by md5 on three files). The event *type* is
missing, not the file.

### 2.1 `PushEvent` goes to exactly zero

Push rows per day, with hours that are present but contain no `PushEvent`:

| day | events | `PushEvent` | push share |
| --- | --- | --- | --- |
| 2026-10-26 | 824,353 | 630,483 | **0.765** |
| 2026-10-27 | 82,128 | **0** | 0.000 |
| 2026-10-28 | 81,054 | **0** | 0.000 |
| 2026-10-29 | 94,066 | **0** | 0.000 |
| 2026-10-30 | 98,714 | **0** | 0.000 |
| 2026-10-31 | 107,736 | **0** | 0.000 |

Every day 2026-11-01 .. 2026-11-08 is the same: 24 files present, push share 0.00–0.55.
The hourly event volume also collapses to ~15% of normal, and the surviving events are
concentrated in a handful of event types.

### 2.2 The damage is not confined to push

Labels extracted for the same hypothetical window, over the 9,000 cohort users:

| label window | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-01 .. 10-05 (5 usable days) | 445 | 253 | 318 | 288 | 215 | **2,254** |
| 2026-10-27 .. 11-03 | 355 | 204 | 242 | 258 | **85** | **24** |

`push` loses 94× of its positives, and `pr_review_comment` loses 2.5×. A validation week in
this range does not measure the model — it measures how much of GH Archive survived.

### 2.3 Why this specifically breaks a *temporal* model

The features are "how long ago did this user's last event of type *a* happen". When 5–10
days of a stream are missing, every affected user looks **staler than they were**. The
consequences are not random noise:

- `Δt` is inflated, so `exp(-Δt / τ)` edge weights collapse toward zero for exactly the
  users who were most active — the model learns that recency does not matter.
- Positives fall, so `pos_weight = (1-p)/p` shifts. Measured on 2026-11-01, push prevalence
  was 3.14% where a clean week gives 30–50%; a `pos_weight` fitted there is ~10× too high.
- A metric computed on such a window mostly rewards "predict 0".

This is why the truncation at 2026-10-31 was the right call, and why **2026-10-22 ..
2026-10-31 must also be excluded** — the push outage starts on Oct 27, five days before the
cut. Truncating at Oct 31 removes November but leaves the broken tail in place.

---

## 3. What "usable" means here

`scripts/verify_raw_data.py` audits all six activity types **independently** per hour and
writes `raw_data/coverage.csv` (per hour) and `raw_data/coverage_activity.csv` (per day ×
activity). Definitions, as implemented:

- **hour passes `ok_<activity>`** — the hour carries ≥30% of that activity's *local*
  (±7-day) median density and ≤4× it. Local, because density drifts all year: a
  corpus-wide baseline marks an entire sparse month as broken. Two-sided, because too
  *many* events is also damage — 2026-09-04 .. 09-06 carries ~1,400 `PullRequestEvent`s/day
  against a ~110/day neighbourhood while pushes vanish, which is redistribution, not
  activity.
- **day is `clean` for an activity** — every hour of that day passes.
- **day is `usable` for an activity** — the day's total sits inside the band and ≥75% of
  its hours pass. `usable` exists because 24/24 on a typical 19/24 day has probability
  ~0.4%; requiring it makes whole months read as unusable.

Current corpus state (2025-12-01 .. 2026-10-31, 8,040 hourly files, 335 days):

| | count |
| --- | --- |
| hours scanned | 8,040 |
| hours sane for all six activities | **5,050 (62.8%)** |
| unreadable files | **0** |
| days usable for all six | **181 / 335** |
| days clean for all six | 58 / 335 |
| longest run usable for all six | **2026-02-05 .. 2026-02-28 (24 d)** |
| last day usable for all six | **2026-10-21** |

Per-activity coverage:

| activity | clean days | usable days | latest cutoff with a usable 7-day window |
| --- | --- | --- | --- |
| `open_pr` | 139 | 263 | 2026-08-08 |
| `open_issue` | 135 | 248 | 2026-08-08 |
| `issue_comment` | 161 | 269 | 2026-08-08 |
| `pr_review` | 166 | 275 | 2026-08-08 |
| `pr_review_comment` | 121 | 257 | 2026-08-08 |
| `push` | 240 | 256 | 2026-09-30 |

The per-activity column is why "the model scored well" needs a caveat: `push` alone can be
validated as late as 2026-09-30, the other five cannot. Scoring is macro-F1 over all six,
so a validation window must cover all six to mean anything.

---

## 4. Choosing the validation window

A cutoff needs **7 consecutive usable days**. Runs of days usable for all six activities
(`(start, length)`), the only places a cutoff can go:

```text
2025-12-02 .. 12-14 (13d)   2026-02-05 .. 02-28 (24d)   2026-05-13 .. 05-17 (5d)
2025-12-16 .. 12-31 (16d)   2026-03-02 .. 03-20 (19d)   2026-06-11 .. 06-16 (6d)
2026-01-02 .. 01-23 (22d)   2026-04-16 .. 04-19 (4d)    2026-09-16 .. 09-18 (3d)
2026-01-25 .. 02-03 (10d)   2026-05-01 .. 05-04 (4d)    2026-09-22 .. 09-25 (4d)
                            2026-05-08 .. 05-11 (4d)    2026-10-01 .. 10-05 (5d)
                            2026-05-21 .. 05-24 (4d)    2026-10-11 .. 10-13 (3d)
                            2026-05-29 .. 05-31 (3d)    2026-10-20 .. 10-21 (2d)
```

68 cutoffs satisfy the 7-day rule. Ranked candidates:

| cutoff | label window | usable lookback (30 d) | verdict |
| --- | --- | --- | --- |
| **2026-02-15** | 02-15 .. 02-22 | 28–30 of 30 per activity | **recommended** |
| 2026-03-02 | 03-02 .. 03-09 | 29–30 of 30 | good, similar regime |
| 2026-03-08 | 03-08 .. 03-15 | 29–30 of 30 | good, last comfortable |
| 2026-03-14 | 03-14 .. 03-20 | 29–30 of 30 | **latest legal cutoff** |
| 2026-01-15 | 01-15 .. 01-22 | 12–13 of 30 | usable, thinner lookback |
| 2026-10-01 | 10-01 .. 10-08 | 12–17 of 30 | **illegal** — 10-06..08 are not usable |

### 4.1 The catch you must know about: regime mismatch

Recommendation is February because it is the *last* dense stretch with a full 7-day
window — not because it resembles the test week. It does not:

| month | median `PullRequestEvent`s/day | median `PushEvent`s/day |
| --- | --- | --- |
| 2025-12 | 289,057 | 2,424,468 |
| 2026-01 | 293,888 | 2,388,484 |
| 2026-02 | 295,045 | 2,378,689 |
| 2026-03 | 192,415 | 2,726,882 |
| 2026-06 | 34,798 | 3,240,820 |
| 2026-07 | 7,151 | 3,679,042 |
| 2026-08 | 7,856 | 2,864,872 |
| 2026-09 | 76,985 | 1,397,530 |
| 2026-10 | 40,117 | 1,158,959 |

November 2026 sits in the sparse regime, near Oct (40k PR/day) or worse. February has
**7.4×** the PR volume. So a February F1 is optimistic relative to the test week, and the
per-activity prevalence differs accordingly:

| activity | Feb-15 validation (9,000 users) | 2026-11-01 window (broken) |
| --- | --- | --- |
| `open_pr` | 1,368 · 15.20% | 377 · 4.19% |
| `open_issue` | 560 · 6.22% | 198 · 2.20% |
| `issue_comment` | 1,042 · 11.58% | 256 · 2.84% |
| `pr_review` | 885 · 9.83% | 273 · 3.03% |
| `pr_review_comment` | 629 · 6.99% | 105 · 1.17% |
| `push` | 3,288 · 36.53% | 283 · 3.14% |

Practical rule: **tune architecture, thresholds, and `pos_weight` on Feb-15, but expect the
test-week prevalence to be roughly a third of it, and re-derive `POS_WEIGHTS` from
`temporal_output/official_window_weights_clean.csv` (§5) rather than from the February
window.** For a regime check, also score the copy-last-week baseline on the Oct 1–5 island
(§4.2) and compare *rankings* between the two eras, not absolute F1.

### 4.2 If you need a recent-regime sanity check

The only recent window fully usable for all six is 2026-10-01 .. 10-05 — five days, not
seven, so `make_split --days 5 --cutoff 2026-10-01`. Use it to confirm the model has not
regressed catastrophically in the sparse regime. Do not tune on it: 5 days × 9,000 users
with ~4% prevalence is roughly 450 positives, and `pr_review_comment` has 215.

---

## 5. How to apply the per-activity flags

Never filter at month granularity. Apply the flags:

```python
# per-activity, per-day usability from the audit
cov = pd.read_csv("raw_data/coverage_activity.csv")          # day, activity, clean, usable
usable = cov.pivot_table(index="day", columns="activity", values="usable").astype(bool)

# For LABEL generation: drop a (day, activity) cell only when that activity's
# stream was damaged that day. A day can be a valid `push` day and an invalid
# `pr_review_comment` day.
```

`src/data/activity_decay_analysis.py` implements exactly this as
`anchor_usability(anchors, per_day, mode, horizon_days, max_delta)`: it returns a
DataFrame indexed by anchor with one boolean column per activity, and a damaged cell
contributes no rows and no positives (recorded as NaN, not 0 — "unobserved" and "did not
happen" are different facts).

For **features** the same flags apply more weakly: a lost hour makes `Δt` too large, but
dropping the row is not better. Accept the noise and keep the dense history. The exception
is the label side, where a missing hour silently manufactures negatives.

Loss weights and edge taus, already measured on this corpus:

```python
# edge weight = exp(-dt_days / tau); dt = days since the user's own last event of that type
TEMPORAL_EDGE_TAU_DAYS = {
    'open_pr': 5.96, 'open_issue': 5.37, 'issue_comment': 6.02,
    'pr_review': 6.73, 'pr_review_comment': 6.91, 'push': 7.50,
}

POS_WEIGHTS = {   # BCEWithLogitsLoss(pos_weight=...), all-users clean-anchor denominator
    'open_pr': 7.28, 'open_issue': 18.08, 'issue_comment': 11.05,
    'pr_review': 12.83, 'pr_review_comment': 19.29, 'push': 1.32,
}
```

`push` takes its tau from clean anchors only (the audit finds its stream damaged on more
days than the other five); the other five use the all-anchor fit. See
`temporal_output/summary.md` for the derivation and `activity_summary.json` for the raw fits.

---

## 6. Checklist before trusting a training score

Run these in order. Any failure means stop, do not tune on the number.

1. **Does the label window have 7 usable days for all six?**
   `python scripts/verify_raw_data.py --workers 32` then check `raw_data/coverage_activity.csv`.
   A cutoff whose window includes a non-usable day is not a validation window.
2. **Did the split come from an explicit cutoff?** `make_split` defaults to
   `last_data_day - 7`, which is now **2026-10-24**. That window (10-24 .. 10-31) contains
   only three live push days: measured positives are open_pr 325, open_issue 192,
   issue_comment 223, pr_review 227, **pr_review_comment 68**, **push 1,074** — against
   178 and 1,794 for the fully live week before it. Any cutoff at or after **2026-10-27**
   drops push to ~0 positives and is not a split at all. Always pass `--cutoff`.
3. **Did the leakage check pass?** `make_split` prints
   `leakage check: OK - no feature is known at/after the cutoff`. It refuses to write on failure.
4. **Are the positives plausible?** February-scale values are 1,300 / 560 / 1,040 / 885 /
   630 / 3,290. Values in the low hundreds for `push` mean you are reading a broken window.
5. **Is every per-activity prevalence non-zero but below ~50%?** 0% is damage, 90%+ is
   usually a leaked label rather than a good model.
6. **Compare against the published baselines.** README reference scores on
   2026-09-27 .. 10-04: predict-all-0 = 0.00, random = 0.2126, GNN without time encoding
   = 0.3570, copy-last-week = 0.4961, GNN with time encoding = 0.5482. Landing below
   copy-last-week on a *usable* window means something is broken, not that the model is weak.

---

## 7. Why not just delete the bad months?

Tempting, and wrong. The density table in §4.1 shows the archive collapsing from 295k to
7k `PullRequestEvent`s/day between February and July. That is a property of GH Archive's
sampling, not of GitHub users, but it is *stable within each month*, so a per-day local
baseline can tell "this month is sparse" apart from "this day lost an event type".

Deleting months would:

- remove 2026-03 .. 2026-10 entirely, which is most of the temporal diversity;
- leave the sparse regime unrepresented while the test week is sparse;
- not help at all with the actual damage, which is *hour- and activity-level*, not
  month-level (2026-10 has 8 clean `open_pr` days and 31 broken ones).

Apply the flags, keep the rows.

---

## 8. Blocker for the actual test week

README §4: the test week is **2026-11-22 00:00 – 2026-11-29 00:00 Taiwan time**, i.e.
**2026-11-21 16:00 UTC .. 2026-11-28 16:00 UTC**.

Features for that prediction need rows from before ~2026-11-21. The corpus now ends
**2026-10-31 23:55**. So:

- 2026-11-01 .. 11-08: deleted, and provably broken at source (§2).
- 2026-11-09 .. 11-21: **never downloaded.** Nothing is known about these 13 days yet.

Training and validation can proceed on the current corpus. Producing a test submission
cannot: every user's `Δt` would be at least 21 days stale, far outside the 0..30 d range the
decay curves were fitted on, and `copy last week`-style features would read all zeros.

Before the final run, download and audit 2026-11-09 .. 11-21:

```bash
# edit START_DATE/END_DATE in scripts/get_raw_data.py (CRLF file, patch bytes not with `edit`)
python scripts/get_raw_data.py
python scripts/verify_raw_data.py --workers 32
```

Then check `raw_data/coverage_activity.csv` for those days. If any of the six activities is
damaged there, note that README §4 says an activity whose data is missing **in the test
week** is not scored — so damage in the *feature* window is a modelling problem, while
damage in 11-22 .. 11-29 removes that activity from the score entirely.

---

## 9. Artifacts in this repo that are now wrong

| artifact | state | what to do |
| --- | --- | --- |
| `splits/answer.csv` | anchored 2026-11-01, built from deleted November rows | regenerate with an approved cutoff |
| `splits/features/*.parquet` | same cutoff | regenerate |
| `processed_tables/db` | still holds November rows (max `created_at` 2026-11-08) | re-run `scripts/preprocess.py --raw raw_data --out processed_tables` |
| `raw_data/coverage*.csv` | current (8,040 hours, no November) | fine — regenerate after any new download |
| `temporal_output/` | curves computed on the pre-truncation corpus (343 days, incl. November); anchors stop at 2026-09-30 | re-run `python -m src.data.activity_decay_analysis` after rebuilding tables |

`TEMPORAL_EDGE_TAU_DAYS` and `POS_WEIGHTS` in §5 are **not** invalidated by the truncation:
they come from clean anchors spanning 2025-12-02 .. 2026-09-30 (252 anchors, 9,000 users),
all of which predate the broken tail. Only the unfiltered comparison pass and the
`official_window_weights.csv` row for 2026-11-01 read data that no longer exists — regenerate
the curves when convenient, but you can copy §5 into model config today.

---

## 10. Multi-anchor training: what we chose and why

**Decision.** Validate on one week (`2026-02-15`, one of the nine fully clean anchors), and
train on *many* rolling anchors: one per weekly cutoff from `2025-12-22` up to the week before
the validation cutoff. Each anchor is its own graph. Label cells the audit marks unusable are
**dropped from the loss**, not zeroed.

```text
anchor 2025-12-22   features < 12-22   labels 12-22 .. 12-29
anchor 2025-12-29   features < 12-29   labels 12-29 .. 2026-01-05   (open_pr unobserved)
...
anchor 2026-02-08   features < 02-08   labels 02-08 .. 02-15
val    2026-02-15   features < 02-15   labels 02-15 .. 02-22          (clean, never trained on)
```

Implemented in `scripts/run_training.py` (`--anchors`) and
`src/models/train.py` (`train_model_multi_anchor`, `Anchor`); the usability verdicts come from
`src/data/label_coverage.py`, which wraps `anchor_usability`.

### 10.1 Why not a single training week

One anchor gives one week of labels for the users who exist at that cutoff. Measured on this
corpus:

| setup | labelled rows | cohort coverage | `user_age_days` seen |
| --- | --- | --- | --- |
| single anchor (2026-02-08) | 5,706 | 5,873 / 9,000 (65%) | 0 .. 75 |
| 6 weekly anchors (Jan 04 .. Feb 08) | 31,387 | 4,744 .. 5,706 | 0 .. 103 |
| 14 weekly anchors (Dec 08 .. Mar 09) | 72,154 | 3,203 .. 6,420 | 0 .. 159 |

Three consequences, all measured rather than assumed:

1. **The cohort is a moving target.** Users present by the cutoff grow from 3,203 (2025-12-08)
to 6,420 (2026-03-09) and 9,000 by 2026-06-01. A user who joined on 2026-01-20 has
`user_age_days ~= 26` and sparse history; a February-only model never sees that regime, yet the
test week is entirely made of such users.
2. **Prevalence drifts, so calibration drifts.** Per-anchor prevalence spans `push` 47.7% ..
72.9%, `open_pr` 19.1% .. 32.2%, `pr_review_comment` 8.2% .. 18.3%. The `POS_WEIGHTS` in §5
were fitted across many clean anchors, so they describe an *average* anchor; a single-week
model has no reason to agree with them.
3. **Thresholds stop overfitting one week.** CMA-ES tunes theta on the validation rows only.
When the model has seen one week, the logit distribution is narrow and theta absorbs that
week's noise — in the 3-epoch single-anchor run `open_pr` pinned at 0.012, i.e. the lower
bound. More anchors widen the distribution and make theta a calibration parameter again.

### 10.2 Why not start in December, and why not end in October

**Earlier than 2025-12-22 is censored.** `days_since_*` is capped at 30.0 and data starts
2025-12-01, so anchor 2025-12-08 can express at most 7 days of staleness and 2025-12-22 at
most 21. Teaching the model on those windows teaches it that recency does not matter — the
exact failure mode §2.3 warns about. Hence the default `--anchor-start 2025-12-22`.

**Later anchors have no usable labels.** Applying "every day of the window must be usable for
that activity" to a 14-day stride from 2025-12-22 to 2026-10-18 keeps **67 of 264**
(anchor, activity) cells. `2026-08-03`, `2026-08-17`, `2026-08-31`, `2026-09-14` and
`2026-10-12` keep **zero** activities; the 2026-10-24 window keeps 5 of 42 day-cells with
`pr_review` at 0/7 days. Training there would teach "nobody pushes" — and `push` is 53% of the
cohort, so it carries most of micro-F1.

If you want the later anchors anyway (`--anchor-end 2026-10-18`), the trainer prints a loud
warning: training on data from *after* the validation cutoff makes the score retrospective
rather than a forecast. Use it to sanity-check thresholds and feature scaling, never to report
a result.

### 10.3 The masking rule, and what it does to the loss

A label is a count over 7 days. A damaged day can only lose positives, so it silently turns
"did open a PR" into "did not". Marking the cell **unobserved** rather than 0 is what makes a
partially damaged anchor usable: `2026-01-19` keeps five activities and drops `open_pr`, so it
contributes 25,850 observed cells of a possible 31,020 instead of poisoning all six columns.

In code (`Anchor.observed`):

- loss = `binary_cross_entropy_with_logits(..., reduction="none")` masked by `observed`,
  normalised by the number of observed entries;
- anchor `i` contributes weight `n_observed_i / sum_j n_observed_j`, so the epoch objective is
  the exact mean BCE over every observed (user, activity) entry;
- the *train* score is computed only on rows where **all six** activities are observed, because
  the TA metric has no notion of an unobserved cell. Validation anchors must be fully observed
  and the trainer refuses to run otherwise.

Deterministic when seeded; a 3-anchor, 3-epoch smoke run printed
`total labelled rows 16,574`, `resident graph memory: 0.15 GB`, and
`Score: 53.49%` from `scripts/evaluate.py` — identical to the in-memory number.

### 10.4 Cost

Measured graph footprint and step time on this corpus (128/64/64 HGT, 2 layers, 4 heads):

| anchor | nodes | graph | build | train step |
| --- | --- | --- | --- | --- |
| 2025-12-22 | 197,312 | 12.6 MB | 3.2 s | — |
| 2026-03-02 | 853,949 | 55.7 MB | 7.0 s | — |
| 2026-06-01 | 1,932,031 | 125.3 MB | 11.6 s | — |
| 2026-10-05 | 2,529,962 | 163.8 MB | 13.4 s | — |
| 2025-12-08 (step probe) | 69,587 | — | 1.8 s | 236 ms |
| 2026-03-09 (step probe) | 931,752 | — | 7.7 s | 2,568 ms |

Mean 80.5 MB per anchor, so the default 8-anchor Dec 22 .. Feb 08 span keeps ~0.2 GB resident
and 40 anchors would need ~3.1 GB. Graphs live on CPU and are moved to the device one at a
time, so GPU memory is bounded by the largest single anchor, not by the anchor count.

Default run (7-8 anchors) is ~5 s of graph building plus ~2-4 s per epoch on the validation
machine; the dominant cost remains validation + CMA-ES (3,008 score evaluations, ~2 s).

### 10.5 What this does *not* fix

The test week is 2026-11-22 .. 11-29 and no data exists for it (§8). Multi-anchor training
improves the model that will consume November features; it does not create them. And it does not
shift `user_age_days` coverage to 354 days — the clean late anchors that would do that do not
exist. Until 2026-11-09 .. 11-21 is downloaded, audited and preprocessed, there is no test-week
inference path, and §8 stands unchanged.

---

## 11. One-line summary

Train on 2025-12-01 .. 2026-10-31 with per-activity coverage flags; validate at
`--cutoff 2026-02-15` (never later than 2026-03-14); treat 2026-10-22 .. 2026-10-31 and
everything after it as unobserved, not as negative; and do not attempt a test-week
submission until 2026-11-09 .. 11-21 has been downloaded and audited.
