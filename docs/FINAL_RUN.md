# The submission run (`--mode final`, the default)

**What this is.** The run that produces `submission/submission.csv` for the graded week.
It is not a measurement run: there is no held-out week, so it reports no score. The
measurement protocol lives in [`VALIDATION_DATA.md`](VALIDATION_DATA.md) and is still
reachable with `--mode validate`.

```bash
# the submission run (~3 h on the 1x RTX PRO 6000 box)
WANDB_API_KEY=<your own key> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m scripts.run_training
```

Everything below was measured on this corpus (`raw_data/`, 8,526 hourly files,
2025-12-01 .. 2026-11-21 05:00 UTC) by `scripts/verify_raw_data.py`, by
`src.data.temporal_split.extract_labels`, or by a probe of the trainer itself. Nothing is
assumed.

---

## 1. What the run does

| step | detail |
| --- | --- |
| anchors | 18: weekly `2025-12-22 .. 2026-02-09` (the block the ablation grid and all 60 HPO trials validated), then 28-day anchors to `2026-11-14` |
| labels | 7-day window after each anchor; 124,735 labelled rows |
| observability | `--anchor-observed hybrid` (§3) drops a label cell only when the evidence says it was damaged |
| loss | masked BCE over observed cells, `pos_weight` from this run's own anchors (`--pos-weights auto`) |
| schedule | 310 epochs, warm restarts (`t0=5, x2`), lr 1e-3, 1 HGT layer, hidden 256 / latent 128 / head 128 / 4 heads, dropout 0.4, weight decay 3e-5 — the HPO/ablation consensus defaults |
| thresholds | fitted **after** training, per activity, on the anchors of the last 84 days (§4); no validation week is used or needed |
| output | `submission/submission.csv`, `splits/test_probs_<cutoff>.csv`, `checkpoints/hgt_final_2026-11-14.pt` (written every 50 epochs and at the end) |

The graded week is **2026-11-22 00:00 – 11-29 00:00 Taiwan time**, i.e. **2026-11-21 16:00
.. 11-28 16:00 UTC**, so the feature cutoff is `2026-11-21 16:00` and the last legal training
anchor is `2026-11-14` (its label window ends exactly at the graded week's start).

## 2. What the data looks like now (the verification this run depends on)

`scripts/verify_raw_data.py --workers 48` over the full corpus:

```text
hours scanned        : 8,526        unreadable files: 0
usable hours/activity: open_pr 77.6%  open_issue 75.4%  issue_comment 78.6%
                       pr_review 79.5%  pr_review_comment 75.0%  push 77.8%
days clean for all six: 58 / 356
by month (clean days):  2025-12: 20-31   2026-03: 8-31   2026-07: 11-19
                        2026-01: 19-31   2026-04: 4-30   2026-08: 0-10
                        2026-02: 19-28   2026-06: 1-26   2026-09: 2-12
                                                        2026-10: 7-9
                                                        2026-11: 0  <-- nothing clean
```

November 2026 is downloaded and **is not clean** — and the way it fails decides the whole
final-run design. `PullRequestEvent`, `IssuesEvent`, `IssueCommentEvent` and
`PullRequestReviewEvent` arrive at their **October absolute volume**; `PushEvent` and
`PullRequestReviewCommentEvent` do not:

| per-hour median (2026) | Oct | Nov | ratio |
| --- | --- | --- | --- |
| total events | 39,493 | 4,098 | **0.10** |
| `PushEvent` | 28,884 | **0** | **0.00** |
| `PullRequestEvent` | 1,501 | 1,505 | 1.00 |
| `IssuesEvent` | 615 | 621 | 1.01 |
| `IssueCommentEvent` | 754 | 697 | 0.92 |
| `PullRequestReviewEvent` | 366 | 356 | 0.97 |
| `PullRequestReviewCommentEvent` | 232 | 93 | **0.40** |

Per-day cohort label counts say the same thing in the unit that matters: October anchors
produced 306-447 `open_pr` positives, November anchors 356-371; `push` fell from 2,058-2,183
to 315-457 and `pr_review_comment` from 143-197 to 65-98.

**Why this breaks the audit and not the labels.** The audit judges an hour by the *share* of
events of one type against the local (±7 day) median share. Push is ~90% of a normal hour, so
losing push divides the denominator by ten and every surviving type's share rises 4-7x — above
the audit's `max_activity_ratio=4.0`. Result: `usable=False` for all six activities on all 21
November days, including the four that lost nothing. That is a denominator artifact, not a
label defect, and §3 is how the run tells the two apart.

## 3. `--anchor-observed hybrid`: audit **or** label-count sanity

A cell is observable when **either** verdict says so:

1. **audit** — every day of the 7-day label window is `usable` for that activity in
   `raw_data/coverage_activity.csv` (the rule `VALIDATION_DATA.md` prescribes), or
2. **label-count sanity** — the anchor's positive count is within `0.6 .. 2.0` of the median
   of its ±4 neighbouring anchors, excluding itself (`count_sanity_flags`), which is zero
   positives or a genuine collapse.

The union is what lets the sparse regime in while still refusing a dead stream. Per anchor:

| anchor | rows | observed cells | dropped |
| --- | --- | --- | --- |
| 2025-12-22 .. 2026-07-27 (14 anchors) | 4,320 .. 9,000 | 25,920 .. 54,000 (100%) | — |
| 2026-08-24 | 9,000 | 9,000 (16.7%) | open_pr, issue_comment, pr_review, pr_review_comment, push |
| 2026-09-21 | 9,000 | 45,000 (83.3%) | pr_review_comment |
| 2026-10-19 | 9,000 | 45,000 (83.3%) | pr_review_comment |
| 2026-11-14 | 9,000 | 45,000 (83.3%) | **push** |

`pr_review_comment` stays observable on 2026-11-14 (77 positives = 0.68x its neighbourhood,
inside the band). That is deliberate: the test-week labels come from the same degraded feed,
so matching its under-count is more consistent than correcting it.

## 4. Thresholds, and why they are fitted on the last 84 days

`open_pr` prevalence in the 7-day label window:

| anchor | open_pr | open_issue | issue_comment | pr_review | prc | push |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | **15.2%** | 6.2% | 11.6% | 9.8% | 7.0% | 36.5% |
| 2026-07-13 | 1.7% | 1.4% | 1.4% | 1.1% | 0.7% | 38.6% |
| 2026-10-19 | 3.4% | 2.5% | 2.0% | 2.4% | 1.8% | 22.9% |
| 2026-11-14 | **4.0%** | 2.3% | 2.4% | 2.7% | 0.9% | — |

A 3.8x regime shift. A threshold that maximises F1 over the pooled history reflects the dense
winter weeks and over-predicts on a sparse test week, so `--threshold-lookback-days 84`
(default) fits each activity's threshold on the anchors from `2026-08-22` onward
(2026-08-24, 09-21, 10-19, 11-14) while the **model** still trains on all 18 anchors.

The fitter is `optimal_thresholds_per_activity`, not the CMA-ES of the validation protocol:
the joint objective is the TA score, which needs all six activities of a row at once, and the
late anchors have unobserved columns. F1 for one activity depends only on that activity, so
each threshold is fitted on exactly the rows where it was measured — an exact 1-D search over
the sorted scores, `O(n log n)`, no budget or seed.

## 5. Cost, measured on this box

* **VRAM**: 25.6 GiB per million graph nodes (probe: 197k→5.1, 629k→16.1, 1.23M→31.4,
  2.0M→51.2, 2.59M→66.3 GiB). The largest anchor is 66.3 GiB, so a co-tenant process on the
  GPU is fatal; `preflight_largest_anchor` runs one forward+backward on it *before* training
  and refuses to start otherwise.
* **Time**: 1.3 s per million nodes per epoch (0.30 s at 197k, 0.85 s at 629k, 1.60 s at
  1.23M, 3.26 s at 2.59M). ~24M nodes per epoch → **~31 s/epoch → ~2.7 h for 310 epochs**,
  plus ~3 min of graph building and ~1 min of inference.
* **Host RAM**: 1.48 GB for all 18 graphs (kept on CPU, moved one at a time).

## 6. What to check when it finishes

```text
Submission written to submission/submission.csv
predicted positives per activity:
  open_pr                ...   theta ...
Format check: no errors (4465 users)
```

* 4,465 rows, no missing/duplicate `UserID`, `scripts/evaluate.py --input` clean.
* Predicted positive rates should land within a few points of the base rates in §4
  (F1-optimal thresholds deliberately over-predict the base rate when discrimination is
  weak, so a bit above it is expected, not a bug).
* The checkpoint carries `thresholds`, so predictions can be refreshed without retraining:

```bash
# after downloading/preprocessing fresher hours
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.run_training --predict-only
```

## 7. Known limits, stated honestly

1. **The feature window is 10.2 hours stale.** The corpus ends 2026-11-21 05:49 UTC and the
   graded week starts 16:00 UTC; GH Archive publishes an hour only after it completes, so the
   hours 06:00-16:00 of 21 November cannot exist before the deadline. Every user's
   `days_since_*` is therefore 0.42 days too large and `count_7d_*` misses up to 10 hours.
   This is the same for all users and small next to the 7-day horizon, but it is real.
2. **`push` and `pr_review_comment` are the weak columns.** Push labels stop being trustworthy
   around 2026-10-27 and PRC loses ~2.5x of its positives in November. If the grader treats a
   missing test-week activity as unscored (README §4), they cost nothing; if it scores them
   against an all-zero truth, only a zero prediction can earn F1 there, and the model does not
   know that in advance.
3. **No number here is a score.** February-15 F1 (~65.8% at CMA-ES in the ablation grid) is a
   dense-regime measurement and does not transfer to a sparse test week; the calibrated
   threshold is the only defence, and it is fitted in-sample.
4. **18 anchors is a deviation from the validated 8.** It was chosen to cover the sparse
   regime within the ~3 h budget, not because it was measured to be better. The dense block
   the ablation grid validated is still there, and per-cell masking keeps the damaged late
   anchors from teaching wrong labels.
