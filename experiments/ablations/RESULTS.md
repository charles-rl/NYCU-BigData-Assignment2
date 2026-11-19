# Ablation + baseline results

24 cells: 7 GNN variants × 3 seeds (42 / 123 / 999) plus a LightGBM baseline × 3
seeds, all on the documented validation split — cutoff **2026-02-15**, 7-day
label week, 8 training anchors (2025-12-22 … 2026-02-08), 6,736 test users, 5,873
validation users.

Scores are the assignment metric (mean per-activity F1 over the six activities):

- **`@0.5`** — every activity thresholded at 0.5.
- **`@CMA-ES`** — per-activity thresholds fitted by CMA-ES (`src/utils/
  threshold_optimizer.py`) on **training-anchor rows only** (8 anchors, fully
  observed rows, ~40.4k rows), then applied unchanged to the validation week.
  The optimizer never sees a validation label or probability
  (`train_engine._fit_thresholds_on_train`), so this is a held-out number, not
  a tuned-on-validation one. This is the value the suite reports as
  `score_cmaes` and the one used for ranking below. Both arms in this directory
  use the same protocol; the LightGBM baseline fits theta the same way.

## Where these numbers were produced

**Machine: 4× NVIDIA GeForce RTX 3090 (24 GiB each, sm_86 Ampere).** This is the
*new* machine described in [`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md) —
**not** the original 1× RTX PRO 6000 Blackwell (96 GiB, sm_120) box this training
code was developed on. Every number in this directory was measured on the 3090
box, so read the machine row before comparing these scores with anything
produced elsewhere.

| | this machine |
| --- | --- |
| GPUs | 4× GeForce RTX 3090, 24,576 MiB each, no NVLink |
| driver / torch wheel | 580.178.04 / `torch 2.11.0+cu128` (its arch list contains sm_86) |
| CPU / RAM | 48 threads, 2× Intel Xeon E5-2650 v4 @ 2.20 GHz / 251 GiB |
| OS | Ubuntu, kernel 7.0.0-42-generic, glibc 2.43 |
| env | conda env `bda_asg2`, python 3.11.18, torch-geometric 2.8.1, numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, lightgbm 4.7.0 (pip wheel → CPU-only), cma 4.5.0 |
| repo revision | `5e46c58` (the 21 original cells) / `fc20c3c` (the 3 calibration cells) |
| clock | first cell finished 2026-11-16 20:03 UTC, last 2026-11-18 20:36 UTC |
| raw corpus | 2025-12-01 → **2026-03-21**, 2,664 hourly GH-Archive files (93 GB) |

What that machine changes, concretely:

1. **24 GiB per card is the binding constraint.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   is required for every GNN cell here (without it, fragmentation alone costs
   ~1.5 GiB and the 2-layer arm dies inside `hgt_conv.py` message passing). The
   `depth-2layer` arm additionally needs `--grad-checkpoint
   --keep-val-graph-on-host`; both are numerically neutral (non-reentrant
   recompute keeps dropout masks identical, host-parking is placement only), but
   they are why those three cells took 112 min instead of ~70 min.
2. **One process per GPU.** Two GNN cells on one card makes the second stall
   ~55 min before its first epoch without ever raising `OutOfMemoryError`. The
   grid therefore ran as separate shards pinned with `CUDA_VISIBLE_DEVICES`;
   there is no DDP in this repo.
3. **Per-cell wall time here:** ~68 min for each 1-layer arm (310 epochs),
   ~112 min for `depth-2layer`, 3.8 min for LightGBM. Planned makespan on 3 GPUs
   is 7.1 h; actual elapsed was ~22 h because the 2-layer arm had to be
   re-run after the OOM above. The three calibration cells were added later and
   ran as a single shard on one card (`--skip-existing`, 3 × ~67 min = 3.4 h).
4. **The corpus ends 2026-03-21** (this box's free disk cannot hold the full
   2025-12-01 → 2026-10-31 download, ~221 GB), so 2026-02-15 was the latest
   *approved* cutoff available — see `docs/VALIDATION_DATA.md`.

## Results (mean of 3 seeds, %)

| variant | macro-F1 | micro-F1 | score @0.5 | score @CMA-ES | Δ@CMA | Δ@0.5 | min/cell | params | @CMA spread | beats `full` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 63.06 | 68.36 | 57.28 | **65.71** | — | — | 68.3 | 2,304,630 | 65.63–65.80 | — |
| pos-weights-auto | 63.16 | 68.51 | **61.54** | 65.84 | +0.13 | **+4.26** | 66.9 | 2,304,630 | 65.79–65.91 | 3/3 |
| depth-2layer | 63.24 | 68.40 | 56.96 | 65.82 | +0.11 | −0.32 | 112.4 | 3,048,166 | 65.69–65.90 | 2/3 |
| no-temporal-decay | 63.07 | 68.48 | 57.05 | 65.77 | +0.06 | −0.23 | 67.9 | 2,304,630 | 65.66–65.88 | 2/3 |
| depth-1layer | 62.99 | 68.37 | 57.00 | 65.68 | −0.04 | −0.28 | 67.6 | 2,304,630 | 65.61–65.73 | 1/3 |
| no-text | 62.93 | 68.33 | 57.28 | 65.63 | −0.09 | +0.00 | 67.8 | 2,304,630 | 65.58–65.71 | 0/3 |
| no-residual-fusion | 62.87 | 67.97 | 55.87 | 65.42 | **−0.29** | **−1.41** | 68.4 | 2,288,502 | 65.32–65.53 | 0/3 |
| lightgbm | 60.33 | 66.33 | 59.86 | 63.33 | **−2.38** | **+2.58** | 3.8 | 1,860 | 63.11–63.45 | 0/3 |

**Noise floor:** `full` alone scores 65.63 / 65.80 / 65.70 across seeds
(sd 0.07, range 0.17). Treat any Δ below ~0.2 as seed noise, not an effect. The
"beats `full`" column counts per-seed wins, which is the honest tie-breaker.

- **Loss-weight calibration is where the free points are.** `pos-weights-auto`
  (identical graph, identical architecture, identical anchors — only the BCE
  positive weights change) gains **+4.26 @0.5, winning all 3 seeds** (+3.47 /
  +4.71 / +4.60) at zero parameter and zero wall-time cost. Its @CMA gain (+0.13,
  also 3/3) is inside the noise band, so the honest claim is: *calibration moves
  the uncalibrated score enormously and the tuned score barely at all.* See
  [Experiment C](#experiment-c-pos_weight-measured-on-this-runs-own-anchors).
- **Residual fusion is the only feature-side component that pays.** −0.29 @CMA
  and −1.41 @0.5, and it loses to same-seed `full` on 3/3 seeds — the only
  consistent loss among the feature ablations. It costs 16k parameters.
- **Text features and temporal edge decay are free to drop** (−0.09 and +0.06,
  0/3 and 2/3 wins): indistinguishable from noise.
- **A second HGT layer buys nothing real:** +0.11 @CMA (2/3 seeds) for +32%
  parameters and +65% wall time. The depth study is effectively a no-op; 1 layer
  is the right choice.
- **LightGBM is the baseline that matters:** 2.4 points behind after threshold
  tuning, but **+2.6 ahead of `full` at the default 0.5 threshold**, 18× cheaper
  and 1,860 "parameters". With doc weights the GNN's advantage exists only
  *after* per-activity threshold optimization. `pos-weights-auto` is the fix: it
  turns that −2.6 into **+1.7 @0.5** (61.54 vs 59.86) and keeps the +2.5 lead at
  @CMA, so a submission no longer has to choose between calibrated probabilities
  and tuned thresholds — it can have both.

## Experiment C: `pos_weight` measured on this run's own anchors

The static weights in `docs/VALIDATION_DATA.md` §5 were fitted on a 252-anchor
panel spanning 2025-12-02 … 2026-09-30 — days beyond the 2026-02-15 validation
week and beyond anything a single run trains on. Measured on *this* run's 8
anchors, `pos_weight[a] = (N_observed[a] − N_pos[a]) / N_pos[a]`
(`--pos-weights auto`, masked cells excluded from the denominator because an
unobserved cell never enters the loss) says the doc constants are 1.5–2.6× too
high:

| activity | doc constants | auto (this run) | doc ÷ auto |
| --- | --- | --- | --- |
| open_pr | 7.28 | 3.46 | 2.10× |
| open_issue | 18.08 | 9.05 | 2.00× |
| issue_comment | 11.05 | 4.37 | 2.53× |
| pr_review | 12.83 | 5.04 | 2.55× |
| pr_review_comment | 19.29 | 7.56 | 2.55× |
| push | 1.32 | 0.85 | 1.55× |

The mechanism the arm was designed to test shows up directly in the fitted
thresholds — CMA-ES had been spending its budget undoing the inflation:

| | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
| --- | --- | --- | --- | --- | --- | --- |
| mean θ, `full` (doc weights) | 0.762 | 0.874 | 0.827 | 0.871 | 0.892 | 0.406 |
| mean θ, `pos-weights-auto` | 0.631 | 0.814 | 0.701 | 0.726 | 0.775 | 0.315 |

Per-activity F1 at the **default 0.5 threshold** (mean of 3 seeds, %):

| activity | full | pos-weights-auto | Δ |
| --- | --- | --- | --- |
| pr_review_comment | 47.99 | 55.32 | **+7.33** |
| issue_comment | 53.51 | 59.77 | **+6.26** |
| pr_review | 57.22 | 62.28 | **+5.06** |
| open_issue | 36.35 | 41.37 | **+5.02** |
| open_pr | 56.50 | 59.77 | **+3.27** |
| push | 78.37 | 77.92 | −0.45 |

Every activity whose doc weight was ≥2× inflated gains 3–7 points; `push`, whose
weight was already closest (1.55×), is flat. So the doc constants were not merely
suboptimal, they were systematically over-predicting five of six activities and
the 0.5 decision boundary was paying for it.

**Caveats.** 3 seeds, and the @CMA delta (+0.13) sits inside the ±0.2 noise band
even though it wins 3/3 — do not claim a tuned-score improvement. These three
cells ran at `fc20c3c` while `full` ran at `5e46c58`; `python -m
experiments.ablations.audit_parity` passes 19/19 shared arguments and the only
training-adjacent change between those revisions is W&B's *local* run directory
(`wandb/` → `.wandb-runs/`), so the comparison is numerically matched. A `full`
re-run at `fc20c3c` would settle it for ~68 min.

**Takeaway for the graded submission:** use `--pos-weights auto` regardless of
what you do about thresholds — it costs nothing, and it is worth +4.3 points if
thresholds cannot be tuned on the real test week. With thresholds it is 65.84,
the best mean in the grid.

## Per-activity F1 (mean of 3 seeds, @CMA-ES, %)

| activity | full | pos-weights-auto | lightgbm | Δ lgb−full | no-residual-fusion |
| --- | --- | --- | --- | --- | --- |
| push | 78.54 | 78.65 | 78.38 | −0.16 | 78.52 |
| pr_review | 66.60 | 66.52 | 65.15 | −1.46 | 66.45 |
| issue_comment | 63.21 | 63.60 | 59.44 | −3.76 | 62.44 |
| pr_review_comment | 62.00 | 61.85 | 59.04 | −2.96 | 61.80 |
| open_pr | 60.98 | 60.87 | 54.77 | −6.20 | 60.70 |
| open_issue | 47.05 | 47.45 | 45.22 | −1.83 | 47.32 |

The graph model's whole advantage sits in the **structurally rare** activities
(`open_pr`, `issue_comment`, `pr_review_comment`). On `push` — 590k of the
979k cohort events — a flat GBM already matches it.

## Provenance and reproduction

Tracked here: `ablation_summary.csv` / `.md` (the unified 24-row summary) and,
under `outputs/<variant>_seed<seed>/`, the `result.json` for every cell — which
carries the full command line, git revision, duration, parameter count, all
per-activity F1s, the chosen thresholds, the val cutoff, the `pos_weights` and
`pos_weights_mode` actually used, and the W&B run name.
`logs/<variant>_seed<seed>.log` keeps stdout, including the memory mode and the
resolved pos weights each cell ran in.

Deliberately **not** in git (all regenerable by re-running a cell, per
`experiments/ablations/.gitignore`): `model.pt` weights (~9 MB × 21 cells =
195 MB), `val_*.npy` / `val_*.csv` caches (18 MB), `wandb/` (28 MB, the original
21 cells) and `.wandb-runs/` (3.4 MB, the calibration cells — the local W&B dir
moved there in `f62ad04` because a plain `wandb/` next to the package shadows the
installed package), `smoke/`, `logs_dry_run/`. Weights are only needed for
inference — `run_all_ablations.py` rewrites them from scratch in ~68 min per arm.

W&B: all 24 cells synced online as `abl-<variant>-s<seed>` under
`nycu-bigdata-asg2` (see the `View run at` line in each log); the calibration runs
carry the extra tags `calibration` and `pos-weights`. No key is stored in the
repo; supply `WANDB_API_KEY` yourself.

Reproduce on a 24 GiB×N box:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# one shard per GPU; --variants splits the grid, LightGBM needs no GPU
CUDA_VISIBLE_DEVICES=0 python -m experiments.ablations.run_all_ablations \
  --skip-existing --num-threads 8 --seeds 42,123,999 \
  --output-dir experiments/ablations/outputs --log-dir experiments/ablations/logs
```

Then re-run the same command **without** `--variants` to rebuild the combined
summary from the per-cell `result.json` files (`--skip-existing` reuses every
cell already marked `ok`). Note that each shard overwrites
`ablation_summary.csv` with only its own cells — the combine pass is not
optional.

Adding one arm to an existing grid is a single command, because
`--skip-existing` runs only the cells with no `ok` result yet — this is exactly
how the 3 calibration cells were produced on 2026-11-18:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python -m experiments.ablations.run_all_ablations --skip-existing --num-threads 8
```

Keep `--variants` out of it and the summary it writes is already the full 24-row
one. Verify parity after a code change with `python -m
experiments.ablations.audit_parity` (currently PASS, 19/19 shared arguments).
