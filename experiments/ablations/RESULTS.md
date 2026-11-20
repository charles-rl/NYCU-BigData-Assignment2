# Ablation + baseline results

27 cells: 7 GNN variants × 3 seeds (42 / 123 / 999) plus 2 LightGBM arms ×
3 seeds (`lightgbm` with the doc constants and `lightgbm-pos-weights-auto` with
weights measured on the run's own anchors), all on the documented validation
split — cutoff **2026-02-15**, 7-day label week, 8 training anchors
(2025-12-22 … 2026-02-08), 6,736 test users, 5,873 validation users.

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
| repo revision | `5e46c58` (the 21 original cells) / `fc20c3c` (the 3 GNN calibration cells) / the 6 LightGBM cells re-run together in the commit that adds `lightgbm-pos-weights-auto` |
| clock | first cell finished 2026-11-16 20:03 UTC, GNN grid last 2026-11-18 20:36 UTC; the 3 LightGBM calibration cells ran 2026-11-21 01:21–01:25 UTC |
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
   re-run after the OOM above. The three GNN calibration cells were added later
   and ran as a single shard on one card (`--skip-existing`, 3 × ~67 min =
   3.4 h). The three LightGBM calibration cells were also added later; they are
   CPU-only, so they ran as 3 parallel shards (one per seed, 8 threads each) and
   finished in 3.8 min each (~4 min wall time for all three).
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
| lightgbm-pos-weights-auto | 60.12 | 66.31 | **61.78** | 63.22 | **−2.49** | **+4.50** | 3.8 | 1,860 | 63.10–63.43 | 0/3 |

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
- **The calibration fix ports to LightGBM, but only ~45% as strongly.**
  `lightgbm-pos-weights-auto` lifts the baseline from 59.86 to **61.78 @0.5
  (+1.92, 3/3 seeds)** while leaving @CMA untouched (63.33 → 63.22, −0.12). So the
  gain is not a GNN artifact — the same inflated `scale_pos_weight` was
  mis-centering the GBM's probabilities too — but the GNN collected 4.26 points
  from the same change versus 1.92 here. The residual GNN advantage is therefore
  mostly in the *tuned* score (+2.62 @CMA over `lightgbm-pos-weights-auto`),
  which is the part calibration cannot buy.
- **At the default threshold the tabular baseline now wins outright:** 61.78
  (`lightgbm-pos-weights-auto`) vs 61.54 (`pos-weights-auto`) and 57.28 (`full`).
  Only after CMA-ES tuning does the graph model pull ahead (65.84 vs 63.22), and
  that is exactly the regime where the 8-anchor, 0.5-threshold submission cannot
  operate.

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

## Experiment C on the tabular baseline: `lightgbm-pos-weights-auto`

`run_all_ablations.py` now carries a ninth arm that applies the *identical*
recalibration to LightGBM: same 21-feature `Users.x` block, same tree
hyperparameters, same split/threshold protocol, only `scale_pos_weight` swapped
from the doc constants to `(N_observed − N_pos) / N_pos` measured on the run's
own 8 anchors. The measured weights are the same six numbers the GNN arm used
(3.46 / 9.05 / 4.37 / 5.04 / 7.56 / 0.85), since both read the same anchors.

| | lightgbm (doc) | lightgbm-pos-weights-auto | Δ |
| --- | --- | --- | --- |
| Macro-F1 | 60.33 | 60.12 | −0.21 |
| Micro-F1 | 66.33 | 66.31 | −0.02 |
| Score @0.5 | 59.86 | **61.78** | **+1.92** |
| Score @CMA-ES | 63.33 | 63.22 | −0.12 |

Per-seed @0.5: 61.89 / 61.78 / 61.68 vs 59.88 / 59.83 / 59.86 — the auto arm
wins all three seeds by +2.01 / +1.95 / +1.82, so the effect is far larger than
the seed spread of either arm (sd 0.10 vs 0.03); the @CMA per-seed deltas are
−0.02 / −0.01 / −0.32, i.e. noise to slightly negative.

Per-activity F1 at the **default 0.5 threshold** (mean of 3 seeds, %):

| activity | lightgbm | lightgbm-auto | Δ |
| --- | --- | --- | --- |
| issue_comment | 55.14 | 59.45 | **+4.32** |
| pr_review_comment | 52.99 | 56.38 | **+3.39** |
| pr_review | 60.64 | 63.49 | **+2.85** |
| open_issue | 40.19 | 42.76 | **+2.58** |
| open_pr | 57.67 | 57.63 | −0.04 |
| push | 78.42 | 77.66 | −0.76 |

The same five activities that benefited in the GNN benefit here (and `push`,
the least-inflated weight, degrades slightly), so the mechanism is the loss
weighting itself, not anything graph-specific. The magnitude is the finding:
**+1.92 @0.5 for the GBM vs +4.26 for the GNN from the same recalibration.**
The GNN's doc weights over-shoot further because its BCE operates on
message-passed logits that are already sharper; the GBM's flatter scores absorb
less of the inflation. Mean fitted thresholds move the same direction:

| | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
| --- | --- | --- | --- | --- | --- | --- |
| Ω, lightgbm (doc) | 0.784 | 0.851 | 0.840 | 0.861 | 0.885 | 0.493 |
| Ω, lightgbm-auto | 0.653 | 0.752 | 0.696 | 0.727 | 0.777 | 0.378 |

**Caveats.** Same 3-seed caveat as above. The doc-weight `lightgbm` cells were
re-run alongside these at the same revision, because the two LightGBM arms had
been sharing one artifact directory (`outputs/lightgbm_seed<seed>/`) and the
calibration run was overwriting the doc arm's cached predictions; the driver now
derives its artifact dir from `--results-json`, so each arm writes only its own
directory. The re-run reproduced the original metrics exactly (59.88 / 59.83 /
59.86 @0.5), so the grid is unchanged — the cells now just point at correct
`val_probs.npy` files. Note that the GNN calibration cells ran at `fc20c3c`; the
only training-path change since then is that directory derivation, which is
placement-only.

**Takeaway:** calibration is a *general* fix, but it does not make the tabular
baseline competitive at tuned thresholds. `lightgbm-pos-weights-auto` wins the
uncalibrated-threshold comparison (61.78, ahead of every GNN arm with doc
weights) and still loses the tuned one by 2.6 points (63.22 vs 65.84), which is
where the graph structure actually pays.

## Per-activity F1 (mean of 3 seeds, @CMA-ES, %)

| activity | full | pos-weights-auto | lightgbm | lightgbm-auto | Δ lgb−full | no-residual-fusion |
| --- | --- | --- | --- | --- | --- | --- |
| push | 78.54 | 78.65 | 78.38 | 78.51 | −0.16 | 78.52 |
| pr_review | 66.60 | 66.52 | 65.15 | 65.08 | −1.46 | 66.45 |
| issue_comment | 63.21 | 63.60 | 59.44 | 59.70 | −3.76 | 62.44 |
| pr_review_comment | 62.00 | 61.85 | 59.04 | 58.65 | −2.96 | 61.80 |
| open_pr | 60.98 | 60.87 | 54.77 | 53.78 | −6.20 | 60.70 |
| open_issue | 47.05 | 47.45 | 45.22 | 45.00 | −1.83 | 47.32 |

The graph model's whole advantage sits in the **structurally rare** activities
(`open_pr`, `issue_comment`, `pr_review_comment`). On `push` — 590k of the
979k cohort events — a flat GBM already matches it.

## Provenance and reproduction

Tracked here: `ablation_summary.csv` / `.md` (the unified 27-row summary) and,
under `outputs/<variant>_seed<seed>/`, the `result.json` for every cell — which
carries the full command line, git revision, duration, parameter count, all
per-activity F1s, the chosen thresholds, the val cutoff, the `pos_weights` and
(`pos_weights_mode` for the calibration arms) actually used, and the W&B run
name.
`logs/<variant>_seed<seed>.log` keeps stdout, including the memory mode and the
resolved pos weights each cell ran in.

Deliberately **not** in git (all regenerable by re-running a cell, per
`experiments/ablations/.gitignore`): `model.pt` weights (~9 MB × 21 GNN cells =
195 MB), `val_*.npy` / `val_*.csv` caches (~24 MB), `wandb/` (28 MB, the original
21 cells) and `.wandb-runs/` (the calibration cells — the local W&B dir
moved there in `f62ad04` because a plain `wandb/` next to the package shadows the
installed package), `smoke/`, `logs_dry_run/`. Weights are only needed for
inference — `run_all_ablations.py` rewrites them from scratch in ~68 min per arm.

W&B: all 27 cells synced online as `abl-<variant>-s<seed>` under
`nycu-bigdata-asg2` (see the `View run at` line in each log); the calibration runs
carry the extra tags `calibration` and `pos-weights`. The two LightGBM arms share
the `github-activity-baselines` group; run names are `lightgbm-s<seed>` for the
doc arm and `lightgbm-pos-auto-s<seed>` for the calibration arm. No key is stored
in the repo; supply `WANDB_API_KEY` yourself.

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
how the 3 GNN calibration cells were produced on 2026-11-18 and the 3 LightGBM
calibration cells on 2026-11-21:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python -m experiments.ablations.run_all_ablations --skip-existing --num-threads 8

# The LightGBM arm is CPU-only, so 3 seed-shards can run in parallel.
for S in 42 123 999; do
  python -m experiments.ablations.run_all_ablations --skip-existing \
    --variants lightgbm-pos-weights-auto --seeds $S --num-threads 8 \
    --summary-csv /tmp/shard_$S.csv --summary-md /tmp/shard_$S.md &
done; wait
```

Keep `--variants` out of it and the summary it writes is already the full 27-row
one. Verify parity after a code change with `python -m
experiments.ablations.audit_parity` (currently PASS, 19/19 shared arguments).
