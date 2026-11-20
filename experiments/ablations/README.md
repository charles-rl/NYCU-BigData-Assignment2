# Experiment suite: ablations, tabular baseline, and depth study

Strictly isolated experiment suite for the GitHub user-activity HGT pipeline.
**No file under `src/` or `scripts/` is modified.** The ablation code is a copy
of the base pipeline kept entirely inside `experiments/`, so a smoke test or a
failed run can never change the production scripts.

```text
experiments/
├── ablations/
│   ├── build_graph_ablation.py        # copy of build_graph.py + edge weights + text mask
│   ├── hetero_model_ablation.py       # copy of hetero_model.py + decay-aware HGT + fusion switch
│   ├── train_engine.py                # copy of train.py + W&B group/tags + metric artifacts
│   ├── run_training_ablation.py       # copy of scripts/run_training.py + variant CLI + result.json
│   ├── train_full.py                  # GNN full-model control
│   ├── train_no_text.py               # mask the 6 structural/textual features
│   ├── train_no_temporal_decay.py     # all edge weights = 1.0
│   ├── train_no_residual_fusion.py    # heads see only the post-GNN user embedding
│   ├── train_pos_weights_auto.py      # Experiment C: pos_weight from this run's anchors
│   ├── train_gnn_depth.py             # Experiment B: --num_layers 1 vs 2
│   ├── run_all_ablations.py           # 9 variants × 3 seeds = 27 runs; CSV + Markdown
│   └── audit_parity.py                # hyperparameter-parity audit
└── baselines/
    ├── train_lightgbm.py              # Experiment A: 6 binary LightGBM models on Users.x
    └── outputs/                       # per-run artifacts (generated)
```

Generated artifacts are ignored: `ablation_summary.csv`, `ablation_summary.md`,
`ablation_summary_dry_run.*`, `outputs/`, `smoke/`, `logs/`, `logs_dry_run/`,
`wandb/`.

## The 27-run grid

`run_all_ablations.py` runs **9 variants × 3 seeds = 27 runs** when executed
without `--dry-run`.

| variant | module | text features | temporal edge decay | raw-user fusion | pos_weight | GNN layers | W&B group |
|---|---|---|---|---|---|---|---|
| `full` | `train_full.py` | present | ON (`exp(-dt/τ)`) | ON | doc constants | 1 | `github-activity-ablations` |
| `no-text` | `train_no_text.py` | **masked** | ON | ON | doc constants | 1 | `github-activity-ablations` |
| `no-temporal-decay` | `train_no_temporal_decay.py` | present | **OFF (all 1.0)** | ON | doc constants | 1 | `github-activity-ablations` |
| `no-residual-fusion` | `train_no_residual_fusion.py` | present | ON | **OFF** | doc constants | 1 | `github-activity-ablations` |
| `pos-weights-auto` | `train_pos_weights_auto.py` | present | ON | ON | **(N_obs−N_pos)/N_pos** | 1 | `github-activity-ablations` |
| `lightgbm` | `../baselines/train_lightgbm.py` | n/a (tabular only) | n/a | n/a | doc constants | n/a | `github-activity-baselines` |
| `lightgbm-pos-weights-auto` | `../baselines/train_lightgbm.py --pos-weights auto` | n/a (tabular only) | n/a | n/a | **(N_obs−N_pos)/N_pos** | n/a | `github-activity-baselines` |
| `depth-1layer` | `train_gnn_depth.py --num_layers 1` | present | ON | ON | doc constants | **1** | `github-activity-depth-study` |
| `depth-2layer` | `train_gnn_depth.py --num_layers 2` | present | ON | ON | doc constants | **2** | `github-activity-depth-study` |

The four `full`/`no-*` arms change exactly one model factor. The LightGBM arm
is a separate model family. The depth arms change only the layer count.
`depth-1layer` is architecture-identical to `full`; it is rerun under the
depth-study W&B group so the 1-vs-2 comparison has a matched control pair.
`pos-weights-auto` changes no architecture at all — it is the graph and model of
`full` with the loss weights measured on the run's own anchors, which isolates
calibration from every other factor (Experiment C).
`lightgbm-pos-weights-auto` is the same calibration change applied to the
tabular baseline: identical LightGBM features, hyperparameters and protocol,
with `scale_pos_weight` recomputed from the run's own anchors instead of the doc
constants. It is the matched control that shows whether the +4-point @0.5 gain
is specific to the GNN or a property of the loss weighting itself.

## Hyperparameter parity

`audit_parity.py` compares the copied GNN driver with
`scripts/run_training.py` argument-by-argument:

```bash
python -m experiments.ablations.audit_parity
```

Expected result: **PASS**, with:

* **33/33** shared GNN hyperparameters matching, including
  `lr=1e-3`, `weight_decay=3e-5`, `hidden_dim=256`, `latent_dim=128`,
  `head_hidden_dim=128`, `num_heads=4`, `num_gnn_layers=1`, `dropout=0.4`,
  `warm-restarts` (`t0=5`, `t_mult=2`), `cma_maxfevals=3000`,
  `cma_popsize=16`, `cma_sigma0=0.15`, `val_interval=5`, `grad_clip=1.0`;
* **`epochs=310`** as the one intentional suite-wide override (base default
  is 150);
* the exact measured positive-class weights
  `[7.28, 18.08, 11.05, 12.83, 19.29, 1.32]` (`--pos-weights doc`);
* **19/19** LightGBM split/threshold arguments matching the GNN protocol;
* all eight orchestrator modules and three W&B groups present.

`device`/`wandb_mode` are intentionally dynamic (dry-run → CPU/offline) and are
excluded from the parity table. Artifact-path defaults are suite-specific and
also excluded.

## Important: where the temporal decay lives

The checked-in base pipeline (`src/features/build_graph.py` +
`src/models/hetero_model.py`) does **not** currently attach temporal edge
weights: `HGTConv` receives only `edge_index_dict`. `IMPLEMENTATION_NOTES.md`
and `docs/VALIDATION_DATA.md` specify the missing decay as

```text
w_e = exp(-min(dt_days, max_days) / tau_activity)
TEMPORAL_EDGE_TAU_DAYS = {
    open_pr: 5.96, open_issue: 5.37, issue_comment: 6.02,
    pr_review: 6.73, pr_review_comment: 6.91, push: 7.50,
}
```

This suite implements that specification **inside the copied files**:

* `full` = base architecture **plus** the documented temporal edge weighting;
* `no-temporal-decay` = the same code path with every event edge weight set to
  exactly `1.0`, which is mathematically identical to stock `HGTConv`;
* static `Repositories -> Users owner` edges are unit-weighted in both arms
  because they have no event timestamp.

Consequently, `full` is comparable to `no-temporal-decay` and to the other
ablations, but it is **not** byte-identical to `scripts/run_training.py`. If the
control arm must instead be exactly the base script (no decay at all), say so
before launching the full grid.

The driver verifies this at runtime: `check_edge_weights()` fails if a decay
arm is all ~1.0 or if a no-decay arm contains any non-unit weight, and the
smoke log prints the mean/min/max weight per activity.

## The six text features

These are the six structural/textual pattern columns from
`IMPLEMENTATION_NOTES.md`; they live in three node types, not in `Users`.

| node type | column index | feature |
|---|---|---|
| `Issues` | 3 | `log_issue_len` |
| `Issues` | 4 | `has_markdown_structure` |
| `Issues` | 5 | `has_prefix` |
| `PullRequests` | 3 | `branch_depth` |
| `PRReviewComments` | 1 | `log_diff_len` |
| `PRReviewComments` | 2 | `path_depth` |

`train_no_text.py` supports:

* `--text-mode zero` (default): write zeros, keep the feature width. Parameter
  count and architecture stay identical to `full`.
* `--text-mode drop`: slice the six columns out; the per-node encoder input
  widths shrink accordingly.

## Experiment A: LightGBM tabular baseline

`experiments/baselines/train_lightgbm.py` trains **six independent binary
LightGBM classifiers**, one per activity, strictly on the 21-column user block
that the GNN sees as `data['Users'].x`:

```text
6 activities × (days_since, has_ever, count_7d)
+ is_bot + is_cohort + user_age_days = 21 features
```

No graph structure, no node embeddings, no text features.

Fairness contract:

* same temporal split, anchor selection, leakage checks, and cohort mask
  (helpers imported from the copied GNN driver);
* same calibrated `pos_weights`, passed to LightGBM as `scale_pos_weight`;
* thresholds fitted by CMA-ES on **training rows only**, then applied to the
  validation week;
* same `(macro-F1 + micro-F1) / 2` scorer and same validation CSV format;
* the feature block is verified against a real `build_hetero_graph` output on
  the validation anchor: the smoke run prints
  `max|diff|=0.000e+00 (shape (23454, 21))`.

LightGBM-specific defaults (documented and exposed as CLI flags):
`n_estimators=310`, `learning_rate=0.05`, `num_leaves=31`,
`min_child_samples=20`, `subsample=0.8` (`freq=1`), `colsample_bytree=0.8`,
`reg_lambda=1.0`, `deterministic=True`, CPU by default. The result JSON's
`n_parameters` is the total number of trees (`parameter_kind="trees"`), not a
neural parameter count.

The baseline runs on CPU even when the GNN children use `--device cuda`; the
orchestrator deliberately does not forward `--device` to it.

## Experiment B: GNN depth study

`train_gnn_depth.py` reuses the copied base driver and adds the explicit

```bash
--num_layers {1,2}
```

argument. It selects the `depth-1layer` or `depth-2layer` variant, changes only
`--num-gnn-layers`, and logs under `github-activity-depth-study`. All other
hyperparameters are the verified base defaults (epochs 310, warm-restarts,
doc pos_weights). The default is one layer, matching `run_training.py`.

## Quick start: smoke tests (CPU, 1 epoch / 1 iteration)

These are the commands used to verify the suite. They force one epoch (GNN) or
one boosting round (LightGBM), one anchor, a 100-evaluation CMA-ES budget, CPU,
and offline W&B. They do **not** start full training loops.

```bash
# from the repository root, with the bda_asg2 conda env active
export OMP_NUM_THREADS=8

# --- GNN ablations ---
python -m experiments.ablations.train_full --dry-run \
    --seed 42 --device cpu --num-threads 8 \
    --output-dir experiments/ablations/smoke

python -m experiments.ablations.train_no_text --dry-run \
    --seed 42 --device cpu --num-threads 8 \
    --output-dir experiments/ablations/smoke

python -m experiments.ablations.train_no_temporal_decay --dry-run \
    --seed 42 --device cpu --num-threads 8 \
    --output-dir experiments/ablations/smoke

python -m experiments.ablations.train_no_residual_fusion --dry-run \
    --seed 42 --device cpu --num-threads 8 \
    --output-dir experiments/ablations/smoke

# --- Experiment A: LightGBM ---
python -m experiments.baselines.train_lightgbm --dry-run \
    --seed 42 --device cpu --num-threads 8 \
    --output-dir experiments/ablations/smoke

python -m experiments.baselines.train_lightgbm --dry-run --pos-weights auto \
    --seed 42 --device cpu --num-threads 8 \
    --output-dir experiments/ablations/smoke

# --- Experiment B: depth study ---
python -m experiments.ablations.train_gnn_depth --dry-run \
    --seed 42 --device cpu --num_layers 1 --num-threads 8 \
    --output-dir experiments/ablations/smoke

python -m experiments.ablations.train_gnn_depth --dry-run \
    --seed 42 --device cpu --num_layers 2 --num-threads 8 \
    --output-dir experiments/ablations/smoke
```

All nine write `experiments/ablations/smoke/<variant>_seed42/result.json` and
exit `0` on success. The result JSON is the source of truth for the summary.

Aggregate the smoke results without retraining:

```bash
python -m experiments.ablations.run_all_ablations --dry-run --device cpu \
    --num-threads 8 --skip-existing
```

This writes `ablation_summary_dry_run.csv` / `.md` (the dry-run names are
deliberate: a smoke test cannot overwrite the real summary).

## Full grid: 21 runs (only when the GPU is free)

```bash
# Suite defaults: epochs=310, lr schedule=warm-restarts, doc pos_weights.
python -m experiments.ablations.run_all_ablations --device cuda

# Optional HPO-comparable schedule:
python -m experiments.ablations.run_all_ablations --device cuda \
    --lr-schedule cosine

# Reuse completed cells / stop at the first failure:
python -m experiments.ablations.run_all_ablations --skip-existing
python -m experiments.ablations.run_all_ablations --fail-fast
```

The orchestrator runs cells sequentially so 21 runs cannot fight over GPU
memory. `--skip-existing` reuses any completed `result.json`. The LightGBM
child always runs on CPU.

Runs write only under `experiments/`:

```text
ablations/outputs/<variant>_seed<seed>/       # GNN + depth
    result.json, val_pred.csv, val_truth.csv,
    val_probs.npy, val_labels.npy, model.pt
baselines/outputs/lightgbm_seed<seed>/         # when run directly
    result.json, val_pred.csv, val_truth.csv,
    val_probs.npy, val_labels.npy, feature_importance.csv
ablations/logs/<variant>_seed<seed>.log
```

The base pipeline's `splits/`, `checkpoints/`, and `logs/` are never touched.

## W&B logging

| experiment | group |
|---|---|
| GNN full model + ablations | `github-activity-ablations` |
| LightGBM baseline | `github-activity-baselines` |
| GNN depth study | `github-activity-depth-study` |

Tags follow `<variant-tag>`, `seed-<seed>`, and `ablation` (plus `dry-run` in
smoke mode, plus any `--wandb-tags`). Run names are `abl-<variant>-s<seed>` for
the GNN arms, `lightgbm-s<seed>` for the doc-weight baseline, and
`lightgbm-pos-auto-s<seed>` for the tabular calibration arm.

`wandb.summary` records both metric variants at the best epoch/iteration:

```text
best_val/score_cmaes
best_val/macro_f1_cmaes
best_val/micro_f1_cmaes
best_val/score_default_0.5
best_val/macro_f1_default_0.5
best_val/micro_f1_default_0.5
best_epoch / n_estimators
best_thresholds
official_eval_score   (LightGBM)
```

## The summary table

`run_all_ablations.py` writes:

* `ablation_summary.csv` — one row per (variant, seed) with:
  `macro_f1`, `micro_f1`, `score_default_0_5`, `score_cmaes`, `status`,
  `returncode`, `duration_s`, `epochs`, `device`, `val_cutoff`,
  `n_train_anchors`, `n_parameters`, `wandb_group`, `wandb_tags`,
  `result_json`, `log_path`.
  **CSV values are fractions in `[0, 1]`.**
* `ablation_summary.md` — per-run table in **percent**, mean ± std over seeds
  per variant, and Δ vs the Full Model in percentage points.

`Score @0.5` and `Score @CMA-ES` are computed from the exact same
`val_probs.npy` / `val_labels.npy` arrays, so they differ only in the decision
thresholds. The GNN driver also checks that its re-scored CMA-ES value matches
the trainer's reported best score.

## Metric definitions

For each activity, `F1 = 2·tp / (2·tp + fp + fn)`. Then:

* Macro-F1 = mean of the six F1s;
* Micro-F1 = one pooled F1 over all six activities;
* Score = `(Macro-F1 + Micro-F1) / 2`.

This is the TA scorer (`src/utils/metrics.py`), used in memory and re-checked
against `scripts/evaluate.py` by both the GNN trainer and the LightGBM
baseline.

## Notes on comparability

* `--seed` controls Python/NumPy/PyTorch seeding and the anchor shuffle order;
  each run is a separate process, so the three seeds are independent.
* Oracle thresholds are fitted on **training rows only** and merely applied to
  validation rows.
* The base driver defaults are `150` epochs and `warm-restarts`. This suite
  deliberately sets **`epochs=310`** everywhere; all other GNN hyperparameters
  match the base exactly (see `audit_parity.py`). Warm-restarts remains the
  default; pass `--lr-schedule cosine` only if you want the HPO study's
  schedule.
* `--decay-max-days 30` clips `dt` to the same 30-day horizon used by the user
  recency features. Set it to `0` to disable clipping.
* On a shared server use `--num-threads 8` (and/or `OMP_NUM_THREADS=8`); the
  default lets PyTorch/LightGBM take every core.
* The LightGBM baseline's tree hyperparameters are model-specific and cannot
  be parity-matched to the GNN's optimizer/scheduler; the audit verifies that
  every split, label, calibration, threshold, and metric setting is shared.
