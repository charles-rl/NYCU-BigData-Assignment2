# Ablation experiment summary

- Generated (UTC): `2026-11-20T17:43:00.884328+00:00`
- Git revision: `24f814f`
- W&B groups: ablations=`github-activity-ablations`, baselines=`github-activity-baselines`, depth=`github-activity-depth-study`
- Mode: **full**
- Variants: `full`, `no-text`, `no-temporal-decay`, `no-residual-fusion`, `pos-weights-auto`, `lightgbm`, `lightgbm-pos-weights-auto`, `depth-1layer`, `depth-2layer`
- Seeds: 42, 123, 999
- Epochs: 310
- Device: auto (GNN cuda/cpu; LightGBM cpu)
- Runs: 27/27 successful

Scores are percentages in this table; the CSV stores fractions.

## Per-run results

| Variant                   | Seed | Macro-F1 (%) | Micro-F1 (%) | Score @0.5 (%) | Score @CMA-ES (%) | Status |
| ------------------------- | ---- | ------------ | ------------ | -------------- | ----------------- | ------ |
| Full Model                | 42   | 62.96        | 68.31        | 57.20          | 65.63             | ok     |
| Full Model                | 123  | 63.20        | 68.41        | 57.52          | 65.80             | ok     |
| Full Model                | 999  | 63.03        | 68.38        | 57.11          | 65.70             | ok     |
| No text features          | 42   | 62.87        | 68.31        | 57.11          | 65.59             | ok     |
| No text features          | 123  | 62.90        | 68.27        | 57.67          | 65.58             | ok     |
| No text features          | 999  | 63.02        | 68.40        | 57.07          | 65.71             | ok     |
| No temporal decay         | 42   | 63.08        | 68.47        | 57.02          | 65.78             | ok     |
| No temporal decay         | 123  | 63.27        | 68.49        | 57.00          | 65.88             | ok     |
| No temporal decay         | 999  | 62.87        | 68.46        | 57.12          | 65.66             | ok     |
| No residual fusion        | 42   | 62.89        | 67.92        | 55.30          | 65.41             | ok     |
| No residual fusion        | 123  | 62.77        | 67.88        | 56.45          | 65.32             | ok     |
| No residual fusion        | 999  | 62.95        | 68.10        | 55.85          | 65.53             | ok     |
| Auto pos-weights          | 42   | 63.12        | 68.46        | 60.67          | 65.79             | ok     |
| Auto pos-weights          | 123  | 63.10        | 68.52        | 62.24          | 65.81             | ok     |
| Auto pos-weights          | 999  | 63.25        | 68.56        | 61.70          | 65.91             | ok     |
| LightGBM tabular          | 42   | 60.41        | 66.49        | 59.88          | 63.45             | ok     |
| LightGBM tabular          | 123  | 60.09        | 66.14        | 59.83          | 63.11             | ok     |
| LightGBM tabular          | 999  | 60.50        | 66.36        | 59.86          | 63.43             | ok     |
| LightGBM auto pos-weights | 42   | 60.35        | 66.51        | 61.89          | 63.43             | ok     |
| LightGBM auto pos-weights | 123  | 60.07        | 66.14        | 61.78          | 63.10             | ok     |
| LightGBM auto pos-weights | 999  | 59.94        | 66.28        | 61.68          | 63.11             | ok     |
| GNN depth 1               | 42   | 62.88        | 68.33        | 57.19          | 65.61             | ok     |
| GNN depth 1               | 123  | 62.98        | 68.41        | 56.70          | 65.70             | ok     |
| GNN depth 1               | 999  | 63.10        | 68.36        | 57.10          | 65.73             | ok     |
| GNN depth 2               | 42   | 63.34        | 68.47        | 57.25          | 65.90             | ok     |
| GNN depth 2               | 123  | 63.02        | 68.36        | 55.92          | 65.69             | ok     |
| GNN depth 2               | 999  | 63.35        | 68.37        | 57.71          | 65.86             | ok     |

## Mean over seeds

| Variant                   | Runs | Macro-F1 (%) | Micro-F1 (%) | Score @0.5 (%) | Score @CMA-ES (%) |
| ------------------------- | ---- | ------------ | ------------ | -------------- | ----------------- |
| Full Model                | 3    | 63.06 ± 0.12 | 68.36 ± 0.05 | 57.28 ± 0.22   | 65.71 ± 0.09      |
| No text features          | 3    | 62.93 ± 0.08 | 68.33 ± 0.07 | 57.28 ± 0.33   | 65.63 ± 0.07      |
| No temporal decay         | 3    | 63.07 ± 0.20 | 68.48 ± 0.02 | 57.05 ± 0.06   | 65.77 ± 0.11      |
| No residual fusion        | 3    | 62.87 ± 0.10 | 67.97 ± 0.12 | 55.87 ± 0.58   | 65.42 ± 0.10      |
| Auto pos-weights          | 3    | 63.16 ± 0.08 | 68.51 ± 0.05 | 61.54 ± 0.80   | 65.84 ± 0.06      |
| LightGBM tabular          | 3    | 60.33 ± 0.22 | 66.33 ± 0.18 | 59.86 ± 0.03   | 63.33 ± 0.19      |
| LightGBM auto pos-weights | 3    | 60.12 ± 0.21 | 66.31 ± 0.19 | 61.78 ± 0.10   | 63.22 ± 0.19      |
| GNN depth 1               | 3    | 62.99 ± 0.11 | 68.37 ± 0.04 | 57.00 ± 0.26   | 65.68 ± 0.06      |
| GNN depth 2               | 3    | 63.24 ± 0.19 | 68.40 ± 0.06 | 56.96 ± 0.93   | 65.82 ± 0.11      |

## Effect vs Full Model

| Variant                   | Δ Macro-F1 | Δ Micro-F1 | Δ Score @0.5 | Δ Score @CMA-ES |
| ------------------------- | ---------- | ---------- | ------------ | --------------- |
| Full Model                | +0.00      | +0.00      | +0.00        | +0.00           |
| No text features          | -0.13      | -0.04      | +0.00        | -0.09           |
| No temporal decay         | +0.01      | +0.11      | -0.23        | +0.06           |
| No residual fusion        | -0.19      | -0.40      | -1.41        | -0.29           |
| Auto pos-weights          | +0.10      | +0.15      | +4.26        | +0.12           |
| LightGBM tabular          | -2.73      | -2.03      | +2.58        | -2.38           |
| LightGBM auto pos-weights | -2.94      | -2.05      | +4.51        | -2.50           |
| GNN depth 1               | -0.08      | +0.00      | -0.28        | -0.04           |
| GNN depth 2               | +0.18      | +0.04      | -0.32        | +0.11           |

Deltas are percentage points; positive means better than Full Model.

## Effect vs LightGBM baseline

| Variant                   | Δ Macro-F1 | Δ Micro-F1 | Δ Score @0.5 | Δ Score @CMA-ES |
| ------------------------- | ---------- | ---------- | ------------ | --------------- |
| Full Model                | +2.73      | +2.03      | -2.58        | +2.38           |
| No text features          | +2.59      | +2.00      | -2.58        | +2.30           |
| No temporal decay         | +2.74      | +2.15      | -2.81        | +2.44           |
| No residual fusion        | +2.54      | +1.64      | -3.99        | +2.09           |
| Auto pos-weights          | +2.82      | +2.18      | +1.68        | +2.50           |
| LightGBM tabular          | +0.00      | +0.00      | +0.00        | +0.00           |
| LightGBM auto pos-weights | -0.21      | -0.02      | +1.92        | -0.12           |
| GNN depth 1               | +2.65      | +2.04      | -2.86        | +2.34           |
| GNN depth 2               | +2.90      | +2.07      | -2.90        | +2.49           |

Deltas are percentage points; positive means better than LightGBM. Both arms share the split, label window and CMA-ES protocol, so a positive Δ@CMA-ES with a negative Δ@0.5 means the arm needs threshold calibration more than the baseline does.

## Calibration arm: `pos-weights-auto` vs both references

| Arm              | Score @0.5 (%) | Score @CMA-ES (%) | Δ@0.5 vs full | Δ@CMA vs full | Δ@0.5 vs lgbm | Δ@CMA vs lgbm |
| ---------------- | -------------- | ----------------- | ------------- | ------------- | ------------- | ------------- |
| Full Model       | 57.28          | 65.71             | +0.00         | +0.00         | -2.58         | +2.38         |
| Auto pos-weights | 61.54          | 65.84             | +4.26         | +0.12         | +1.68         | +2.50         |
| LightGBM tabular | 59.86          | 63.33             | +2.58         | -2.38         | +0.00         | +0.00         |

`full` uses the static 252-anchor constants from `docs/VALIDATION_DATA.md` §5; `pos-weights-auto` uses `(N_observed - N_pos) / N_pos` measured on its own 8 training anchors. Everything else is identical, so a difference at a fixed threshold is the calibration effect, and a difference that survives CMA-ES tuning is the part thresholds cannot fix.

## Calibration arm: `lightgbm-pos-weights-auto` vs both references

| Arm                       | Score @0.5 (%) | Score @CMA-ES (%) | Δ@0.5 vs full | Δ@CMA vs full | Δ@0.5 vs lgbm | Δ@CMA vs lgbm |
| ------------------------- | -------------- | ----------------- | ------------- | ------------- | ------------- | ------------- |
| Full Model                | 57.28          | 65.71             | +0.00         | +0.00         | -2.58         | +2.38         |
| LightGBM auto pos-weights | 61.78          | 63.22             | +4.51         | -2.50         | +1.92         | -0.12         |
| LightGBM tabular          | 59.86          | 63.33             | +2.58         | -2.38         | +0.00         | +0.00         |

The same `(N_observed - N_pos) / N_pos` recalibration applied to the tabular baseline's `scale_pos_weight`. Its `Δ vs lgbm` is the matched contrast (identical model, only the weights change); its `Δ vs full` says how much of the GNN's calibration gap the baseline closes.
