# Ablation experiment summary

- Generated (UTC): `2026-11-18T10:17:17.411368+00:00`
- Git revision: `5e46c58`
- W&B groups: ablations=`github-activity-ablations`, baselines=`github-activity-baselines`, depth=`github-activity-depth-study`
- Mode: **full**
- Variants: `full`, `no-text`, `no-temporal-decay`, `no-residual-fusion`, `lightgbm`, `depth-1layer`, `depth-2layer`
- Seeds: 42, 123, 999
- Epochs: 310
- Device: cpu
- Runs: 21/21 successful

Scores are percentages in this table; the CSV stores fractions.

## Per-run results

| Variant            | Seed | Macro-F1 (%) | Micro-F1 (%) | Score @0.5 (%) | Score @CMA-ES (%) | Status |
| ------------------ | ---- | ------------ | ------------ | -------------- | ----------------- | ------ |
| Full Model         | 42   | 62.96        | 68.31        | 57.20          | 65.63             | ok     |
| Full Model         | 123  | 63.20        | 68.41        | 57.52          | 65.80             | ok     |
| Full Model         | 999  | 63.03        | 68.38        | 57.11          | 65.70             | ok     |
| No text features   | 42   | 62.87        | 68.31        | 57.11          | 65.59             | ok     |
| No text features   | 123  | 62.90        | 68.27        | 57.67          | 65.58             | ok     |
| No text features   | 999  | 63.02        | 68.40        | 57.07          | 65.71             | ok     |
| No temporal decay  | 42   | 63.08        | 68.47        | 57.02          | 65.78             | ok     |
| No temporal decay  | 123  | 63.27        | 68.49        | 57.00          | 65.88             | ok     |
| No temporal decay  | 999  | 62.87        | 68.46        | 57.12          | 65.66             | ok     |
| No residual fusion | 42   | 62.89        | 67.92        | 55.30          | 65.41             | ok     |
| No residual fusion | 123  | 62.77        | 67.88        | 56.45          | 65.32             | ok     |
| No residual fusion | 999  | 62.95        | 68.10        | 55.85          | 65.53             | ok     |
| LightGBM tabular   | 42   | 60.41        | 66.49        | 59.88          | 63.45             | ok     |
| LightGBM tabular   | 123  | 60.09        | 66.14        | 59.83          | 63.11             | ok     |
| LightGBM tabular   | 999  | 60.50        | 66.36        | 59.86          | 63.43             | ok     |
| GNN depth 1        | 42   | 62.88        | 68.33        | 57.19          | 65.61             | ok     |
| GNN depth 1        | 123  | 62.98        | 68.41        | 56.70          | 65.70             | ok     |
| GNN depth 1        | 999  | 63.10        | 68.36        | 57.10          | 65.73             | ok     |
| GNN depth 2        | 42   | 63.34        | 68.47        | 57.25          | 65.90             | ok     |
| GNN depth 2        | 123  | 63.02        | 68.36        | 55.92          | 65.69             | ok     |
| GNN depth 2        | 999  | 63.35        | 68.37        | 57.71          | 65.86             | ok     |

## Mean over seeds

| Variant            | Runs | Macro-F1 (%) | Micro-F1 (%) | Score @0.5 (%) | Score @CMA-ES (%) |
| ------------------ | ---- | ------------ | ------------ | -------------- | ----------------- |
| Full Model         | 3    | 63.06 ± 0.12 | 68.36 ± 0.05 | 57.28 ± 0.22   | 65.71 ± 0.09      |
| No text features   | 3    | 62.93 ± 0.08 | 68.33 ± 0.07 | 57.28 ± 0.33   | 65.63 ± 0.07      |
| No temporal decay  | 3    | 63.07 ± 0.20 | 68.48 ± 0.02 | 57.05 ± 0.06   | 65.77 ± 0.11      |
| No residual fusion | 3    | 62.87 ± 0.10 | 67.97 ± 0.12 | 55.87 ± 0.58   | 65.42 ± 0.10      |
| LightGBM tabular   | 3    | 60.33 ± 0.22 | 66.33 ± 0.18 | 59.86 ± 0.03   | 63.33 ± 0.19      |
| GNN depth 1        | 3    | 62.99 ± 0.11 | 68.37 ± 0.04 | 57.00 ± 0.26   | 65.68 ± 0.06      |
| GNN depth 2        | 3    | 63.24 ± 0.19 | 68.40 ± 0.06 | 56.96 ± 0.93   | 65.82 ± 0.11      |

## Effect vs Full Model

| Variant            | Δ Macro-F1 | Δ Micro-F1 | Δ Score @0.5 | Δ Score @CMA-ES |
| ------------------ | ---------- | ---------- | ------------ | --------------- |
| Full Model         | +0.00      | +0.00      | +0.00        | +0.00           |
| No text features   | -0.13      | -0.04      | +0.00        | -0.09           |
| No temporal decay  | +0.01      | +0.11      | -0.23        | +0.06           |
| No residual fusion | -0.19      | -0.40      | -1.41        | -0.29           |
| LightGBM tabular   | -2.73      | -2.03      | +2.58        | -2.38           |
| GNN depth 1        | -0.08      | +0.00      | -0.28        | -0.04           |
| GNN depth 2        | +0.18      | +0.04      | -0.32        | +0.11           |

Deltas are percentage points; positive means better than Full Model.
