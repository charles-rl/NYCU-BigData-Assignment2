# The submission run (what to type)

```bash
# 1. download the hourly files (edit START_DATE/END_DATE in the script first:
#    the file is CRLF, so patch bytes rather than using `edit`)
python scripts/get_raw_data.py

# 2. audit them (~10 min, 48 workers). Writes raw_data/coverage.csv and
#    raw_data/coverage_activity.csv; the trainer reads the second one.
python scripts/verify_raw_data.py --workers 48

# 3. build the 8 tables (~30 min). Required: the model reads nothing else.
python scripts/preprocess.py --raw raw_data --out processed_tables

# 4. the submission run (~3 h): 18 anchors, no held-out week, thresholds fitted
#    on the last 84 days of anchors, then the test-week graph and
#    submission/submission.csv. See docs/FINAL_RUN.md.
WANDB_API_KEY=<your own key> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m scripts.run_training
```

`python -m scripts.run_training` is `--mode final` by default. Useful variants:

```bash
# check the anchor plan, weights and graph sizes without training (~3 min)
python -m scripts.run_training --dry-run

# re-predict from the saved checkpoint after refreshing raw data / tables
# (the feature window is 10 h short of the graded week; this refreshes it fast)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.run_training --predict-only

# the measurement protocol on the held-out week 2026-02-15 (what the ablation
# grid and all 60 HPO trials ran)
python -m scripts.run_training --mode validate
```

Other one-off steps:

```bash
# leakage-checked temporal split written to splits/ (features, answer.csv)
python -m src.data.make_split --cutoff 2026-02-08

# decay curves + POS_WEIGHTS from clean anchors
python -m src.data.activity_decay_analysis
```

Need the GPU to yourself: the final run peaks at 66 GiB of VRAM for the largest anchor
(25.6 GiB per million graph nodes, measured). It refuses to start if another process is
holding the card.

# For Ablations
Terminal A — GPU 0, two cells (~3.7 h):

 ```bash
   cd /train-data-5-hdd/charles/NYCU-BigData-Assignment2 && conda activate bda_asg2
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
   python -m experiments.ablations.run_all_ablations --skip-existing --num-threads 8 \
     --variants depth-2layer --seeds 42,123 \
     --output-dir experiments/ablations/outputs --log-dir experiments/ablations/logs
 ```

 ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 \
   python -m experiments.ablations.run_all_ablations --skip-existing --num-threads 8 \
     --variants depth-2layer --seeds 999 \
     --output-dir experiments/ablations/outputs --log-dir experiments/ablations/logs
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
   python -m experiments.ablations.run_all_ablations --skip-existing --num-threads 8
```

 Terminal B — GPU 1, one cell (~1.9 h): same command with CUDA_VISIBLE_DEVICES=2 --seeds 999.

Then, once both finish — rebuild the combined summary (~2 min, reuses all ok cells):

 ```bash
   python -m experiments.ablations.run_all_ablations --skip-existing --num-threads 8 \
     --output-dir experiments/ablations/outputs --log-dir experiments/ablations/logs
 ```

```bash
winget install Graphviz.Graphviz
```
