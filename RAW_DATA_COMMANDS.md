```bash
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
```

```bash
mkdir raw_data
mkdir processed_tables
```

```bash
python scripts/get_raw_data.py
```

```bash
python scripts/preprocess.py --raw raw_data --out processed_tables
```

```bash
python -m src.data.make_split
```

```bash
python -m scripts.run_training --anchor-end 2026-10-18 --anchor-stride 14
```

```bash
# type the key inline, never `export` it (shared shell account) and never commit it
WANDB_API_KEY=<your own key> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.run_training --epochs 1000
```

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
