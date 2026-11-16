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

```bash
winget install Graphviz.Graphviz
```
