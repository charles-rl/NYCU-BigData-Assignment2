```
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
```

```
mkdir raw_data
mkdir processed_tables
```

```
python scripts/get_raw_data.py
```

```
python scripts/preprocess.py --raw raw_data --out processed_tables
```

```
python -m src.data.make_split
```

```
WANDB_API_KEY="[REDACTED-WANDB-KEY]" python -m scripts.run_training --anchor-end 2026-10-18 --anchor-stride 14
```

```
winget install Graphviz.Graphviz
```

