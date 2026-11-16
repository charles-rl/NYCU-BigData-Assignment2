# Environment and machine-migration notes

Audience: the agent (or human) picking this repo up on the **new machine**.
Everything here was measured on the old box; anything not marked *assumed* was
read off the working tree or the live environment.

Old box: 1× **RTX PRO 6000 Blackwell Server Edition** (96 GiB, sm_120),
driver 615.71.09, CUDA 12.8 wheels.
New box: **4× RTX 3090** (24 GiB each, sm_86, Ampere), CUDA 12.x driver.

---

## 1. What actually changes

| | old box | new box | consequence |
| --- | --- | --- | --- |
| compute capability | sm_120 (Blackwell) | **sm_86** (Ampere) | any modern cu11x/cu12x torch wheel already contains sm_86 kernels — no special build needed |
| VRAM per GPU | 96 GiB | **24 GiB** | the real risk: see §5 |
| GPU count | 1 | **4** | the code is single-device, so 4 GPUs means 4 processes, not DDP (§6) |
| CUDA runtime / driver | cu128 / 615.71.09 | depends on the box | keep the pinned cu128 wheels **only if** the driver is R570+; otherwise drop to a wheel the driver supports (§3) |
| NVLink | n/a | none unless bridged | irrelevant: no multi-GPU collectives in this repo |

Nothing in `src/` or `scripts/` is Blackwell-specific. There is no `torch.compile`,
no Triton kernel, no AMP/`bfloat16` autocast, and no flash-attention dependency —
all paths are plain fp32 PyTorch + PyG. The migration is a **packaging and VRAM**
problem, not a port.

## 2. Python and the library list

Python **3.11** (`pyrightconfig.json` pins 3.11; the old env is 3.11.18). Exact
versions from the old env that the tests pass against:

```text
# python 3.11
torch==2.11.0+cu128          # swap the +cuXXX suffix for the new driver, §3
torch-geometric==2.8.1
numpy==2.4.6
pandas==3.1.1
pyarrow==26.0.1
scipy==1.17.1
scikit-learn==1.9.1
lightgbm==4.7.0              # CPU only; `--device gpu` needs a source build
optuna==5.0.0
cma==4.5.0                   # CMA-ES threshold fitting (NOT the `cmaes` package)
wandb==0.31.0
matplotlib==3.11.2
networkx==3.6.1
graphviz==0.21               # python binding; needs the system `dot` binary
tqdm==4.70.1
pytest==9.2.0
```

`torchvision==0.26.0+cu128` and `pillow==13.0.0` are present in the env but
nothing in this repo imports them; they come along with the torch install.

Setup:

```bash
conda create -n bda_asg2 python=3.11 -y && conda activate bda_asg2

# torch wheel must match the NEW box's driver, not the old box's:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install torch-geometric==2.8.1 numpy==2.4.6 pandas==3.1.1 pyarrow==26.0.1 \
            scipy==1.17.1 scikit-learn==1.9.1 lightgbm==4.7.0 optuna==5.0.0 \
            cma==4.5.0 wandb==0.31.0 matplotlib==3.11.2 networkx==3.6.1 \
            graphviz==0.21 tqdm==4.70.1 pytest==9.2.0

# system packages: graphviz is used by scripts/generate_schema_graph.py
sudo apt-get install -y graphviz
```

PyG 2.8.1 is pure Python here: `HGTConv` is imported from `torch_geometric.nn`
and no compiled extension (`torch-scatter` / `torch-sparse` / `pyg-lib`) is
imported anywhere, so do not install those wheels.

## 3. The torch wheel ↔ driver rule

The only hard constraint is that the wheel's CUDA runtime must be supported by
the driver, and sm_86 is inside every cu11x/cu12x build:

| driver | pick |
| --- | --- |
| ≥ R570 | `+cu128` (what the old box used) |
| R550–R569 | `+cu124` |
| < R550 | `+cu121`, or install a driver that reaches at least R550 |

Check on the new box before installing anything:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, \
torch.cuda.get_device_capability(0))"   # expect (8, 6) on a 3090
```

`--device` in `scripts/run_training.py` defaults to `cuda` iff
`torch.cuda.is_available()`, so a broken wheel silently lands on CPU instead of
failing loudly; the capability check above is the cheap smoke test.

## 4. What is *not* in git and must be moved separately

Everything below is gitignored and regenerable, but regenerating the first two
costs hours of bandwidth/compute. `rsync` them if the boxes are connected.

| path | size (old box) | how to rebuild |
| --- | --- | --- |
| `raw_data/` | **214 GB**, 8,042 hourly `.json.gz` files | `python scripts/get_raw_data.py` (Dec 2025 → Oct 2026; ~8k files over HTTP) |
| `processed_tables/db/` | 300 MB, 2.58 M rows across 8 parquet tables | `python scripts/preprocess.py --raw raw_data --out processed_tables` |
| `splits/` | 311 MB of per-anchor feature parquet | `python -m src.data.make_split --cutoff 2026-02-15 --days 7` — or let `run_training.py` rebuild them |
| `hpo/` | 571 MB (`hgt-hpo/trial-*/`, `study.db`, `trials.csv`) | **not rebuildable cheaply**: 60 trials ≈ 58 min each. Copy it if the 60-trial history still matters |
| `checkpoints/` | 8.9 MB | re-train |
| `.wandb-runs/`, `experiments/*/wandb/` | small | none — the old runs are deliberately deleted |

`hpo/hgt-hpo/trials.csv` is the only trustworthy score record in the project
(see the contamination warning at the top of `docs/HPO.md`). If it is not
copied, every score in `docs/HPO.md` becomes unreproducible history.

The 214 GB download is the migration bottleneck. Doing it on the new box is
fine — `get_raw_data.py` already fetches in parallel and skips existing files
by size — but budget the disk: `raw_data` + `processed_tables` + `splits` ≈
215 GB.

## 5. The real risk: 96 GiB → 24 GiB per GPU

Measured on the old box (`docs/HPO.md`): **one HPO trial reserved ~22–41 GiB**,
with two concurrent workers at ~45–80 GiB. A 3090 has 24 GiB, so:

- the lean end of the HPO space still fits, the fat end will OOM;
- on 24 GiB run **exactly one training process per GPU**, never two.

Why it is that big: the graph has 2.58 M table rows, of which 1.68 M are pushes,
so an anchor graph carries roughly 1.7 M+ edges with `edge_attr`; HGT keeps
per-edge and per-head message tensors resident for backward. Per-anchor label
matrices are tiny — the graph, not the labels, is the cost.

Mitigations, cheapest first:

1. Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the trainers
   and `scripts/hpo.py` already default it; the manual CLI runs in
   `RAW_DATA_COMMANDS.md` set it by hand).
2. One process per GPU and `--n-jobs 4` (not 8) for HPO.
3. If a config OOMs, shrink the model before shrinking the data:
   `--num-gnn-layers 1`, `--hidden-dim 256`, `--latent-dim 128` (these are
   already the defaults). `--max-anchors` reduces the number of anchors, which
   cuts CPU RAM and wall-clock, **not** peak VRAM per anchor.
4. Only then constrain the graph itself (shorter history / fewer edge types in
   `src/features/build_graph.py`) — that changes the experiment, so record it.

An OOM inside `scripts/hpo.py` is caught as a failed trial and does not consume
`--n-trials`; check `hpo/<study>/trial-*/train.log` for `CUDA out of memory`
before concluding a config is bad.

## 6. Using all 4 GPUs (there is no DDP)

`scripts/run_training.py`, `src/models/train.py` and the ablation copies all
take a single `--device cuda` and move one graph to `cuda:0`. There is no
`DistributedDataParallel`, no `DataParallel`, no torchrun entry point. Do **not**
try `torchrun`; the way to use 4×3090 is process-level parallelism:

```bash
# 4 independent HPO studies, one per GPU (each study keeps its own journal)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    python -m scripts.hpo run --n-trials 15 --n-jobs 1 --study-name gpu$i &
done
wait
```

Same trick for ablations and seeds:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m experiments.ablations.run_all_ablations --device cuda &
done
```

**Trap:** `scripts/hpo.py` does not pin GPUs. Its `worker_env()` sets thread
counts, `WANDB_MODE` and `PYTORCH_CUDA_ALLOC_CONF`, but never
`CUDA_VISIBLE_DEVICES`, and `run_trial()` inherits `os.environ` verbatim. So
`--n-jobs 4` on the 4-GPU box puts **all four trials on GPU 0** and leaves three
idle. Either set `CUDA_VISIBLE_DEVICES` per process as above, or add explicit
round-robin pinning to `worker_env()`/the trial launcher before trusting a
multi-GPU HPO run.

## 7. Machine-local paths to fix on the new box

| file | line | old value | action |
| --- | --- | --- | --- |
| `pyrightconfig.json` | `venvPath` | `/home/mses101137/miniconda3/envs` | point at the new conda root (or delete `venvPath`/`venv` and let pyright use the active env) |
| `docs/HPO.md` | §"Resuming, from scratch" | `cd /home/mses101137/charles/NYCU-BigData-Assignment2` | update the `cd` line |

Everything else resolves relative to the repo root: `DEFAULT_DB` is
`processed_tables/db`, outputs go to `splits/`, `hpo/`, `checkpoints/`,
`experiments/*/outputs/`. No absolute path appears in any tracked `.py` file.

## 8. Verification after setup

```bash
# unit tests: 167 passed, documented-split tests skip without processed_tables
python -m pytest -q

# data present and shaped as expected (8 tables, 2.58 M rows)
python -m src.data.inspect_graph_scale

# split builds, leakage checks run, guard refuses anything near the test week
python -m src.data.make_split --cutoff 2026-02-15 --days 7

# no GPU work, no W&B: builds splits + graph only
python -m scripts.run_training --dry-run --device cpu

# one short real step on each of the 4 GPUs
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m scripts.run_training \
    --epochs 1 --device cuda --wandb-mode disabled --run-name smoke-gpu$i
done
```

`tests/test_build_graph.py::...` and
`tests/test_temporal_split.py::test_real_data_*` skip when
`processed_tables/db/*.parquet` is missing — a green suite on a bare clone is
not evidence the data pipeline works. Run the four data-dependent commands
above.

## 9. Gotchas worth knowing before running anything

- **Validation data**: read `docs/VALIDATION_DATA.md` first. The only approved
  validation cutoff is `2026-02-15` (second choice `2026-03-08`); November 2026
  must never be scored. `src/data/make_split.py` now hard-refuses any cutoff at
  or after `2026-10-24`, including its own default — that `SystemExit` is a
  guard working as designed, not a bug to patch around.
- **Scores**: only the 60 clean Optuna trials in `hpo/hgt-hpo/trials.csv` are
  citable (best 65.81%, trial 41). The pre-HPO W&B numbers were leaked and must
  not be used as a baseline.
- **LightGBM device**: the pip wheel is CPU-only. `experiments/baselines/
  train_lightgbm.py --device gpu` will fail on the new box unless LightGBM is
  rebuilt with CUDA. The ablation orchestrator already uses `cpu`.
- **W&B**: `wandb` defaults to `online`. Put `WANDB_API_KEY` **inline on the
  command**, never `export` it and never commit it — the shell account here is
  shared, so an exported key is inherited by everyone else's runs on the box
  (and a key was committed once already and had to be purged):

  ```bash
  WANDB_API_KEY=<your own key> python -m scripts.run_training --epochs 310
  ```

  Use `--wandb-mode offline` / `disabled` for smoke runs and for anything that
  must not touch the network.
- **pandas 3 / numpy 2**: the code is written for this stack (float32 features,
  `pd.Timestamp` naive UTC throughout). Do not "downgrade for safety" — the
  test suite encodes the pandas-3 behaviours (copy-on-write, `sum` dtypes).
