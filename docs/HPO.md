# Hyperparameter search

Two files do all the work:

| file | role |
|---|---|
| `configs/hpo_space.yaml` | what to search (`space:`), what to freeze (`fixed:`), and the study settings |
| `scripts/hpo.py` | samples configs, shells out one `scripts/run_training.py` per trial, records the score |

Nothing in the training path changed to enable this, except one additive flag:
`--val-pred-path` (`scripts/run_training.py`), which defaults to the old
behaviour and exists so parallel trials do not overwrite one shared CSV.

## The objective

`scripts/hpo.py` reads the line the trainer already prints:

```
Best validation score: 65.81%
```

That is `best_val_score` = `(macro-F1 + micro-F1) / 2` on the held-out week,
with six per-activity thresholds fitted by CMA-ES **on training rows only**
(`src/models/train.py:183-219`). It is the same quantity as `val/score_cmaes`
in W&B and is the number `scripts/evaluate.py` reproduces. So a trial's Optuna
value is directly comparable to your grading metric.

> **Contaminated-result warning.** The pre-HPO W&B runs (all of
> `.wandb-runs/`, since deleted) reported 65.5-66.0% on this objective because
> of a leak: CMA-ES was allowed to fine-tune the per-activity thresholds on
> validation data. **Those numbers are invalid and must not be cited as a
> baseline, an upper bound, or evidence that any config is better than
> another.** The only trustworthy scores in this repo are the 60 Optuna trials
> in `hpo/hgt-hpo/trials.csv`, which share one fixed evaluation protocol and
> were therefore all inflated or not inflated equally. Best clean trial:
> **65.81%** (trial 41; `python -m scripts.hpo report --top 10` reprints the
> config). The consensus-of-top-10 defaults it implies are set as the argparse
> defaults in `scripts/run_training.py`, plus the three settings the ablation grid
> settled: `--epochs 310` (every one of the 27 ablation cells and all 60 trials
> ran 310), `--num-gnn-layers 1` (a second layer measured +0.11 @CMA-ES on 2/3
> seeds and -0.32 @0.5, i.e. noise, for +32% parameters and +65% wall time) and
> `--pos-weights auto` (the measured `(N_observed - N_pos)/N_pos` weights: +4.26
> @0.5 on 3/3 seeds over the doc constants). `--lr-schedule` is still
> `warm-restarts`, which is what the 27-cell grid ran; all 60 HPO trials ran
> `cosine` at 310 instead, and the two are within 0.1 of each other, so pass
> `--lr-schedule cosine` only to reproduce a specific trial's regime.

The trainer also re-runs `scripts/evaluate.py` on the final predictions as a
cross-check; each trial's copy lands in its own
`hpo/<study>/trial-NNNN/val_pred.csv`.

## Run it

```bash
# 1. see exactly what would be launched, train nothing
python -m scripts.hpo run --dry-run --n-trials 3

# 2. the real search: 60 trials, 2 concurrent (GPU-memory bound)
python -m scripts.hpo run --n-trials 60 --n-jobs 2

# 3. leaderboard at any time (safe while trials are running)
python -m scripts.hpo report --top 10
```

Interrupt with Ctrl-C, then rerun the same command: the SQLite journal
(`hpo/<study>/study.db`) resumes, and `--n-trials` is the total you want, not
the number to add. See "Stopping and resuming" below for the details,
including what to do if the process was killed hard instead.


## The two-stage plan

Tuning and confirmation happen on **different weeks on purpose**. A config that
wins on one 7-day window may not win on another, and the only way to find out
is to re-score it somewhere else.

```bash
# stage 1 -- search on the dense primary week (fixed.val_cutoff in the YAML)
python -m scripts.hpo run --n-trials 60 --n-jobs 2

# stage 2 -- re-score the top 3 on the latest legal week
python -m scripts.hpo confirm --top 3 --n-jobs 2 \
    --override val_cutoff=2026-03-14

# optional -- collapse check on the sparse recent regime.
# --days 5 changes the label window, so the score is NOT 7-day comparable:
# read it only as "does the model still beat copy-last-week?", never as a rank.
python -m scripts.hpo confirm --top 1 --n-jobs 1 \
    --override val_cutoff=2026-10-01 --override days=5
```

`confirm` writes `hpo/<study>/confirm-<date>-<days>d/confirm.csv` with
`hpo_score` vs `confirm_score` and the delta. If the ranking inverts between
stages, that is the finding: it means the configuration is fitting the week,
not the task, and the more conservative configuration is the safer submission.

Why these two weeks (from `docs/VALIDATION_DATA.md`):

- `2026-02-15` -- 30/30 usable lookback days, densest February regime
  (~295k `PullRequestEvent`/day). Cheapest per trial at 8 anchors (~58 min).
- `2026-03-14` -- the **latest** cutoff with a legal 7-day label window
  (`2026-03-02 .. 03-20` is the last long usable run; the absolute last usable
  day is `2026-10-21`). 12 anchors, so ~50% more compute per trial.
- `2026-10-01` -- only a 5-day window exists, so it is illegal for tuning
  and about 450 positives; use it as a sanity check, never a ranking.

## Cost

Measured on this machine (`run-20261112_085910-m1b81uc5`,
`run-20261112_093155-sp4iwemo`): ~10.8 s/epoch with 8 anchors and 40,403
training rows, 778k parameters. Cost is linear in `epochs` and roughly linear
in the number of anchors.

| stage | per trial | budget |
|---|---|---|
| search, 310 epochs, 8 anchors | ~58 min solo, ~96-106 min at `--n-jobs 2` | 60 trials / 2 workers ~= 46-51 h |
| confirm, 310 epochs, 12 anchors | ~85 min | 3 runs in parallel ~= 1.5 h |

Two concurrent trials do **not** halve the wall clock: measured, one trial solo
is 11.7 s/epoch and a pair costs 17.8 s/epoch together, so concurrency buys
about 1.31x, not 2x. `--n-jobs 2` is chosen for GPU memory (see below), not
throughput.

`--threads-per-job` defaults to `cores / n_jobs`; raise it if the GPU is idle
and the CPU is the bottleneck.

## Two things to know before spending the budget

**1. `epochs` is also the cosine cycle length.** There is no `--lr-t-max`:
`src/models/train.py:139-141` derives `scheduler_t_max` from `--epochs` and
builds `CosineAnnealingLR(T_max=epochs, eta_min=max(1e-5, lr*0.01))`
(`src/models/train.py:168`). `--lr-t0` / `--lr-t-mult` are ignored under
`cosine`; they only feed `CosineAnnealingWarmRestarts`. So changing `epochs`
does not just shorten training, it reshapes the learning-rate curve.

`epochs: 310` was originally justified by the per-epoch curve of
`run-20261112_093155-sp4iwemo` (a 1000-epoch run), which appeared to plateau
from epoch ~245 onward. **That run is one of the contaminated pre-HPO runs and
has been deleted, so the plateau claim is no longer evidence you can check.**
Do not re-derive an epoch budget from it.

What is still true and independent of any score: that run used `T_max=1000`,
so its lr at epoch 310 was ~7.8e-04, whereas with `epochs=310` the cosine is
compressed and lr at epoch 310 is 1.0e-05. Same epoch count, very different
trajectory. Since every one of the 60 trials above was run at 310, 310 is
the only epoch count with clean comparative support -- treat it as "the
budget the search was done at", not as a validated optimum. If you change it,
re-score a small set (150 / 310 / 450) under the same protocol rather than
trusting either the old curve or this comment.

**2. Do not compare against the README baseline table.** Those baselines
(copy-last-week 0.4961, GNN with time 0.5482) are scored on
`2026-09-27 .. 10-04`, the sparse recent regime. A February score in the
mid-60s is not evidence of beating them; the corpora differ by roughly 7x in
event density. The test week is expected to look far more like the sparse
regime, so expect the graded score to be well below any February number.

## Editing the search space

Entries under `space:` accept:

```yaml
lr: [3.0e-3, 1.0e-5]        # bare list -> inferred: log-uniform (hi/lo = 300)
dropout: [0.0, 0.5]         # inferred: uniform (hi/lo < 10)
hidden_dim: [64, 128, 256]  # all ints -> categorical
num_gnn_layers: "int: [1, 2]"   # explicit integer range, inclusive
weight_decay: "log: [1.0e-6, 1.0e-2]"   # explicit, quoted
```

Three YAML traps, all verified:

- Write `1.0e-4`, never `1e-4`. PyYAML reads bare `1e-4` as the **string**
  `"1e-4"`, which silently becomes a categorical choice.
- Quote the prefixed forms. Unquoted `int: [1, 3]` parses as a nested mapping
  and the file fails to load.
- `null` means "do not search, use the trainer default".

### The lr range is constrained by `eta_min`

`src/models/train.py:114-116` floors the cosine at
`eta_min = max(1e-5, base_lr * 0.01)`. That floor is a hard lower bound on any
sensible `lr` search: below `1e-5` the floor sits *above* the base lr and
`CosineAnnealingLR` anneals **upward**, so a trial with `lr=1e-6` is not a
low-lr trial, it is a different schedule (1e-6 -> 5.5e-6 -> 1e-5). Measured:

| base lr | lr at epoch 310 | shape |
|---|---|---|
| 3e-3 / 1e-3 / 1e-4 | 1.00e-05 | full cosine decay |
| 3e-5 | 1.00e-05 | mild decay (33% drop) |
| 1e-5 | 1.00e-05 | constant |
| 5e-6 / 1e-6 | 1.00e-05 | **rises** |

So the space is `[3.0e-3, 1.0e-5]`: the upper end is 3x above the validated
`1e-3` incumbent (an incumbent on a boundary can only be confirmed, never
beaten), and the lower end is the `eta_min` floor rather than a decade of
mislabeled non-annealing trials.

Constraints are filtered, not retried:

```yaml
constraints:
  - name: heads_divide_latent
    rule: latent_dim % num_heads == 0   # HGTConv requires this
    dependent: num_heads                # the key whose choices get filtered
```

With `latent_dim {32,64,128,256}` x `num_heads {2,4,8}` every pair is legal, so
this is currently belt-and-braces. Keep it if you edit either list. Rules are
evaluated by a small AST whitelist (names, numbers, `+ - * // %`, one
comparison) -- there is no `eval`, and a YAML file cannot execute code.

8 dimensions is close to the practical ceiling for TPE at ~60 trials; below
about 10 trials per dimension the surrogate is mostly guessing. If you need to
shrink it, drop `weight_decay` and `head_hidden_dim` first.

## Stopping and resuming

This search is long (46-51 h at 60 trials), so it will almost certainly be
interrupted and picked up later, possibly by another agent with no memory of
this conversation. Everything needed is on disk; nothing lives in memory.

### What survives a stop, and what does not

Survives, because each trial is committed the moment it finishes:

- `hpo/<study>/study.db` -- the Optuna journal, including the TPE history.
- `hpo/<study>/trials.csv` -- one row per finished trial.
- `hpo/<study>/trial-NNNN/{train.log,model.pt,val_pred.csv}`.
- `hpo/<study>/space.snapshot.yaml` -- the config that produced the study.

Lost: the partial work of any trial that was mid-flight. At roughly 100
minutes per trial that is the only real cost of stopping, so stopping between
trials is free-ish while stopping mid-trial throws away up to ~1.7 h per
running worker.

### Ctrl-C is now cheap

The driver installs a `SIGINT` handler. On Ctrl-C it terminates the trainer
subprocesses (SIGTERM to each process group, SIGKILL after a 20 s grace),
reports how many it stopped, and exits with status 130 in about a tenth of a
second. This matters because the default `ThreadPoolExecutor` shutdown waits
for running futures -- a bare `KeyboardInterrupt` used to appear to hang for
an entire trial.

Two follow-on effects, both automatic on the next start:

- Trials left `RUNNING` are marked `FAIL` and reported as
  `[note] marked N trial(s) left RUNNING by an earlier run as FAIL`. This is
  necessary, not cosmetic: Optuna's TPE treats a `RUNNING` trial as a
  constant-liar observation, so a stale row would keep pulling the surrogate
  towards a configuration that was never actually scored.
- The `driver.lock` left behind is detected as stale (its pid is gone) and
  replaced.

Failed trials do not consume the budget: `--n-trials` counts *completed*
trials, so an interrupted run still reaches 60 real results.

### Resuming, from scratch

```bash
cd /train-data-5-hdd/charles/NYCU-BigData-Assignment2
conda activate bda_asg2     # torch 2.11.0+cu128, optuna 5.0.0

# 0. did anything survive the stop? orphaned trainers from a hard kill
#    would compete for the GPU and corrupt their trial rows.
pgrep -af 'run_training|scripts.hpo'
nvidia-smi                  # a trial needs ~22-41 GiB; 2 workers ~= 45-80 GiB

# 1. where did it get to?
python -m scripts.hpo report --top 10
wc -l hpo/hgt-hpo/trials.csv

# 2. continue -- the SAME command, n-trials is the total, not the delta
python -u -m scripts.hpo run --n-trials 60 --n-jobs 2 2>&1 | tee -a hpo/hpo-run.log

# 3. after it finishes: re-score the top 3 on the later week
python -m scripts.hpo confirm --top 3 --n-jobs 2 --override val_cutoff=2026-03-14
```

Run step 2 inside tmux so it survives a disconnect (`tmux attach -t bda1`,
window `hpo`). Keep `-u` and `tee`: the driver's own output is block-buffered
to a file otherwise, which makes a working run look dead.

If `report` says "no study named ...", the journal is missing and this is a
fresh start -- step 2 then begins at trial 0, seeded with the incumbent
configuration from `study.enqueue` in `configs/hpo_space.yaml`. The seed is
only enqueued while the study has no trials, so resuming never duplicates it.

### One rule: one driver per study

Calling `run` twice on the same study is refused (`another driver is already
running on ... (pid N)`, exit 1). This is deliberate. The second driver would
mark the first driver's live trials as `FAIL`, then both would try to `tell`
the same trial numbers and Optuna would reject the later writes -- so the
study ends up with failed rows for work that actually happened. If you see
that message and believe it is wrong, check `pgrep -af scripts.hpo` before
deleting `hpo/<study>/driver.lock` by hand.

### If it was killed hard

`kill -9`, an OOM, a dropped tmux session or a reboot can leave trainer
processes running with no driver to reap them. They keep holding GPU memory
and will finish their epoch loop for nothing. Clean up first, then resume:

```bash
pkill -f 'scripts.hpo run'          # the driver
pkill -f 'scripts.run_training'     # its children, if orphaned
sleep 5 && nvidia-smi              # confirm the memory is actually freed
python -m scripts.hpo report       # RUNNING rows are marked FAIL on restart
```

Never edit `hpo/<study>/study.db` or `trials.csv` while a driver is running,
and never point a test harness at the live study directory -- the resume logic
writes to whatever `HPO_ROOT` resolves to.

## Artifacts

```
hpo/<study>/
  study.db              Optuna journal (resume here)
  space.snapshot.yaml   the config that produced this study
  trials.csv            one row per finished trial
  trial-0000/
    train.log           full trainer output (unbuffered)
    model.pt            best checkpoint
    val_pred.csv        that trial's predictions
  confirm-2026-03-14-7d/
    confirm.csv         hpo_score vs confirm_score
```

`hpo/` is gitignored.
