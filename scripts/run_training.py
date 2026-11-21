"""Train the HGT model end to end and stream train/val curves to W&B.

This is the driver that `src/models/train.py` was missing: it turns raw parquet
tables into a leakage-checked temporal split, builds the heterogeneous graph,
aligns the labels to graph node order, and then calls ``train_model``.

Two modes, and the default is the submission run:

``--mode final`` (default)
    No held-out week.  Trains on every anchor the coverage audit can support up
to the last complete week before the graded test week, fits the six thresholds
on the training anchors, builds the test-week feature graph at ``--test-cutoff``
and writes ``submission/submission.csv``.  This is the run described in
``docs/FINAL_RUN.md``.

``--mode validate``
    The measurement protocol: a clean held-out week (default 2026-02-15) scored
    by the TA metric, thresholds fitted by CMA-ES on the training anchors only.
    Every ablation cell and all 60 HPO trials used this path.

Temporal setup for the validation mode (from docs/VALIDATION_DATA.md):

  train: features before 2026-02-08, labels 2026-02-08 .. 2026-02-15
  val:   features before 2026-02-15, labels 2026-02-15 .. 2026-02-22

Both label windows were verified to have 7 usable days for all six activities
in ``raw_data/coverage_activity.csv``.  Features for the validation graph may
include the training label window (that is the normal "predict the next week
from everything known at the cutoff" setup); the labels themselves never
overlap.

Examples
--------
    # the submission run: train on everything usable, predict 2026-11-22..29
    python -m scripts.run_training

    # same, but skip training and re-predict from the saved checkpoint
    python -m scripts.run_training --predict-only

    # the measurement protocol (validation week 2026-02-15)
    python -m scripts.run_training --mode validate

    # sanity check the data plumbing, no training
    python -m scripts.run_training --dry-run

    # short run to see the curves
    python -m scripts.run_training --mode validate --epochs 3 --val-interval 1

    # no network
    python -m scripts.run_training --wandb-mode offline
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.data.label_coverage import load_day_flags, observed_activities
from src.data.make_split import (
    assert_before_test_week,
    build_split,
    cohort_users,
    leakage_failures,
)
from src.data.split_io import DEFAULT_DB, load_tables, parse_cutoff
from src.data.temporal_split import LABEL_COLS, extract_labels, mask_future_leakage
from src.features.build_graph import build_hetero_graph
from src.models.hetero_model import GitHubHeteroHGT
from src.models.train import (
    ACTIVITY_NAMES,
    Anchor,
    train_model,
    train_model_final,
    train_model_multi_anchor,
)

# Measured on this corpus (docs/VALIDATION_DATA.md section 5).
DOC_POS_WEIGHTS: list[float] = [7.28, 18.08, 11.05, 12.83, 19.29, 1.32]
DOC_TAU_DAYS: list[float] = [5.96, 5.37, 6.02, 6.73, 6.91, 7.50]

# README reference scores on 2026-09-27 .. 10-04, for context in the report.
BASELINES = {
    "predict-all-0": 0.0000,
    "random": 0.2126,
    "GNN without time": 0.3570,
    "copy-last-week": 0.4961,
    "GNN with time": 0.5482,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train HGT on a leakage-checked temporal split, log to W&B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--db", default=DEFAULT_DB, help="parquet folder with the tables")
    ap.add_argument(
        "--mode",
        choices=("final", "validate"),
        default="final",
        help=(
            "final: no held-out week, train on every usable anchor, fit thresholds "
            "on train rows, predict the graded test week and write the submission. "
            "validate: the held-out-week measurement protocol (default 2026-02-15)."
        ),
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="shorthand for --mode final (already the default)",
    )
    ap.add_argument(
        "--test-cutoff",
        default="2026-11-21 16:00:00",
        help=(
            "feature cutoff for the submission: the graded week is 2026-11-22..29 "
            "Taiwan time = 2026-11-21 16:00 .. 11-28 16:00 UTC"
        ),
    )
    ap.add_argument(
        "--test-input",
        default="submission/test_input.csv",
        help="official list of test users (UserID column)",
    )
    ap.add_argument(
        "--submission-out",
        default="submission/submission.csv",
        help="where to write the test-week predictions",
    )
    ap.add_argument(
        "--predict-only",
        action="store_true",
        help=(
            "skip training: load --checkpoint and only build the test graph and "
            "write the submission (use after refreshing raw data / preprocessing)"
        ),
    )
    ap.add_argument(
        "--anchor-dense-end",
        default="2026-02-09",
        help=(
            "final mode: last anchor of the weekly dense block (the regime the "
            "ablation grid validated); after it anchors are --anchor-stride apart"
        ),
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="final mode: write the checkpoint every N epochs (0 disables)",
    )
    ap.add_argument(
        "--threshold-lookback-days",
        type=int,
        default=84,
        help=(
            "final mode: fit the six thresholds on the anchors within this many days "
            "of the last training anchor, so the operating point comes from the "
            "regime the test week lives in (open_pr prevalence 4.0%% in Nov vs "
            "15.2%% in Feb). 0 = use every anchor."
        ),
    )
    ap.add_argument(
        "--val-cutoff",
        default="2026-02-15",
        help="validation cutoff (features < cutoff, labels in the next --days days)",
    )
    ap.add_argument(
        "--cutoff",
        default=None,
        help="training cutoff (default: val-cutoff minus --days)",
    )
    ap.add_argument("--days", type=int, default=7, help="label window length")
    ap.add_argument(
        "--anchors",
        default="auto",
        help=(
            "'auto' = weekly anchors from --anchor-start to val-cutoff-days, "
            "'none' = single anchor at --cutoff, or a comma list of cutoff dates"
        ),
    )
    ap.add_argument(
        "--anchor-start",
        default="2025-12-22",
        help="earliest training anchor (needs ~30d of history before it)",
    )
    ap.add_argument(
        "--anchor-end", default=None, help="latest training anchor (default: val-cutoff - days)"
    )
    ap.add_argument("--anchor-stride", type=int, default=7, help="days between anchors")
    ap.add_argument(
        "--anchor-stride-final",
        type=int,
        default=28,
        help=(
            "final mode: anchor spacing after --anchor-dense-end. 28 keeps the "
            "run near three hours on this corpus (measured 1.3 s per million "
            "graph nodes per epoch, ~23M nodes per epoch at stride 28)"
        ),
    )
    ap.add_argument(
        "--anchor-observed",
        choices=("hybrid", "label", "strict", "off"),
        default=None,
        help=(
            "usability rule from coverage_activity.csv for a label cell. "
            "hybrid (final-mode default): audit OR label-count sanity, so the "
            "intact 2026-10/11 pull-request cells are usable while the collapsed "
            "push stream is dropped. label (validate-mode default): the audit "
            "alone. strict: audit over the label window and the feature lookback. "
            "off: trust every cell."
        ),
    )
    ap.add_argument(
        "--anchor-min-observed",
        type=int,
        default=1,
        help="skip an anchor unless at least N of its 6 activities are observable",
    )
    ap.add_argument("--max-anchors", type=int, default=0, help="cap anchors, newest first (0 = all)")
    ap.add_argument(
        "--no-anchor-shuffle",
        action="store_true",
        help="visit anchors in chronological order instead of shuffling per epoch",
    )
    ap.add_argument(
        "--val-mode",
        choices=("temporal", "cohort"),
        default="temporal",
        help="temporal: separate earlier train week. cohort: random 80/20 on one week",
    )
    ap.add_argument("--val-fraction", type=float, default=0.2, help="cohort mode holdout")
    ap.add_argument(
        "--epochs",
        type=int,
        default=310,
        help=(
            "training epochs (default 310: every ablation cell and all 60 clean "
            "HPO trials ran 310; 150 was never measured against the grid)"
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--lr-schedule",
        choices=("cosine", "warm-restarts", "none"),
        default="warm-restarts",
        help=(
            "warm-restarts: sawtooth with geometrically growing cycles, lr snaps "
            "back to --lr at the end of each cycle (default). cosine: one decay "
            "over the whole run (bigger --epochs stretches it, it does not "
            "repeat). none: constant lr."
        ),
    )
    ap.add_argument(
        "--lr-t0",
        type=int,
        default=5,
        help="warm-restarts: epochs in the first cycle (default 5)",
    )
    ap.add_argument(
        "--lr-t-mult",
        type=int,
        default=2,
        help=(
            "warm-restarts: cycle-length multiplier, so cycles run "
            "t0, t0*t_mult, t0*t_mult**2, ... (2 = 5,10,20,40; 1 = constant)"
        ),
    )
    ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--head-hidden-dim", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument(
        "--num-gnn-layers",
        type=int,
        default=1,
        help=(
            "HGT layers (default 1). A second layer measured +0.11 @CMA-ES on "
            "2/3 seeds and -0.32 @0.5, i.e. noise, for +32%% parameters and "
            "+65%% wall time; HPO's top 10/10 trials also used 1 layer. 2 layers "
            "additionally needs activation checkpointing to fit a 24 GiB card, "
            "which this model does not implement."
        ),
    )
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm")
    ap.add_argument("--cma-maxfevals", type=int, default=3000)
    ap.add_argument("--cma-popsize", type=int, default=16)
    ap.add_argument("--cma-sigma0", type=float, default=0.15)
    ap.add_argument("--project", default="nycu-bigdata-asg2")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pos-weights",
        default="auto",
        help="'doc' (measured constants), 'auto' (from train prevalence), or 6 comma floats",
    )
    ap.add_argument("--checkpoint", default=None, help="path for the best model state")
    ap.add_argument(
        "--val-pred-path",
        default=None,
        help=(
            "where to write the validation predictions CSV "
            "(default: splits/val_best_pred_<val-cutoff>.csv). Set this when "
            "several runs share a val cutoff, e.g. parallel HPO trials."
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="build splits only, no training")
    return ap.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_window_usability(cutoff: pd.Timestamp, days: int) -> list[str]:
    """Warn about (day, activity) cells the audit marks unusable in the label window."""
    path = Path("raw_data/coverage_activity.csv")
    if not path.exists():
        print("  [warn] raw_data/coverage_activity.csv missing; skipping usability check")
        return []

    coverage = pd.read_csv(path)
    cov_day = pd.to_datetime(coverage["day"]).to_numpy()
    start_np = np.datetime64(cutoff.to_datetime64())
    end_np = np.datetime64((cutoff + pd.Timedelta(days=days)).to_datetime64())
    in_window = (cov_day >= start_np) & (cov_day < end_np)
    if not in_window.any():
        return [f"no coverage rows for {cutoff.date()} .. {(cutoff + pd.Timedelta(days=days)).date()}"]

    problems: list[str] = []
    n_days = int(np.unique(cov_day[in_window]).size)
    if n_days < days:
        problems.append(f"label window has only {n_days}/{days} days of coverage")

    window = coverage.loc[in_window]
    unusable = ~window["usable"].astype(bool).to_numpy()
    for rec in window.loc[unusable].itertuples(index=False):
        day = str(np.datetime64(rec.day, "D"))
        problems.append(f"{day} {rec.activity}: {rec.hours_ok}/{rec.hours} hours usable")
    return problems


def align_labels(
    labels_df: pd.DataFrame, users_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter labels into graph-node order (graph Users node i == users_df row i).

    ``labels_df`` is indexed by the internal ``user_id``, while graph nodes are
    ordered by ``users_df`` rows, so the join must be on ``user_id``.
    """
    n_users = len(users_df)
    y = np.zeros((n_users, len(LABEL_COLS)), dtype=np.int64)
    present = np.zeros(n_users, dtype=bool)

    node_pos = pd.Series(np.arange(n_users), index=users_df["user_id"].to_numpy())
    label_rows = labels_df.set_index("user_id")
    shared = node_pos.index.intersection(label_rows.index)
    if len(shared) == 0:
        raise SystemExit("no user_id overlap between the graph and the labels")

    rows = node_pos.loc[shared].to_numpy()
    y[rows] = label_rows.loc[shared, LABEL_COLS].to_numpy(dtype=np.int64)
    present[rows] = True
    return y, present


def load_dataset(
    tables: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    days: int,
    name: str,
    observed_activity: np.ndarray | None = None,
) -> Dataset:
    """Build one leakage-checked split and its graph.

    ``observed_activity`` is a `(6,)` bool vector from the coverage audit.  An
    activity the audit could not measure gets its label column zeroed *and*
    marked unobserved, so the loss skips it rather than reading a false "0".
    """
    started = time.time()
    split = build_split(tables, cutoff, days)
    failures = leakage_failures(tables, split)
    if failures:
        joined = "\n  - ".join(failures)
        raise SystemExit(f"leakage check FAILED for {cutoff.date()}:\n  - {joined}")

    graph = build_hetero_graph(split.features, cutoff)
    users = split.features["Users"].reset_index(drop=True)
    if graph["Users"].num_nodes != len(users):
        raise SystemExit(
            f"graph has {graph['Users'].num_nodes} user nodes but {len(users)} user rows"
        )

    y, present = align_labels(split.labels, users)
    flags = (
        np.ones(len(LABEL_COLS), dtype=bool)
        if observed_activity is None
        else np.asarray(observed_activity, dtype=bool).reshape(len(LABEL_COLS))
    )
    observed = np.broadcast_to(flags, y.shape).copy()
    y = np.where(observed, y, 0)

    is_cohort = users["is_cohort"].fillna(False).to_numpy(dtype=bool)
    mask = is_cohort & present
    user_ids = users["github_user_id"].to_numpy()[mask]

    positives = np.array(
        [int(y[mask & observed[:, i], i].sum()) for i in range(len(ACTIVITY_NAMES))]
    )
    if mask.sum() == 0:
        raise SystemExit(f"no cohort users with labels at cutoff {cutoff.date()}")

    print(
        f"  [{name}] cutoff {cutoff.date()} | label window "
        f"{split.cutoff.date()} .. {split.end.date()} | leakage check: OK"
    )
    print(
        f"  [{name}] nodes {graph['Users'].num_nodes:,} users, "
        f"{mask.sum():,} cohort rows, {sum(graph[nt].num_nodes for nt in graph.node_types):,} total"
    )
    if not flags.all():
        dropped = [a for a, f in zip(ACTIVITY_NAMES, flags, strict=True) if not f]
        print(f"  [{name}] unobserved activities (dropped from the loss): {', '.join(dropped)}")
    print(
        f"  [{name}] positives " + "  ".join(
            (f"{a}=0/-" if not f else f"{a}={int(p):,} ({p / max(mask.sum(), 1) * 100:.1f}%)")
            for a, p, f in zip(ACTIVITY_NAMES, positives, flags, strict=True)
        )
    )
    print(f"  [{name}] built in {time.time() - started:.1f}s")

    for act, p, f in zip(ACTIVITY_NAMES, positives, flags, strict=True):
        if f and p == 0:
            print(f"  [warn] {name}: activity '{act}' has zero positives - bad window")
    return Dataset(name, cutoff, graph, y, observed, mask, user_ids, positives)


def cohort_holdout(mask: np.ndarray, y: np.ndarray, fraction: float, seed: int):
    """Stratified train/val masks over the same graph (cohort mode)."""
    rng = np.random.default_rng(seed)
    n = int(mask.sum())
    idx = np.flatnonzero(mask)
    val = np.zeros(n, dtype=bool)

    # hold out `fraction` of the positives of each activity, plus random negatives
    for col in range(y.shape[1]):
        pos = idx[y[idx, col] == 1]
        if len(pos):
            val[rng.choice(len(pos), size=max(1, int(round(len(pos) * fraction))), replace=False)] = True
    neg = idx[y[idx].sum(axis=1) == 0]
    if len(neg):
        val[rng.choice(len(neg), size=int(round(len(neg) * fraction)), replace=False)] = True

    # never empty
    if val.all() or not val.any():
        val[:] = False
        val[rng.permutation(n)[: max(1, int(round(n * fraction)))]] = True

    train_mask = np.zeros_like(mask)
    val_mask = np.zeros_like(mask)
    train_mask[idx[~val]] = True
    val_mask[idx[val]] = True
    return train_mask, val_mask


@dataclass
class Dataset:
    """One temporal split, already aligned to graph node order."""

    name: str
    cutoff: pd.Timestamp
    graph: HeteroData
    labels: np.ndarray  # (n_users, 6) int64
    observed: np.ndarray  # (n_users, 6) bool, label was measurable
    mask: np.ndarray  # (n_users,) bool, cohort users we are scored on
    user_ids: np.ndarray  # github_user_id of the masked rows
    positives: np.ndarray  # (6,) positive count per activity

    def to_anchor(self) -> Anchor:
        return Anchor(
            name=self.name,
            cutoff=self.cutoff,
            graph=self.graph,
            labels=self.labels,
            observed=self.observed,
            mask=self.mask,
            positives=self.positives,
        )

    @property
    def graph_megabytes(self) -> float:
        total = 0
        for node_type in self.graph.node_types:
            store = self.graph[node_type]
            x = getattr(store, "x", None)
            if x is not None:
                total += x.numel() * x.element_size()
        for edge_type in self.graph.edge_types:
            store = self.graph[edge_type]
            total += store.edge_index.numel() * store.edge_index.element_size()
            attr = getattr(store, "edge_attr", None)
            if attr is not None:
                total += attr.numel() * attr.element_size()
        return total / 2**20


def anchor_cutoffs(
    spec: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stride: int,
) -> list[pd.Timestamp]:
    """Weekly (or `stride`-day) cutoffs from `start` through `end`, inclusive."""
    if spec != "auto":
        listed = [parse_cutoff_arg(part) for part in spec.split(",") if part.strip()]
        if not listed:
            raise SystemExit("--anchors needs dates or 'auto'")
        return sorted(listed)
    if stride < 1:
        raise SystemExit("--anchor-stride must be >= 1")
    if end < start:
        raise SystemExit(
            f"anchor start {start.date()} is after the last allowed anchor {end.date()}"
        )
    count = (end - start).days // stride + 1
    cutoffs = [cast(pd.Timestamp, start + pd.Timedelta(days=stride * i)) for i in range(count)]
    if cutoffs and cutoffs[-1] != end:
        cutoffs.append(end)
    return cutoffs


def resolve_pos_weights(
    spec: str,
    train_positives: np.ndarray,
    n_rows: int,
    observed_counts: np.ndarray | None = None,
) -> list[float]:
    """Turn a ``--pos-weights`` mode into the six BCE positive-class weights.

    ``doc``
        The static 252-clean-anchor constants from
        ``docs/VALIDATION_DATA.md`` section 5 (``DOC_POS_WEIGHTS``).  They were
        fitted on a different, wider panel than any single run trains on, and
        the ablation grid measured them as 2.0-2.6x the prevalence ratio of the
        anchors a run actually sees: keeping them costs 4.26 points at a fixed
        0.5 threshold (``experiments/ablations/RESULTS.md``, Experiment C).
    ``auto`` (default)
        ``(N_observed - N_pos) / N_pos`` per activity, computed from the
        training anchors of *this* run.  The denominator is the number of
        **observed** cells for that activity, not the row count: a cell the
        coverage audit marked unobserved never enters the loss, so counting it
        as a negative would deflate the weight for exactly the activities that
        lost days.  Mathematically identical to ``(1 - p) / p`` with
        ``p = N_pos / N_observed``, which is the usual BCE ``pos_weight``.
        Measured: +4.26 score @0.5 (3/3 seeds) and +0.13 @CMA-ES over ``doc``,
        with the fitted CMA-ES thresholds dropping from 0.76-0.89 to 0.63-0.81
        because theta no longer has to undo the inflated weights.
    ``<six comma-separated floats>``
        Explicit override, e.g. ``--pos-weights 7.3,18,11,12.8,19.3,1.3``.

    ``observed_counts`` is optional so older callers (and tests) that only know
    the row count keep working; it then falls back to ``n_rows`` per activity,
    which is what the observed count equals when every label cell is observed.
    """
    if spec == "doc":
        return list(DOC_POS_WEIGHTS)
    if spec == "auto":
        positives = np.asarray(train_positives, dtype=np.float64).reshape(-1)
        if positives.size != len(ACTIVITY_NAMES):
            raise SystemExit(
                f"auto pos-weights needs {len(ACTIVITY_NAMES)} per-activity "
                f"positive counts, got {positives.size}"
            )
        if observed_counts is None:
            denominator = np.full(len(ACTIVITY_NAMES), float(max(n_rows, 1)))
        else:
            denominator = np.asarray(observed_counts, dtype=np.float64).reshape(-1)
            if denominator.size != len(ACTIVITY_NAMES):
                raise SystemExit(
                    f"auto pos-weights needs {len(ACTIVITY_NAMES)} per-activity "
                    f"observed counts, got {denominator.size}"
                )
            denominator = np.maximum(denominator, 1.0)
        # clip keeps a zero-positive activity finite instead of dividing by zero
        positives = np.clip(positives, 1.0, None)
        negatives = np.maximum(denominator - positives, 1.0)
        return [float(w) for w in negatives / positives]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != len(ACTIVITY_NAMES):
        raise SystemExit(f"--pos-weights needs {len(ACTIVITY_NAMES)} comma-separated values")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"--pos-weights values must be numbers: {exc}") from exc


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def parse_cutoff_arg(value: str) -> pd.Timestamp:
    """parse_cutoff with an explicit non-NaT Timestamp type."""
    ts = cast(pd.Timestamp, parse_cutoff(value))
    if pd.isna(ts):
        raise SystemExit(f"invalid cutoff date: {value!r}")
    return ts


def resolve_mode(args: argparse.Namespace) -> str:
    """`--final` is a shorthand that wins over the default `--mode`."""
    return "final" if args.final or args.mode == "final" else "validate"


def resolve_anchor_observed(args: argparse.Namespace) -> str:
    """Default observability rule: hybrid for the submission run, audit otherwise."""
    if args.anchor_observed:
        return str(args.anchor_observed)
    return "hybrid" if resolve_mode(args) == "final" else "label"


def final_anchor_cutoffs(
    *,
    start: pd.Timestamp,
    dense_end: pd.Timestamp,
    end: pd.Timestamp,
    stride: int,
) -> list[pd.Timestamp]:
    """Weekly anchors over the dense regime, then `stride`-day anchors to the end.

    The ablation grid, the HPO study and every validated score in this repo come
    from weekly anchors over 2025-12-22 .. 2026-02-08, i.e. the dense regime.
    The graded week lives in the sparse regime (a tenth of that event volume),
    so the submission run keeps that validated block **and** continues towards
    the test week.  Continuing weekly would put ~40 anchors and 50M graph nodes
    into every epoch; the graph grows monotonically with history, so the late
    anchors are also the expensive ones.  A stride of 28 days past the dense
    block costs ~20M nodes per epoch and keeps every month of the sparse regime
    represented.
    """
    if stride < 1:
        raise SystemExit("--anchor-stride-final must be >= 1")
    if end < start:
        raise SystemExit(f"anchor start {start.date()} is after the last allowed anchor {end.date()}")

    cutoffs: list[pd.Timestamp] = []
    current = start
    last_dense = min(cast(pd.Timestamp, dense_end), cast(pd.Timestamp, end))
    while current <= last_dense:
        cutoffs.append(cast(pd.Timestamp, current))
        current = cast(pd.Timestamp, current + pd.Timedelta(days=7))

    if end > dense_end:
        current = cast(pd.Timestamp, dense_end + pd.Timedelta(days=stride))
        while current < end:
            cutoffs.append(current)
            current = cast(pd.Timestamp, current + pd.Timedelta(days=stride))
        cutoffs.append(cast(pd.Timestamp, end))
    elif end not in cutoffs:
        cutoffs.append(cast(pd.Timestamp, end))
    return sorted(set(cutoffs))


def anchor_positives(
    tables: dict[str, pd.DataFrame], cutoffs: list[pd.Timestamp], days: int
) -> np.ndarray:
    """`(n_anchors, 6)` positive counts per anchor, from the labels' own tables.

    Needed by the hybrid observability rule: it judges a label cell by how many
    positives the 7-day window actually produced.  Cheap on purpose -- it reads
    the six label tables and never materialises a feature view.
    """
    cohort = cohort_users(tables)
    window = pd.Timedelta(days=days)
    counts: list[np.ndarray] = []
    for cutoff in cutoffs:
        start = cast(pd.Timestamp, cutoff)
        end = cast(pd.Timestamp, start + window)
        labels = extract_labels(tables, start, end, cohort)
        counts.append(labels[LABEL_COLS].sum().to_numpy())
    return np.array(counts)


def build_inference_graph(
    tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp, name: str = "test"
) -> tuple[HeteroData, pd.DataFrame]:
    """Feature-only graph at `cutoff`. No labels are read anywhere here."""
    started = time.time()
    features = mask_future_leakage(tables, cutoff)
    graph = build_hetero_graph(features, cutoff)
    users = features["Users"].reset_index(drop=True)
    if graph["Users"].num_nodes != len(users):
        raise SystemExit(
            f"graph has {graph['Users'].num_nodes} user nodes but {len(users)} user rows"
        )
    total = sum(graph[node_type].num_nodes for node_type in graph.node_types)
    latest = max(
        (cast(pd.Series, features[name][col]).max() for name, col in {
            "Users": "first_seen_at",
            "PullRequests": "first_seen_at",
            "Pushes": "created_at",
            "IssueComments": "created_at",
        }.items() if name in features and col in features[name].columns),
        default=pd.NaT,
    )
    print(
        f"  [{name}] cutoff {cutoff} | latest row {latest} | "
        f"{graph['Users'].num_nodes:,} users, {total:,} nodes | {time.time() - started:.1f}s"
    )
    return graph, users


def predict_test_week(
    *,
    model: GitHubHeteroHGT,
    graph: HeteroData,
    users: pd.DataFrame,
    thresholds: np.ndarray,
    test_input: str,
    out_path: str,
    probs_path: str,
    device: str,
) -> int:
    """Score the test users, write the submission CSV, and format-check it."""
    wanted = pd.read_csv(test_input)
    if "UserID" not in wanted.columns:
        raise SystemExit(f"{test_input} has no UserID column")
    wanted_ids = wanted["UserID"].to_numpy(dtype=np.int64)

    node_pos = pd.Series(
        np.arange(len(users)), index=users["github_user_id"].to_numpy(dtype=np.int64)
    )
    present = np.isin(wanted_ids, node_pos.index.to_numpy())
    rows = np.full(len(wanted_ids), -1, dtype=np.int64)
    rows[present] = node_pos.loc[wanted_ids[present]].to_numpy()
    if not present.any():
        raise SystemExit("no test user appears in the feature view; check --test-cutoff")

    moved = graph.to(device)
    probabilities = np.zeros((len(wanted_ids), len(ACTIVITY_NAMES)), dtype=np.float64)
    probabilities[present] = model.predict_proba(moved).cpu().numpy()[rows[present]]
    del moved
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    binary = (probabilities >= np.asarray(thresholds, dtype=np.float64)[None, :]).astype(np.int64)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(binary, columns=list(ACTIVITY_NAMES))
    submission.insert(0, "UserID", wanted_ids)
    submission.to_csv(out, index=False)

    probs_file = Path(probs_path)
    probs_file.parent.mkdir(parents=True, exist_ok=True)
    prob_frame = pd.DataFrame(probabilities, columns=list(ACTIVITY_NAMES))
    prob_frame.insert(0, "UserID", wanted_ids)
    prob_frame.to_csv(probs_file, index=False)

    print(f"\nWrote {out} ({len(submission):,} rows)")
    print(f"Wrote {probs_file} (probabilities, for inspection)")
    if not present.all():
        print(
            f"  [warn] {int((~present).sum())} test users have no row before the cutoff; "
            "they are predicted all-zero"
        )
    print("predicted positives per activity:")
    for i, act in enumerate(ACTIVITY_NAMES):
        n = int(binary[:, i].sum())
        print(
            f"  {act:<20} {n:>5,} ({n / len(binary) * 100:5.1f}%)  "
            f"theta {thresholds[i]:.3f}"
        )

    print("\nFormat check against the official test list:", flush=True)
    subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--pred", str(out), "--input", test_input],
        check=False,
    )
    return 0


def preflight_largest_anchor(
    model: GitHubHeteroHGT, anchors: list[Anchor], device: str
) -> None:
    """One forward+backward on the largest graph before committing to hours of training.

    The graphs grow monotonically with history, so a run that cannot fit the last
    anchor will die at epoch 1 rather than at epoch 309 -- but only if it is
    checked first.  On this corpus the largest anchor needs ~66 GiB
    (25.6 GiB per million nodes, measured), so a co-tenant process on the GPU is
    the difference between a two-hour run and an immediate OOM.
    """
    if not anchors:
        return
    biggest = max(anchors, key=lambda a: a.graph_megabytes)
    print(
        f"  preflight: largest anchor {biggest.cutoff.date()} "
        f"({biggest.graph_megabytes:.0f} MB graph, {biggest.n_rows:,} rows)"
    )
    model = model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    graph = biggest.graph.to(device)
    mask = torch.from_numpy(biggest.mask).to(device=device, dtype=torch.bool)
    try:
        logits = model(graph)[mask]
        target = torch.from_numpy(biggest.labels[biggest.mask]).float().to(device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            "preflight OOM on the largest training anchor.\n"
            "Measured cost on this corpus: ~25.6 GiB of VRAM per million graph "
            f"nodes; this anchor has {sum(graph[nt].num_nodes for nt in graph.node_types):,} nodes.\n"
            "Free the GPU (another process is holding VRAM), or lower the anchor "
            "range (--anchor-end / --anchor-stride-final)."
        ) from exc
    finally:
        model.zero_grad(set_to_none=True)
        del graph
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 2**30
        free, total = torch.cuda.mem_get_info()
        print(
            f"  preflight: OK, peak {peak:.1f} GiB of {total / 2**30:.1f} GiB "
            f"({free / 2**30:.1f} GiB free afterwards)"
        )
        torch.cuda.reset_peak_memory_stats()


def run_final(args: argparse.Namespace) -> int:
    """The submission run: no held-out week, thresholds fitted on train rows."""
    observed_mode = resolve_anchor_observed(args)
    test_cutoff = parse_cutoff_arg(args.test_cutoff)
    days = args.days
    last_anchor = (
        parse_cutoff_arg(args.anchor_end)
        if args.anchor_end
        else cast(
            pd.Timestamp,
            cast(pd.Timestamp, test_cutoff - pd.Timedelta(days=days)).normalize(),
        )
    )

    if args.anchors == "auto":
        cutoffs = final_anchor_cutoffs(
            start=parse_cutoff_arg(args.anchor_start),
            dense_end=parse_cutoff_arg(args.anchor_dense_end),
            end=last_anchor,
            stride=args.anchor_stride_final,
        )
    else:
        cutoffs = anchor_cutoffs(
            args.anchors,
            start=parse_cutoff_arg(args.anchor_start),
            end=last_anchor,
            stride=args.anchor_stride,
        )

    # The guard now protects the graded week itself: no label may be read at or
    # after the test cutoff, and every anchor's 7-day label window ends before it.
    for cutoff in cutoffs:
        assert_before_test_week(
            cutoff,
            days=days,
            context=f"train anchor {cutoff.date()}",
            ceiling=test_cutoff,
        )

    print("=" * 78)
    print(f"HGT FINAL run (no validation) | test cutoff {test_cutoff}")
    print("=" * 78)

    print("\nLoading tables from", args.db)
    tables = load_tables(args.db)

    if args.predict_only:
        return _run_predict_only(args, tables, test_cutoff)

    print(f"  anchors ({len(cutoffs)}):", ", ".join(str(c.date()) for c in cutoffs))
    print(f"  anchor observability: {observed_mode}")
    day_flags = load_day_flags()
    positives = (
        anchor_positives(tables, cutoffs, days) if observed_mode == "hybrid" else None
    )
    if day_flags is None and observed_mode in {"hybrid", "label", "strict"}:
        print("  [warn] no raw_data/coverage_activity.csv; trusting every label cell")
    verdicts = observed_activities(
        cutoffs,
        day_flags,
        days=days,
        mode=observed_mode,
        positives=positives,
    )

    train_anchors: list[Anchor] = []
    skipped: list[str] = []
    for cutoff in cutoffs:
        flags = verdicts.loc[cutoff].to_numpy(dtype=bool)
        if int(flags.sum()) < args.anchor_min_observed:
            skipped.append(f"{cutoff.date()} (only {int(flags.sum())}/6 activities usable)")
            continue
        dataset = load_dataset(
            tables, cutoff, days, f"train {cutoff.date()}", observed_activity=flags
        )
        train_anchors.append(dataset.to_anchor())
    if args.max_anchors and len(train_anchors) > args.max_anchors:
        train_anchors = train_anchors[-args.max_anchors :]
    if skipped:
        print("\n  [skip] anchors the observability rule cannot support:")
        for line in skipped:
            print("    -", line)
    if not train_anchors:
        raise SystemExit("no usable training anchors; lower --anchor-min-observed")

    train_rows = sum(anchor.n_rows for anchor in train_anchors)
    train_positives = np.sum([anchor.positives for anchor in train_anchors], axis=0)
    train_observed = np.sum(
        [anchor.observed[anchor.mask].sum(axis=0) for anchor in train_anchors], axis=0
    )

    print("\nAnchors:")
    for anchor in train_anchors:
        share = anchor.n_observed / max(anchor.n_rows * len(ACTIVITY_NAMES), 1)
        dropped = [
            act
            for act, flag in zip(
                ACTIVITY_NAMES,
                np.asarray(anchor.observed[anchor.mask].any(axis=0)).reshape(-1).tolist(),
                strict=True,
            )
            if not flag
        ]
        print(
            f"  {anchor.cutoff.date()}  rows {anchor.n_rows:>6,}  "
            f"observed {anchor.n_observed:>7,} ({share * 100:5.1f}%)  "
            f"graph {anchor.graph_megabytes:6.1f} MB"
            + (f"  dropped: {', '.join(dropped)}" if dropped else "")
        )
    total_mb = sum(anchor.graph_megabytes for anchor in train_anchors)
    print(f"  total labelled rows {train_rows:,} across {len(train_anchors)} anchors")
    print(f"  resident graph memory: {total_mb / 1024:.2f} GB (all anchors kept on CPU)")

    pos_weights = resolve_pos_weights(
        args.pos_weights,
        np.asarray(train_positives),
        int(train_rows),
        observed_counts=np.asarray(train_observed),
    )
    print(f"  pos_weight mode: {args.pos_weights}")
    print(
        "  pos_weight:",
        {act: round(w, 2) for act, w in zip(ACTIVITY_NAMES, pos_weights, strict=True)},
    )
    print(
        "  train base rates:",
        {
            act: f"{p / max(n, 1) * 100:.1f}%"
            for act, p, n in zip(
                ACTIVITY_NAMES, train_positives, train_observed, strict=True
            )
        },
    )

    checkpoint = args.checkpoint or f"checkpoints/hgt_final_{last_anchor.date()}.pt"

    calibration_anchors = train_anchors
    if args.threshold_lookback_days > 0:
        floor = cast(
            pd.Timestamp,
            train_anchors[-1].cutoff - pd.Timedelta(days=args.threshold_lookback_days),
        )
        calibration_anchors = [a for a in train_anchors if a.cutoff >= floor]
        if not calibration_anchors:
            calibration_anchors = train_anchors[-1:]
        print(
            f"\n  threshold calibration window: {floor.date()} .. "
            f"{train_anchors[-1].cutoff.date()} "
            f"({len(calibration_anchors)} anchors: "
            f"{', '.join(str(a.cutoff.date()) for a in calibration_anchors)})"
        )

    if args.dry_run:
        print(f"\n--dry-run: {len(train_anchors)} anchors built, not training.")
        print(f"  checkpoint would go to {checkpoint}")
        return 0

    if args.wandb_mode == "online" and not (
        os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").exists()
    ):
        print("\n  [warn] no WANDB_API_KEY and no ~/.netrc: wandb will fail to authenticate")
        print("         export WANDB_API_KEY=... or use --wandb-mode offline")

    in_dims = {node_type: int(graph_shape) for node_type, graph_shape in {
        node_type: train_anchors[-1].graph[node_type].x.shape[1]
        for node_type in train_anchors[-1].graph.node_types
    }.items()}
    print("\nNode feature dims:", in_dims)

    model = GitHubHeteroHGT(
        metadata=train_anchors[-1].graph.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        head_hidden_dim=args.head_hidden_dim,
        num_classes=len(ACTIVITY_NAMES),
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_gnn_layers=args.num_gnn_layers,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    preflight_largest_anchor(model, train_anchors, args.device)

    run_name = args.run_name or f"hgt-final-{last_anchor.date()}-s{args.seed}"
    started = time.time()
    try:
        model, thresholds, fit_metrics = train_model_final(
            model=model,
            train_anchors=train_anchors,
            pos_weights=pos_weights,
            epochs=args.epochs,
            lr=args.lr,
            lr_schedule=args.lr_schedule,
            lr_t0=args.lr_t0,
            lr_t_mult=args.lr_t_mult,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            project_name=args.project,
            run_name=run_name,
            device=args.device,
            wandb_mode=args.wandb_mode,
            save_model_path=checkpoint,
            checkpoint_every=args.checkpoint_every,
            shuffle_anchors=not args.no_anchor_shuffle,
            anchor_seed=args.seed,
            threshold_anchors=calibration_anchors,
            extra_config={
                "mode": "final",
                "train_cutoffs": [str(a.cutoff.date()) for a in train_anchors],
                "test_cutoff": str(test_cutoff),
                "label_window_days": args.days,
                "anchor_observed_mode": observed_mode,
                "n_train_rows": int(train_rows),
                "n_observed_cells": int(sum(a.n_observed for a in train_anchors)),
                "train_positives": [int(p) for p in train_positives],
                "train_observed_cells": [int(n) for n in train_observed],
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "head_hidden_dim": args.head_hidden_dim,
                "num_heads": args.num_heads,
                "num_gnn_layers": args.num_gnn_layers,
                "dropout": args.dropout,
                "seed": args.seed,
                "git_revision": git_revision(),
            },
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        try:
            import wandb

            wandb.finish(exit_code=130)
        except Exception as exc:  # noqa: BLE001 - shutdown must never mask the interrupt
            print(f"  [warn] wandb shutdown failed: {exc}")
        return 130

    print(f"\nTraining finished in {(time.time() - started) / 60:.1f} min")
    print("Thresholds fitted on the training anchors (no validation week exists):")
    fit_f1 = cast(dict[str, float], fit_metrics.get("f1", {}))
    for act in ACTIVITY_NAMES:
        print(
            f"  {act:<20} theta {thresholds[ACTIVITY_NAMES.index(act)]:.3f}  "
            f"F1(fit) {fit_f1.get(act, float('nan')) * 100:5.2f}%"
        )

    print(f"\nBuilding the test-week graph at {test_cutoff} ...")
    graph, users = build_inference_graph(tables, test_cutoff)
    result = predict_test_week(
        model=model,
        graph=graph,
        users=users,
        thresholds=thresholds,
        test_input=args.test_input,
        out_path=args.submission_out,
        probs_path=str(
            Path("splits") / f"test_probs_{test_cutoff.strftime('%Y-%m-%d-%H%M')}.csv"
        ),
        device=args.device,
    )
    print("\n" + "=" * 78)
    print(f"Submission written to {args.submission_out}")
    print("=" * 78)
    return result


def _run_predict_only(
    args: argparse.Namespace, tables: dict[str, pd.DataFrame], test_cutoff: pd.Timestamp
) -> int:
    """Rebuild the test graph and re-predict from a finished checkpoint.

    This exists because the feature window is a moving target: the corpus ends
    2026-11-21 06:00 UTC while the graded week starts 2026-11-21 16:00 UTC. If
    more hours are downloaded and preprocessed, the predictions can be refreshed
    in minutes without retraining.
    """
    checkpoint = Path(args.checkpoint or "checkpoints/hgt_final_2026-11-14.pt")
    if not checkpoint.exists():
        raise SystemExit(f"--predict-only needs a checkpoint, {checkpoint} does not exist")

    print(f"\nLoading checkpoint {checkpoint} ...")
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = blob.get("config", {}) or {}
    thresholds = np.asarray(blob.get("thresholds", []), dtype=np.float64).reshape(-1)
    if thresholds.size != len(ACTIVITY_NAMES) or not np.all(np.isfinite(thresholds)):
        raise SystemExit(
            f"{checkpoint} does not carry usable thresholds ({thresholds}); "
            "retrain (a checkpoint written mid-run has NaN thresholds)"
        )

    print(f"Building the test-week graph at {test_cutoff} ...")
    graph, users = build_inference_graph(tables, test_cutoff)
    in_dims = {node_type: int(graph[node_type].x.shape[1]) for node_type in graph.node_types}
    model = GitHubHeteroHGT(
        metadata=graph.metadata(),
        in_dims=in_dims,
        hidden_dim=int(config.get("hidden_dim", args.hidden_dim)),
        latent_dim=int(config.get("latent_dim", args.latent_dim)),
        head_hidden_dim=int(config.get("head_hidden_dim", args.head_hidden_dim)),
        num_classes=len(ACTIVITY_NAMES),
        num_heads=int(config.get("num_heads", args.num_heads)),
        dropout=float(config.get("dropout", args.dropout)),
        num_gnn_layers=int(config.get("num_gnn_layers", args.num_gnn_layers)),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(args.device).eval()

    return predict_test_week(
        model=model,
        graph=graph,
        users=users,
        thresholds=thresholds,
        test_input=args.test_input,
        out_path=args.submission_out,
        probs_path=str(
            Path("splits") / f"test_probs_{test_cutoff.strftime('%Y-%m-%d-%H%M')}.csv"
        ),
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if resolve_mode(args) == "final":
        set_seed(args.seed)
        return run_final(args)
    return run_validation(args)


def run_validation(argv_or_args: argparse.Namespace | list[str] | None = None) -> int:
    args = (
        argv_or_args
        if isinstance(argv_or_args, argparse.Namespace)
        else parse_args(argv_or_args)
    )
    set_seed(args.seed)

    val_cutoff = parse_cutoff_arg(args.val_cutoff)
    train_cutoff = (
        parse_cutoff_arg(args.cutoff)
        if args.cutoff
        else cast(pd.Timestamp, val_cutoff - pd.Timedelta(days=args.days))
    )

    print("=" * 78)
    multi_anchor = args.anchors != "none" and args.val_mode == "temporal"
    last_anchor = parse_cutoff_arg(args.anchor_end) if args.anchor_end else train_cutoff
    if multi_anchor:
        cutoffs = anchor_cutoffs(
            args.anchors,
            start=parse_cutoff_arg(args.anchor_start),
            end=last_anchor,
            stride=args.anchor_stride,
        )
        label = f"{len(cutoffs)} anchors" if cutoffs else "multi-anchor"
    else:
        cutoffs = [train_cutoff]
        label = f"train cutoff {train_cutoff.date()}"
    print(f"HGT training | {label} | val cutoff {val_cutoff.date()}")
    print("=" * 78)

    # Operational guard, before any table is loaded: no validation cutoff and no
    # train anchor may reach the test-week guard, so a mistyped date can never
    # produce a split that touches the graded window.
    assert_before_test_week(val_cutoff, days=args.days, context="--val-cutoff")
    for cutoff in cutoffs:
        assert_before_test_week(
            cutoff, days=args.days, context="train cutoff"
        )

    if cutoffs and max(cutoffs) >= val_cutoff:
        print("  [WARN] some training anchors are at/after the validation cutoff.")
        print("         The validation score is then retrospective, not a forecast:")
        print("         the model has already seen data from after the week it is")
        print("         scored on. Use it for threshold/feature sanity only.")

    problems = check_window_usability(val_cutoff, args.days)
    if problems:
        print("  [warn] validation label window usability:")
        for p in problems:
            print("    -", p)
    else:
        print("  [val] 7-day label window fully usable for all 6 activities")

    print("\nLoading tables from", args.db)
    tables = load_tables(args.db)

    day_flags = load_day_flags()
    if day_flags is None:
        print("  [warn] no raw_data/coverage_activity.csv; trusting every label cell")
        verdicts = pd.DataFrame(
            True, index=pd.DatetimeIndex(cutoffs), columns=pd.Index(LABEL_COLS)
        )
    else:
        verdicts = observed_activities(
            cutoffs, day_flags, days=args.days, mode=resolve_anchor_observed(args)
        )

    val_ds = load_dataset(tables, val_cutoff, args.days, "val")
    val_mask = val_ds.mask
    val_labels = val_ds.labels[val_mask]
    val_user_ids = val_ds.user_ids
    val_graph = val_ds.graph
    model_graph = val_ds.graph

    if args.val_mode == "cohort":
        tr, va = cohort_holdout(val_ds.mask, val_ds.labels, args.val_fraction, args.seed)
        val_mask = va
        val_labels = val_ds.labels[va]
        val_user_ids = val_ds.user_ids[va[val_ds.mask]]
        train_anchors = [
            Dataset(
                "cohort",
                val_cutoff,
                val_ds.graph,
                val_ds.labels,
                val_ds.observed,
                tr,
                val_ds.user_ids,
                val_ds.labels[tr].sum(axis=0),
            ).to_anchor()
        ]
        train_rows = int(tr.sum())
        train_positives = np.asarray(val_ds.labels[tr].sum(axis=0))
        train_observed = np.asarray(val_ds.observed[tr].sum(axis=0))
        print(
            f"  [cohort] {train_rows:,} train rows / {int(va.sum()):,} val rows "
            "on the same graph (validation shares the week with training)"
        )
    else:
        train_anchors = []
        skipped: list[str] = []
        for cutoff in cutoffs:
            flags = verdicts.loc[cutoff].to_numpy(dtype=bool)
            if int(flags.sum()) < args.anchor_min_observed:
                skipped.append(
                    f"{cutoff.date()} (only {int(flags.sum())}/6 activities usable)"
                )
                continue
            ds = load_dataset(
                tables, cutoff, args.days, f"train {cutoff.date()}", observed_activity=flags
            )
            train_anchors.append(ds.to_anchor())

        if args.max_anchors and len(train_anchors) > args.max_anchors:
            train_anchors = train_anchors[-args.max_anchors :]
        if skipped:
            print("\n  [skip] anchors the coverage audit cannot support:")
            for line in skipped:
                print("    -", line)
        if not train_anchors:
            raise SystemExit("no usable training anchors; lower --anchor-min-observed")
        train_rows = sum(a.n_rows for a in train_anchors)
        train_positives = np.sum([a.positives for a in train_anchors], axis=0)
        # Per-activity denominator for `--pos-weights auto`: how many label cells
        # the coverage audit says are actually observable, so a masked cell is
        # not silently counted as a negative.
        train_observed = np.sum(
            [a.observed[a.mask].sum(axis=0) for a in train_anchors], axis=0
        )

    print("\nAnchors:")
    for a in train_anchors:
        obs_share = a.n_observed / max(a.n_rows * len(ACTIVITY_NAMES), 1)
        print(
            f"  {a.cutoff.date()}  rows {a.n_rows:>6,}  observed cells "
            f"{a.n_observed:>7,} ({obs_share * 100:5.1f}%)  graph {a.graph_megabytes:6.1f} MB"
        )
    print(f"  total labelled rows {train_rows:,} across {len(train_anchors)} anchors")
    total_graph_mb = sum(a.graph_megabytes for a in train_anchors) + val_ds.graph_megabytes
    print(f"  resident graph memory: {total_graph_mb / 1024:.2f} GB (all anchors kept on CPU)")

    pos_weights = resolve_pos_weights(
        args.pos_weights,
        np.asarray(train_positives),
        int(train_rows),
        observed_counts=np.asarray(train_observed),
    )
    print(f"  pos_weight mode: {args.pos_weights}")
    if args.pos_weights == "auto":
        print(
            "  pos_weight denominators (observed cells):",
            {
                a: int(n)
                for a, n in zip(ACTIVITY_NAMES, np.asarray(train_observed), strict=True)
            },
        )
    print("  pos_weight:", {a: round(w, 2) for a, w in zip(ACTIVITY_NAMES, pos_weights, strict=True)})

    if args.dry_run:
        print("\n--dry-run: splits and graphs are valid; not training.")
        return 0

    if args.wandb_mode == "online" and not (
        os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").exists()
    ):
        print("\n  [warn] no WANDB_API_KEY and no ~/.netrc: wandb will fail to authenticate")
        print("         export WANDB_API_KEY=... or use --wandb-mode offline")

    in_dims = {nt: int(model_graph[nt].x.shape[1]) for nt in model_graph.node_types}
    print("\nNode feature dims:", in_dims)

    model = GitHubHeteroHGT(
        metadata=model_graph.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        head_hidden_dim=args.head_hidden_dim,
        num_classes=len(ACTIVITY_NAMES),
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_gnn_layers=args.num_gnn_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # The TA evaluate.py needs the truth on disk; write the split's own answer
    # file so the check never silently reads a stale splits/answer.csv.
    truth_path = Path("splits") / f"val_truth_{val_cutoff.date()}.csv"
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "UserID": val_user_ids,
            **{
                act: val_labels[:, i]
                for i, act in enumerate(ACTIVITY_NAMES)
            },
        }
    ).to_csv(truth_path, index=False)

    run_name = args.run_name or f"hgt-{val_cutoff.date()}-s{args.seed}"
    checkpoint = args.checkpoint or f"checkpoints/hgt_{val_cutoff.date()}.pt"

    try:
        common = {
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "lr_t0": args.lr_t0,
            "lr_t_mult": args.lr_t_mult,
            "weight_decay": args.weight_decay,
            "val_interval": args.val_interval,
            "grad_clip": args.grad_clip,
            "cma_sigma0": args.cma_sigma0,
            "cma_popsize": args.cma_popsize,
            "cma_maxfevals": args.cma_maxfevals,
            "project_name": args.project,
            "run_name": run_name,
            "device": args.device,
            "wandb_mode": args.wandb_mode,
            "truth_path": str(truth_path),
            "val_pred_path": args.val_pred_path
            or f"splits/val_best_pred_{val_cutoff.date()}.csv",
            "save_model_path": checkpoint,
            "extra_config": {
                "val_mode": args.val_mode,
                "train_cutoffs": [str(a.cutoff.date()) for a in train_anchors],
                "val_cutoff": str(val_cutoff.date()),
                "label_window_days": args.days,
                "anchor_observed_mode": resolve_anchor_observed(args),
                "n_train_rows": int(train_rows),
                "n_val_rows": int(val_mask.sum()),
                "n_observed_cells": int(sum(a.n_observed for a in train_anchors)),
                "train_positives": [int(p) for p in train_positives],
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "head_hidden_dim": args.head_hidden_dim,
                "num_heads": args.num_heads,
                "num_gnn_layers": args.num_gnn_layers,
                "dropout": args.dropout,
                "seed": args.seed,
                "git_revision": git_revision(),
                "n_parameters": int(n_params),
                "n_users_nodes": int(val_ds.graph["Users"].num_nodes),
            },
        }
        if multi_anchor:
            model, thresholds, metrics = train_model_multi_anchor(
                model=model,
                train_anchors=train_anchors,
                val_anchor=val_ds.to_anchor(),
                val_user_ids=val_user_ids,
                pos_weights=pos_weights,
                shuffle_anchors=not args.no_anchor_shuffle,
                anchor_seed=args.seed,
                **common,
            )
        else:
            single = train_anchors[-1]
            model, thresholds, metrics = train_model(
                model=model,
                train_graph=single.graph,
                train_labels=torch.from_numpy(single.labels),
                val_graph=val_graph,
                val_labels=val_labels,
                val_user_ids=val_user_ids,
                pos_weights=pos_weights,
                cohort_mask_train=torch.from_numpy(single.mask),
                cohort_mask_val=torch.from_numpy(val_mask),
                **common,
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        try:
            import wandb

            wandb.finish(exit_code=130)
        except Exception as exc:  # noqa: BLE001 - shutdown must never mask the interrupt
            print(f"  [warn] wandb shutdown failed: {exc}")
        return 130

    raw_score = metrics.get("score", 0.0)
    best_score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
    print("\n" + "=" * 78)
    print(f"Best validation score: {best_score * 100:.2f}%")
    print("Per-activity F1 and learned threshold:")
    per_event = metrics.get("per_event", {})
    per_event = cast(dict[str, float], per_event) if isinstance(per_event, dict) else {}
    for i, act in enumerate(ACTIVITY_NAMES):
        f1 = float(per_event.get(act, float("nan")))
        print(f"  {act:<20} F1 {f1 * 100:6.2f}%   threshold {thresholds[i]:.3f}")
    print("\nReference scores (README, 2026-09-27..10-04):")
    for name, value in BASELINES.items():
        print(f"  {name:<20} {value * 100:6.2f}%")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
