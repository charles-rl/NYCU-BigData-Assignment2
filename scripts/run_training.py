"""Train the HGT model end to end and stream train/val curves to W&B.

This is the driver that `src/models/train.py` was missing: it turns raw parquet
tables into a leakage-checked temporal split, builds the heterogeneous graph,
aligns the labels to graph node order, and then calls ``train_model``.

Temporal setup (from docs/VALIDATION_DATA.md):

  train: features before 2026-02-08, labels 2026-02-08 .. 2026-02-15
  val:   features before 2026-02-15, labels 2026-02-15 .. 2026-02-22

Both label windows were verified to have 7 usable days for all six activities
in ``raw_data/coverage_activity.csv``.  Features for the validation graph may
include the training label window (that is the normal "predict the next week
from everything known at the cutoff" setup); the labels themselves never
overlap.

Examples
--------
    # sanity check the data plumbing, no training
    python -m scripts.run_training --dry-run

    # short run to see the curves
    python -m scripts.run_training --epochs 3 --val-interval 1

    # full run
    python -m scripts.run_training

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
from src.data.make_split import build_split, leakage_failures
from src.data.split_io import DEFAULT_DB, load_tables, parse_cutoff
from src.data.temporal_split import LABEL_COLS
from src.features.build_graph import build_hetero_graph
from src.models.hetero_model import GitHubHeteroHGT
from src.models.train import (
    ACTIVITY_NAMES,
    Anchor,
    train_model,
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
        "--anchor-observed",
        choices=("label", "strict", "off"),
        default="label",
        help="usability rule from coverage_activity.csv for a label cell",
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
    ap.add_argument("--epochs", type=int, default=150)
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
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--head-hidden-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-gnn-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
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
        default="doc",
        help="'doc' (measured constants), 'auto' (from train prevalence), or 6 comma floats",
    )
    ap.add_argument("--checkpoint", default=None, help="path for the best model state")
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


def resolve_pos_weights(spec: str, train_positives: np.ndarray, n_rows: int) -> list[float]:
    if spec == "doc":
        return list(DOC_POS_WEIGHTS)
    if spec == "auto":
        prevalence = np.clip(train_positives / max(n_rows, 1), 1e-6, 1 - 1e-6)
        return [float(w) for w in (1.0 - prevalence) / prevalence]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != len(ACTIVITY_NAMES):
        raise SystemExit(f"--pos-weights needs {len(ACTIVITY_NAMES)} comma-separated values")
    return [float(p) for p in parts]


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
            cutoffs, day_flags, days=args.days, mode=args.anchor_observed
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
        args.pos_weights, np.asarray(train_positives), int(train_rows)
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
            "val_pred_path": f"splits/val_best_pred_{val_cutoff.date()}.csv",
            "save_model_path": checkpoint,
            "extra_config": {
                "val_mode": args.val_mode,
                "train_cutoffs": [str(a.cutoff.date()) for a in train_anchors],
                "val_cutoff": str(val_cutoff.date()),
                "label_window_days": args.days,
                "anchor_observed_mode": args.anchor_observed,
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
