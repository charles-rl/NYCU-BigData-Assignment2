"""Shared driver COPY for the ablation suite.

This file is a copy of ``scripts/run_training.py``.  The base driver is left
untouched; every ablation entry point in this folder imports this copy.  The
data plumbing, leakage checks, anchor handling, and training calls are the same
as the base pipeline.  The differences are:

* ``variant`` selects exactly one factor to ablate (see ``VARIANTS``);
* the copied graph builder attaches temporal edge weights (or unit weights);
* the copied model can drop the raw-user residual fusion;
* W&B runs carry the ``github-activity-ablations`` group and variant/seed tags;
* every run writes a machine-readable ``result.json`` with Macro-F1, Micro-F1,
  Score @0.5, and Score @CMA-ES for ``run_all_ablations.py``.

``--dry-run`` here means a real 1-epoch smoke test (force one anchor, tiny
CMA-ES budget, CPU and offline W&B when not overridden), not the base script's
"build splits only" mode.  That is deliberate: it verifies imports, parameter
shapes, the loss, and the metric calculation end to end without launching a
full training loop.

Run from the repository root::

    python -m experiments.ablations.train_no_text --dry-run --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.data.label_coverage import load_day_flags, observed_activities
from src.data.make_split import (
    assert_before_test_week,
    build_split,
    leakage_failures,
)
from src.data.split_io import DEFAULT_DB, load_tables, parse_cutoff
from src.data.temporal_split import LABEL_COLS
from src.utils.metrics import score_in_memory

from .build_graph_ablation import (
    ACTIVITY_BY_EVENT_TABLE,
    ORIGINAL_FEATURE_DIMS,
    TEMPORAL_EDGE_TAU_DAYS,
    TEXT_FEATURE_INDICES,
    build_hetero_graph,
    check_edge_weights,
    edge_weight_stats,
    mask_text_features,
    text_feature_sums,
)
from .hetero_model_ablation import GitHubHeteroHGT
from .train_engine import (
    ACTIVITY_NAMES,
    Anchor,
    train_model,
    train_model_multi_anchor,
)

# Measured on this corpus (docs/VALIDATION_DATA.md section 5).
DOC_POS_WEIGHTS: list[float] = [7.28, 18.08, 11.05, 12.83, 19.29, 1.32]
DOC_TAU_DAYS: list[float] = [5.96, 5.37, 6.02, 6.73, 6.91, 7.50]

ABLATION_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ABLATION_ROOT / "outputs"
DEFAULT_GROUP = "github-activity-ablations"
DEPTH_GROUP = "github-activity-depth-study"


@dataclass(frozen=True)
class VariantSpec:
    """One arm of the suite.

    The three ``full``/``no-*`` arms change exactly one model factor.  The
    ``depth-*`` arms keep the full architecture and change only the number of
    HGT layers, so they answer a separate question with the same data and
    hyperparameters.
    """

    name: str
    tag: str
    mask_text: bool
    edge_decay: bool
    residual_fusion: bool
    description: str
    wandb_group: str = DEFAULT_GROUP
    num_layers: int | None = None
    extra_tags: tuple[str, ...] = ()
    # ``None`` means "whatever --pos-weights says" (a plain ``full`` run obeys
    # the CLI).  An arm that *is* about calibration pins its own mode here, so
    # the difference from `full` stays inside the driver and cannot be lost by
    # forgetting a flag on the command line.
    pos_weights: str | None = None


VARIANTS: dict[str, VariantSpec] = {
    "full": VariantSpec(
        name="full",
        tag="full-model",
        mask_text=False,
        edge_decay=True,
        residual_fusion=True,
        description="Control: temporal edge decay + raw-user residual fusion + 6 text features.",
    ),
    "no-text": VariantSpec(
        name="no-text",
        tag="no-text",
        mask_text=True,
        edge_decay=True,
        residual_fusion=True,
        description="Mask the 6 structural/textual node features (zero or drop).",
    ),
    "no-temporal-decay": VariantSpec(
        name="no-temporal-decay",
        tag="no-temporal-decay",
        mask_text=False,
        edge_decay=False,
        residual_fusion=True,
        description="Set every temporal edge weight to 1.0 (no exponential decay).",
    ),
    "no-residual-fusion": VariantSpec(
        name="no-residual-fusion",
        tag="no-residual-fusion",
        mask_text=False,
        edge_decay=True,
        residual_fusion=False,
        description="Remove the raw Users.x concatenation from the prediction heads.",
    ),
    "pos-weights-auto": VariantSpec(
        name="pos-weights-auto",
        tag="pos-weights-auto",
        mask_text=False,
        edge_decay=True,
        residual_fusion=True,
        description=(
            "Calibration study: pos_weight = (N_observed - N_pos) / N_pos per "
            "activity, measured on this run's training anchors, instead of the "
            "static 252-anchor constants in docs/VALIDATION_DATA.md section 5. "
            "Same graph, same 21 node features, same architecture."
        ),
        pos_weights="auto",
        extra_tags=("calibration", "pos-weights"),
    ),
    "depth-1layer": VariantSpec(
        name="depth-1layer",
        tag="depth-1layer",
        mask_text=False,
        edge_decay=True,
        residual_fusion=True,
        description="Depth study control: exactly the base 1-layer HGT architecture.",
        wandb_group=DEPTH_GROUP,
        num_layers=1,
        extra_tags=("depth-study", "1-layer"),
    ),
    "depth-2layer": VariantSpec(
        name="depth-2layer",
        tag="depth-2layer",
        mask_text=False,
        edge_decay=True,
        residual_fusion=True,
        description="Depth study treatment: the same architecture with 2 HGT layers.",
        wandb_group=DEPTH_GROUP,
        num_layers=2,
        extra_tags=("depth-study", "2-layer"),
    ),
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
    ap.add_argument("--epochs", type=int, default=310)
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
    ap.add_argument("--num-gnn-layers", type=int, default=1)
    ap.add_argument(
        "--num_layers",
        "--num-layers",
        dest="num_layers_cli",
        type=int,
        choices=(1, 2),
        default=None,
        help=(
            "depth study only (train_gnn_depth.py): 1 or 2 HGT layers. "
            "Selects the depth-<n>layer variant; ignored by the other arms."
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
    ap.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="default: online for full runs, offline for --dry-run",
    )
    ap.add_argument(
        "--wandb-group",
        default=None,
        help="default: variant-specific (github-activity-ablations or -depth-study)",
    )
    ap.add_argument(
        "--wandb-tags",
        default="",
        help="extra comma-separated W&B tags (variant/seed/dry-run are added)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="per-run artifact directory (predictions, checkpoint, result.json)",
    )
    ap.add_argument(
        "--results-json",
        default=None,
        help="result JSON path (default: <output-dir>/<variant>_seed<seed>/result.json)",
    )
    ap.add_argument(
        "--text-mode",
        choices=("zero", "drop"),
        default="zero",
        help="no-text variant: zero the 6 columns (same width) or drop them",
    )
    ap.add_argument(
        "--decay-max-days",
        type=float,
        default=30.0,
        help="clip horizon for edge-age dt before exp(-dt/tau); 0 disables clipping",
    )
    ap.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="torch CPU threads (0 = PyTorch default); useful on a shared server",
    )
    ap.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help=(
            "recompute each HGT block in backward (non-reentrant, so dropout "
            "masks are unchanged) instead of keeping its per-edge activations. "
            "Costs ~30%% slower epochs and frees several GiB; only the "
            "depth-2layer arm needs it on a 24 GB card"
        ),
    )
    ap.add_argument(
        "--keep-val-graph-on-host",
        action="store_true",
        help=(
            "park the validation graph on the host between validation passes so "
            "it does not compete with a train anchor graph for device memory. "
            "Placement only - the reported numbers are identical"
        ),
    )
    ap.add_argument(
        "--device",
        default=None,
        help="default: cuda if available, else cpu; --dry-run defaults to cpu",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pos-weights",
        default="doc",
        help="'doc' (measured constants), 'auto' (from train prevalence), or 6 comma floats",
    )
    ap.add_argument("--checkpoint", default=None, help="path for the best model state")
    ap.add_argument(
        "--val-pred-path",
        default=None,
        help="where to write validation predictions (default: <run-dir>/val_pred.csv)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "1-epoch smoke test: force epochs=1, one anchor, tiny CMA-ES, "
            "CPU and offline W&B unless --device/--wandb-mode are given"
        ),
    )
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


def verify_text_variant(
    graph: HeteroData,
    spec: VariantSpec,
    text_mode: str,
    *,
    require_nonzero: bool,
) -> None:
    """Check that the text-feature ablation actually changed the feature matrices."""
    sums = {} if spec.mask_text else text_feature_sums(graph)
    for node_type, indices in TEXT_FEATURE_INDICES.items():
        x = graph[node_type].x
        if spec.mask_text:
            if text_mode == "zero":
                total = float(x[:, list(indices)].abs().sum())
                if total != 0.0:
                    raise RuntimeError(
                        f"no-text: {node_type} text columns are not zero ({total})"
                    )
            else:
                expected = ORIGINAL_FEATURE_DIMS[node_type] - len(indices)
                if x.shape[1] != expected:
                    raise RuntimeError(
                        f"no-text drop: {node_type}.x has {x.shape[1]} columns, "
                        f"expected {expected}"
                    )
        else:
            expected = ORIGINAL_FEATURE_DIMS[node_type]
            if x.shape[1] != expected:
                raise RuntimeError(
                    f"{node_type}.x has {x.shape[1]} columns, expected {expected}"
                )
            if require_nonzero:
                zeros = [
                    name for name, value in sums[node_type].items() if value <= 0.0
                ]
                if zeros:
                    print(
                        f"  [warn] {node_type}: text features with no signal "
                        f"on this graph: {zeros}"
                    )


def load_dataset(
    tables: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    days: int,
    name: str,
    observed_activity: np.ndarray | None = None,
    *,
    variant: str = "full",
    text_mode: str = "zero",
    decay_max_days: float | None = 30.0,
) -> Dataset:
    """Build one leakage-checked split and its variant graph.

    ``observed_activity`` is a `(6,)` bool vector from the coverage audit.  An
    activity the audit could not measure gets its label column zeroed *and*
    marked unobserved, so the loss skips it rather than reading a false "0".

    ``variant`` / ``text_mode`` / ``decay_max_days`` are applied to every graph
    (training anchors and validation) so the train and validation inputs carry
    exactly the same ablation.
    """
    spec = VARIANTS[variant]
    started = time.time()
    split = build_split(tables, cutoff, days)
    failures = leakage_failures(tables, split)
    if failures:
        joined = "\n  - ".join(failures)
        raise SystemExit(f"leakage check FAILED for {cutoff.date()}:\n  - {joined}")

    graph = build_hetero_graph(
        split.features,
        cutoff,
        temporal_decay=spec.edge_decay,
        max_decay_days=decay_max_days,
    )
    check_edge_weights(graph, expect_decay=spec.edge_decay)
    if spec.mask_text:
        mask_text_features(graph, mode=text_mode)
    verify_text_variant(graph, spec, text_mode, require_nonzero=(name == "val"))
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
        fitted on a different, wider panel than any single run trains on.
    ``auto``
        ``(N_observed - N_pos) / N_pos`` per activity, computed from the
        training anchors of *this* run.  The denominator is the number of
        **observed** cells for that activity, not the row count: a cell the
        coverage audit marked unobserved never enters the loss, so counting it
        as a negative would deflate the weight for exactly the activities that
        lost days.  Mathematically identical to ``(1 - p) / p`` with
        ``p = N_pos / N_observed``, which is the usual BCE ``pos_weight``.
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


def _print_pos_weight_report(
    mode: str,
    weights: list[float],
    positives: np.ndarray,
    observed: np.ndarray,
) -> None:
    """Print the calibration inputs and the weights they produced."""
    print(f"  pos_weight mode: {mode}")
    print(
        "  observed cells:",
        {a: int(c) for a, c in zip(ACTIVITY_NAMES, observed, strict=True)},
    )
    print(
        "  positives:",
        {a: int(p) for a, p in zip(ACTIVITY_NAMES, positives, strict=True)},
    )
    print(
        "  pos_weight:",
        {a: round(w, 2) for a, w in zip(ACTIVITY_NAMES, weights, strict=True)},
    )


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


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/path values into JSON-serialisable objects."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_result_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one run's machine-readable result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def apply_dry_run_defaults(args: argparse.Namespace) -> None:
    """Make ``--dry-run`` a real but tiny end-to-end smoke test."""
    print("[dry-run] 1-epoch smoke profile (no full training loop):")
    if args.epochs != 1:
        print(f"  epochs {args.epochs} -> 1")
    args.epochs = 1
    if args.val_interval != 1:
        print(f"  val_interval {args.val_interval} -> 1")
    args.val_interval = 1
    if args.max_anchors == 0:
        args.max_anchors = 1
        print("  max_anchors all -> 1")
    if args.cma_maxfevals == 0 or args.cma_maxfevals > 100:
        print(f"  cma_maxfevals {args.cma_maxfevals} -> 100")
        args.cma_maxfevals = 100
    if args.cma_popsize > 8:
        print(f"  cma_popsize {args.cma_popsize} -> 8")
        args.cma_popsize = 8
    if args.device is None:
        args.device = "cpu"
        print("  device -> cpu (shared-GPU protection; pass --device to override)")
    if args.wandb_mode is None:
        args.wandb_mode = "offline"
        print("  wandb_mode -> offline")


def resolve_device(device: str | None) -> str:
    """Resolve and validate the torch device string."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {device} requested but CUDA is not available")
    return device


def resolve_wandb_mode(args: argparse.Namespace) -> str:
    if args.wandb_mode is not None:
        return str(args.wandb_mode)
    return "offline" if args.dry_run else "online"


def build_wandb_tags(spec: VariantSpec, args: argparse.Namespace) -> list[str]:
    """Variant, seed, and dry-run tags, plus any user-supplied tags."""
    tags = [spec.tag, f"seed-{args.seed}", "ablation"]
    for tag in spec.extra_tags:
        if tag not in tags:
            tags.append(tag)
    if args.dry_run:
        tags.append("dry-run")
    for raw in str(args.wandb_tags).split(","):
        tag = raw.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def print_banner(
    spec: VariantSpec, args: argparse.Namespace, tags: list[str]
) -> None:
    print("=" * 78)
    print(
        f"ABLATION {spec.name} | seed {args.seed} | epochs {args.epochs} | "
        f"device {args.device} | dry_run={args.dry_run}"
    )
    print(f"  {spec.description}")
    print(
        f"  text masked={spec.mask_text} (mode={args.text_mode}) | "
        f"temporal decay={spec.edge_decay} | "
        f"residual fusion={spec.residual_fusion} | "
        f"gnn layers={args.num_gnn_layers}"
    )
    print(
        f"  memory mode: grad_checkpoint={args.grad_checkpoint} | "
        f"val_graph_on_host={args.keep_val_graph_on_host}"
    )
    print(
        f"  W&B group={args.wandb_group} | tags={tags} | "
        f"mode={args.wandb_mode}"
    )
    print("=" * 78)


def print_edge_weight_summary(graph: HeteroData, spec: VariantSpec) -> None:
    """One compact line per activity showing the weights actually in the graph."""
    stats = edge_weight_stats(graph)
    state = "ON  exp(-min(dt,max_days)/tau)" if spec.edge_decay else "OFF all weights = 1.0"
    print(f"  [ablation] temporal edge decay: {state}")
    for src_table, activity in ACTIVITY_BY_EVENT_TABLE.items():
        rows = []
        for edge_type in graph.edge_types:
            if edge_type[0] != src_table:
                continue
            stat = stats["__".join(edge_type)]
            if int(stat["count"]) > 0:
                rows.append(stat)
        if not rows:
            continue
        count = sum(int(row["count"]) for row in rows)
        mean = sum(float(row["mean"]) * int(row["count"]) for row in rows) / count
        low = min(float(row["min"]) for row in rows)
        high = max(float(row["max"]) for row in rows)
        print(
            f"    {activity:<18} edges {count:>9,}  "
            f"mean w {mean:.3f}  min {low:.3f}  max {high:.3f}"
        )


def print_text_feature_summary(
    graph: HeteroData, spec: VariantSpec, text_mode: str
) -> None:
    if spec.mask_text:
        print(f"  [ablation] text features: MASKED (mode={text_mode})")
        for node_type, indices in TEXT_FEATURE_INDICES.items():
            print(
                f"    {node_type:<16} feature dim {graph[node_type].x.shape[1]:>2}  "
                f"indices {list(indices)}"
            )
        return

    sums = text_feature_sums(graph)
    parts = [
        f"{name}={value:.0f}"
        for node_type in sums
        for name, value in sums[node_type].items()
    ]
    print("  [ablation] text features: PRESENT  " + "  ".join(parts))


def collect_metrics(
    run_dir: Path, thresholds: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score @0.5 and @CMA-ES from the arrays the trainer wrote for this run."""
    probs_file = run_dir / "val_probs.npy"
    labels_file = run_dir / "val_labels.npy"
    if not probs_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"trainer did not write {probs_file.name}/{labels_file.name} in {run_dir}"
        )
    probs = np.load(probs_file)
    labels = np.load(labels_file)
    if probs.shape != labels.shape:
        raise ValueError(f"prediction shape {probs.shape} != label shape {labels.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("validation probabilities contain non-finite values")

    theta = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if theta.size != probs.shape[1]:
        raise ValueError(
            f"expected {probs.shape[1]} thresholds, got {theta.size}"
        )

    default = score_in_memory(labels, (probs >= 0.5).astype(np.int64))
    tuned = score_in_memory(
        labels, (probs >= theta[None, :]).astype(np.int64)
    )
    return dict(default), dict(tuned)


def main(argv: list[str] | None = None, variant: str = "full") -> int:
    if variant not in VARIANTS:
        raise SystemExit(
            f"unknown ablation variant {variant!r}; choose from {sorted(VARIANTS)}"
        )
    spec = VARIANTS[variant]
    args = parse_args(argv)

    # Depth-study argument handling.  The base pipeline's --num-gnn-layers is
    # pinned to 1 for every ablation arm so the only difference between an arm
    # and `full` is the ablated component.  The depth arms opt in explicitly.
    if args.num_layers_cli is not None:
        if spec.num_layers is None:
            raise SystemExit(
                "--num_layers/-num-layers is only valid with train_gnn_depth.py"
            )
        if args.num_layers_cli != spec.num_layers:
            raise SystemExit(
                f"variant {variant} requires --num_layers {spec.num_layers}, "
                f"got {args.num_layers_cli}"
            )
        if args.num_gnn_layers != args.num_layers_cli:
            print(
                f"  [warn] --num_layers {args.num_layers_cli} overrides "
                f"--num-gnn-layers {args.num_gnn_layers}"
            )
        args.num_gnn_layers = args.num_layers_cli
    if spec.num_layers is not None:
        if args.num_gnn_layers != spec.num_layers:
            raise SystemExit(
                f"variant {variant} requires {spec.num_layers} GNN layers, "
                f"got {args.num_gnn_layers}"
            )
    elif args.num_gnn_layers != 1:
        raise SystemExit(
            "ablation arms must keep the base --num-gnn-layers 1; "
            "use train_gnn_depth.py for the 1-vs-2 layer study"
        )

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    if args.dry_run:
        apply_dry_run_defaults(args)
    args.device = resolve_device(args.device)
    args.wandb_mode = resolve_wandb_mode(args)
    args.wandb_group = args.wandb_group or spec.wandb_group
    tags = build_wandb_tags(spec, args)

    started = time.time()
    set_seed(args.seed)
    print_banner(spec, args, tags)

    run_slug = f"{variant}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    results_json = (
        Path(args.results_json) if args.results_json else run_dir / "result.json"
    )
    print(f"  run directory: {run_dir}")
    print(f"  result JSON:   {results_json}")

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
    print(f"ABLATION {variant} | {label} | val cutoff {val_cutoff.date()}")
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
            cutoffs, day_flags, days=args.days, mode=args.anchor_observed
        )

    # A smoke test must not build every historical anchor just to throw all but
    # one away.  Full runs keep the base driver's exact semantics (skip first,
    # then cap the newest usable anchors).
    if args.dry_run and args.max_anchors and len(cutoffs) > args.max_anchors:
        print(
            f"  [dry-run] building only the newest {args.max_anchors} of "
            f"{len(cutoffs)} anchors"
        )
        cutoffs = cutoffs[-args.max_anchors :]

    val_ds = load_dataset(
        tables,
        val_cutoff,
        args.days,
        "val",
        variant=variant,
        text_mode=args.text_mode,
        decay_max_days=args.decay_max_days,
    )
    val_mask = val_ds.mask
    val_labels = val_ds.labels[val_mask]
    val_user_ids = val_ds.user_ids
    val_graph = val_ds.graph
    model_graph = val_ds.graph
    print_edge_weight_summary(val_graph, spec)
    print_text_feature_summary(val_graph, spec, args.text_mode)

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
                tables,
                cutoff,
                args.days,
                f"train {cutoff.date()}",
                observed_activity=flags,
                variant=variant,
                text_mode=args.text_mode,
                decay_max_days=args.decay_max_days,
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
        # the audit says are actually observable, so a masked cell is not
        # silently counted as a negative.
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

    # The arm may pin its own calibration mode (``pos-weights-auto``); otherwise
    # the CLI flag decides, which is what keeps `full` and the other ablations on
    # the static doc constants they were measured with.
    pos_weights_spec = spec.pos_weights or args.pos_weights
    pos_weights = resolve_pos_weights(
        pos_weights_spec,
        np.asarray(train_positives),
        int(train_rows),
        observed_counts=np.asarray(train_observed),
    )
    _print_pos_weight_report(
        pos_weights_spec,
        pos_weights,
        np.asarray(train_positives),
        np.asarray(train_observed),
    )

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
        use_residual_fusion=spec.residual_fusion,
        # The no-decay arm stores unit weights; keeping the code path enabled
        # means the two arms differ only in the weight tensor, not parameters.
        use_temporal_decay=True,
        grad_checkpoint=args.grad_checkpoint,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # The TA evaluate.py needs the truth on disk; keep it inside this run's
    # directory so ablation runs never overwrite the base pipeline's splits/.
    truth_path = run_dir / "val_truth.csv"
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

    val_pred_path = (
        Path(args.val_pred_path) if args.val_pred_path else run_dir / "val_pred.csv"
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "model.pt"
    run_name = args.run_name or f"abl-{variant}-s{args.seed}"

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
            "wandb_group": args.wandb_group,
            "wandb_tags": tags,
            "truth_path": str(truth_path),
            "val_pred_path": str(val_pred_path),
            "save_model_path": str(checkpoint),
            "extra_config": {
                "val_mode": args.val_mode,
                "ablation_variant": variant,
                "ablation_tag": spec.tag,
                "mask_text_features": spec.mask_text,
                "text_mask_mode": args.text_mode if spec.mask_text else None,
                "temporal_edge_decay": spec.edge_decay,
                "temporal_edge_tau_days": (
                    dict(TEMPORAL_EDGE_TAU_DAYS) if spec.edge_decay else None
                ),
                "decay_max_days": args.decay_max_days,
                "residual_fusion": spec.residual_fusion,
                "node_feature_dims": in_dims,
                "text_feature_indices": (
                    {key: list(value) for key, value in TEXT_FEATURE_INDICES.items()}
                    if spec.mask_text
                    else None
                ),
                "wandb_group": args.wandb_group,
                "wandb_tags": tags,
                "dry_run": bool(args.dry_run),
                "num_threads": args.num_threads,
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
                "grad_checkpoint": bool(args.grad_checkpoint),
                "keep_val_graph_on_host": bool(args.keep_val_graph_on_host),
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
                # Only this trainer keeps the validation graph and a train
                # anchor graph resident together, so the host-parking switch is
                # passed here rather than through the shared `common` kwargs.
                keep_val_graph_on_host=args.keep_val_graph_on_host,
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

    default_metrics, cma_metrics = collect_metrics(run_dir, thresholds)
    trained_score = metrics.get("score")
    if isinstance(trained_score, (int, float)):
        delta = abs(float(trained_score) - float(cma_metrics["score"]))
        if delta > 1e-5:
            print(
                f"  [warn] trainer score {float(trained_score):.6f} != "
                f"re-scored {float(cma_metrics['score']):.6f} (delta {delta:.2e})"
            )

    duration = time.time() - started
    result: dict[str, Any] = {
        "status": "ok",
        "variant": variant,
        "variant_tag": spec.tag,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "epochs": args.epochs,
        "device": args.device,
        "macro_f1": float(cma_metrics["macro_f1"]),
        "micro_f1": float(cma_metrics["micro_f1"]),
        "score_default_0_5": float(default_metrics["score"]),
        "score_cmaes": float(cma_metrics["score"]),
        "macro_f1_default_0_5": float(default_metrics["macro_f1"]),
        "micro_f1_default_0_5": float(default_metrics["micro_f1"]),
        "per_activity_f1_cmaes": cma_metrics["per_event"],
        "per_activity_f1_default_0_5": default_metrics["per_event"],
        "thresholds": np.asarray(thresholds, dtype=np.float64).tolist(),
        "val_cutoff": str(val_cutoff.date()),
        "val_rows": int(val_mask.sum()),
        "train_anchors": [str(a.cutoff.date()) for a in train_anchors],
        "n_train_anchors": len(train_anchors),
        "n_parameters": int(n_params),
        "gnn_layers": int(args.num_gnn_layers),
        "grad_checkpoint": bool(args.grad_checkpoint),
        "val_graph_on_host": bool(args.keep_val_graph_on_host),
        "depth_study": spec.num_layers is not None,
        "node_feature_dims": in_dims,
        "ablation": {
            "mask_text": spec.mask_text,
            "text_mode": args.text_mode if spec.mask_text else None,
            "temporal_edge_decay": spec.edge_decay,
            "decay_max_days": args.decay_max_days,
            "residual_fusion": spec.residual_fusion,
            "pos_weights_mode": pos_weights_spec,
        },
        # Calibration provenance: which mode, what it resolved to, and the counts
        # it was derived from, so a reader can recompute `auto` by hand.
        "pos_weights_mode": pos_weights_spec,
        "pos_weights": [float(w) for w in pos_weights],
        "train_observed_cells": {
            a: int(c) for a, c in zip(ACTIVITY_NAMES, train_observed, strict=True)
        },
        "train_positive_cells": {
            a: int(p) for a, p in zip(ACTIVITY_NAMES, train_positives, strict=True)
        },
        "wandb_group": args.wandb_group,
        "wandb_tags": tags,
        "wandb_mode": args.wandb_mode,
        "wandb_run_name": run_name,
        "git_revision": git_revision(),
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": round(duration, 2),
        "result_json": str(results_json),
        "run_dir": str(run_dir),
    }
    write_result_json(results_json, result)

    print("\n" + "=" * 78)
    print(f"ABLATION RESULT | {variant} | seed {args.seed}")
    print(f"  Macro-F1        {result['macro_f1'] * 100:6.2f}%")
    print(f"  Micro-F1        {result['micro_f1'] * 100:6.2f}%")
    print(f"  Score @0.5      {result['score_default_0_5'] * 100:6.2f}%")
    print(f"  Score CMA-ES    {result['score_cmaes'] * 100:6.2f}%")
    print(f"  Thresholds      {[round(float(t), 3) for t in result['thresholds']]}")
    print(
        f"  Pos weights     {result['pos_weights_mode']} "
        f"{[round(float(w), 2) for w in result['pos_weights']]}"
    )
    print(f"  Result JSON     {results_json}")
    print(f"  Duration        {duration:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
