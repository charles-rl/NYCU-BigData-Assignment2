"""Experiment A: LightGBM tabular baseline.

Trains six independent binary LightGBM classifiers strictly on the user
behavioral feature block that the GNN sees as ``data['Users'].x`` (21 columns:
6 activities x (days_since, has_ever, count_7d) + is_bot + is_cohort +
user_age_days).  No graph structure, no node embeddings, no text features.

Fairness contract with the GNN:

* the same temporal split, anchors, anchor filtering, and cohort mask
  (helpers imported from the copied ablation driver);
* the same calibrated positive-class weights (``--pos-weights doc``);
* thresholds fitted by CMA-ES on TRAINING rows only, then applied to
  validation (never tuned on the validation week);
* the same ``(macro-F1 + micro-F1) / 2`` scorer from ``src.utils.metrics``;
* the same validation prediction/truth CSVs, cross-checked with the official
  ``scripts/evaluate.py``.

The feature block is verified against a real ``build_hetero_graph`` output for
the validation anchor (``max|diff|`` printed; hard failure above 1e-6), so the
baseline cannot silently drift from the GNN's user features.

Usage (from the repository root)::

    python -m experiments.baselines.train_lightgbm --dry-run --device cpu \
        --seed 42 --num-threads 8

    python -m experiments.baselines.train_lightgbm --seed 42 --epochs 310
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb

from src.data.label_coverage import load_day_flags, observed_activities
from src.data.make_split import assert_before_test_week, build_split, leakage_failures
from src.data.split_io import DEFAULT_DB, load_tables
from src.data.temporal_split import LABEL_COLS
from src.utils.metrics import ACTIVITY_NAMES, score_in_memory
from src.utils.threshold_optimizer import optimize_thresholds_cmaes

from ..ablations.build_graph_ablation import (
    build_hetero_graph,
    extract_user_behavior_features,
)
from ..ablations.run_training_ablation import (
    align_labels,
    anchor_cutoffs,
    check_window_usability,
    git_revision,
    parse_cutoff_arg,
    resolve_pos_weights,
    set_seed,
    write_result_json,
)

BASELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_OUTPUT_DIR = BASELINE_ROOT / "outputs"
BASELINE_GROUP = "github-activity-baselines"
# Dot-prefixed on purpose: a plain ``wandb/`` directory next to this file would
# shadow the installed package for every module in ``experiments.baselines``.
WANDB_DIR = os.environ.get("BASELINE_WANDB_DIR", str(BASELINE_ROOT / ".wandb-runs"))
Path(WANDB_DIR).mkdir(parents=True, exist_ok=True)

N_USER_FEATURES = 21
FEATURE_TOLERANCE = 1e-6
WandbMode = Literal["online", "offline", "disabled"]


@dataclass
class TabularAnchor:
    """One temporal anchor aligned to the same cohort rows as the GNN."""

    name: str
    cutoff: pd.Timestamp
    features: np.ndarray  # (n_users, 21) float32, exactly data['Users'].x
    feature_names: list[str]
    labels: np.ndarray  # (n_users, 6) int64
    observed: np.ndarray  # (n_users, 6) bool
    mask: np.ndarray  # (n_users,) bool, cohort users we score
    user_ids: np.ndarray
    positives: np.ndarray  # (6,) observed positive count

    @property
    def n_rows(self) -> int:
        return int(self.mask.sum())

    @property
    def n_observed(self) -> int:
        return int(self.observed[self.mask].sum())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LightGBM tabular baseline on data['Users'].x with CMA-ES thresholds.",
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
    ap.add_argument(
        "--max-anchors", type=int, default=0, help="cap anchors, newest first (0 = all)"
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=310,
        help="boosting rounds per activity (n_estimators); dry-run forces 1",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pos-weights",
        default="doc",
        help="'doc' (measured constants), 'auto' (from train prevalence), or 6 comma floats",
    )
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
        default=BASELINE_GROUP,
        help="W&B group for the tabular baseline",
    )
    ap.add_argument(
        "--wandb-tags",
        default="",
        help="extra comma-separated W&B tags (lightgbm/seed/dry-run are added)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="per-run artifact directory (result.json, predictions, importances)",
    )
    ap.add_argument(
        "--results-json",
        default=None,
        help="result JSON path (default: <output-dir>/lightgbm_seed<seed>/result.json)",
    )
    ap.add_argument(
        "--device",
        choices=("cpu", "gpu", "cuda"),
        default="cpu",
        help="LightGBM device; cuda maps to gpu. The suite orchestrator uses cpu.",
    )
    ap.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="LightGBM n_jobs (0 = all cores)",
    )
    ap.add_argument(
        "--verify-graph-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="on the val anchor, rebuild the graph and assert Users.x == this feature block",
    )
    ap.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    ap.add_argument("--lgbm-num-leaves", type=int, default=31)
    ap.add_argument("--lgbm-max-depth", type=int, default=-1)
    ap.add_argument("--lgbm-min-child-samples", type=int, default=20)
    ap.add_argument("--lgbm-subsample", type=float, default=0.8)
    ap.add_argument("--lgbm-subsample-freq", type=int, default=1)
    ap.add_argument("--lgbm-colsample-bytree", type=float, default=0.8)
    ap.add_argument("--lgbm-reg-lambda", type=float, default=1.0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "1-iteration smoke test: one boosting round, one anchor, tiny "
            "CMA-ES, CPU/offline W&B unless overridden"
        ),
    )
    return ap.parse_args(argv)


def apply_dry_run_defaults(args: argparse.Namespace) -> None:
    """Make ``--dry-run`` a real but tiny end-to-end smoke test."""
    print("[dry-run] 1-iteration LightGBM smoke profile (no full training loop):")
    if args.epochs != 1:
        print(f"  epochs {args.epochs} -> 1")
    args.epochs = 1
    if args.max_anchors == 0:
        args.max_anchors = 1
        print("  max_anchors all -> 1")
    if args.cma_maxfevals == 0 or args.cma_maxfevals > 100:
        print(f"  cma_maxfevals {args.cma_maxfevals} -> 100")
        args.cma_maxfevals = 100
    if args.cma_popsize > 8:
        print(f"  cma_popsize {args.cma_popsize} -> 8")
        args.cma_popsize = 8
    if args.wandb_mode is None:
        args.wandb_mode = "offline"
        print("  wandb_mode -> offline")


def resolve_wandb_mode(args: argparse.Namespace) -> WandbMode:
    if args.wandb_mode is not None:
        return cast(WandbMode, args.wandb_mode)
    return "offline" if args.dry_run else "online"


def lgbm_device(device: str) -> str:
    return "gpu" if device in {"gpu", "cuda"} else "cpu"


def build_wandb_tags(args: argparse.Namespace) -> list[str]:
    tags = ["lightgbm", f"seed-{args.seed}", "baseline"]
    if args.dry_run:
        tags.append("dry-run")
    for raw in str(args.wandb_tags).split(","):
        tag = raw.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def load_tabular_anchor(
    tables: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    days: int,
    name: str,
    observed_activity: np.ndarray | None = None,
    *,
    verify_graph_features: bool = False,
) -> TabularAnchor:
    """Build one leakage-checked tabular anchor.

    The feature matrix comes from the same ``extract_user_behavior_features``
    function that produces ``data['Users'].x`` in the graph builder.  When
    ``verify_graph_features`` is true and this is the validation anchor, the
    full graph is also built and the two matrices are compared element-wise.
    """
    started = time.time()
    split = build_split(tables, cutoff, days)
    failures = leakage_failures(tables, split)
    if failures:
        joined = "\n  - ".join(failures)
        raise SystemExit(f"leakage check FAILED for {cutoff.date()}:\n  - {joined}")

    features_df = extract_user_behavior_features(split.features, cutoff)
    users = split.features["Users"].reset_index(drop=True)
    if features_df.shape != (len(users), N_USER_FEATURES):
        raise SystemExit(
            f"feature block has shape {features_df.shape}, expected "
            f"({len(users)}, {N_USER_FEATURES})"
        )
    features = np.asarray(features_df.to_numpy(dtype=np.float32))

    if verify_graph_features and name == "val":
        graph = build_hetero_graph(split.features, cutoff)
        graph_x = graph["Users"].x.detach().cpu().numpy()
        if graph_x.shape != features.shape:
            raise SystemExit(
                f"graph Users.x shape {graph_x.shape} != tabular {features.shape}"
            )
        diff = float(np.max(np.abs(graph_x - features)))
        print(
            f"  [{name}] feature parity vs graph['Users'].x: "
            f"max|diff|={diff:.3e} (shape {features.shape})"
        )
        if diff > FEATURE_TOLERANCE:
            raise SystemExit(
                f"tabular features differ from data['Users'].x by {diff:.3e} "
                f"(tolerance {FEATURE_TOLERANCE:.0e})"
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
        f"  [{name}] {len(users):,} users, {mask.sum():,} cohort rows, "
        f"{features.shape[1]} tabular features"
    )
    if not flags.all():
        dropped = [a for a, flag in zip(ACTIVITY_NAMES, flags, strict=True) if not flag]
        print(f"  [{name}] unobserved activities (dropped): {', '.join(dropped)}")
    print(
        f"  [{name}] positives " + "  ".join(
            (
                f"{act}=0/-"
                if not flag
                else f"{act}={int(pos):,} ({pos / max(mask.sum(), 1) * 100:.1f}%)"
            )
            for act, pos, flag in zip(ACTIVITY_NAMES, positives, flags, strict=True)
        )
    )
    print(f"  [{name}] built in {time.time() - started:.1f}s")

    return TabularAnchor(
        name=name,
        cutoff=cutoff,
        features=features,
        feature_names=[str(column) for column in features_df.columns],
        labels=y,
        observed=observed,
        mask=mask,
        user_ids=user_ids,
        positives=positives,
    )


def lgbm_params(args: argparse.Namespace, activity_index: int, pos_weight: float) -> dict[str, Any]:
    """LightGBM hyperparameters for one activity.

    These are model-specific (the GNN's optimizer/scheduler do not map to
    boosted trees); every split, label, threshold, and metric setting is shared
    with the GNN through the re-used helpers.
    """
    return {
        "objective": "binary",
        "n_estimators": int(args.epochs),
        "learning_rate": float(args.lgbm_learning_rate),
        "num_leaves": int(args.lgbm_num_leaves),
        "max_depth": int(args.lgbm_max_depth),
        "min_child_samples": int(args.lgbm_min_child_samples),
        "subsample": float(args.lgbm_subsample),
        "subsample_freq": int(args.lgbm_subsample_freq),
        "colsample_bytree": float(args.lgbm_colsample_bytree),
        "reg_lambda": float(args.lgbm_reg_lambda),
        "scale_pos_weight": float(pos_weight),
        "n_jobs": int(args.num_threads) if args.num_threads > 0 else -1,
        "random_state": int(args.seed) + activity_index,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True,
        "device": lgbm_device(args.device),
    }


def positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Positive-class probability with a single-class fallback."""
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1].astype(np.float64)
    if proba.ndim == 2 and proba.shape[1] == 1:
        classes = np.asarray(getattr(model, "classes_", [0]))
        sole = float(classes[0]) if classes.size else 0.0
        return np.full(len(X), sole, dtype=np.float64)
    if proba.ndim == 1:
        return proba.astype(np.float64)
    raise ValueError(f"unexpected predict_proba shape {proba.shape}")


def activity_probabilities(
    models: list[Any], fallbacks: list[float], X: np.ndarray
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for model, fallback in zip(models, fallbacks, strict=True):
        if model is None:
            columns.append(np.full(len(X), fallback, dtype=np.float64))
        else:
            columns.append(positive_proba(model, X))
    return np.column_stack(columns)


def feature_importance_rows(
    models: list[Any], feature_names: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for activity, model in zip(ACTIVITY_NAMES, models, strict=True):
        if model is None:
            continue
        booster = model.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        for name, gain_value, split_value in zip(
            feature_names, gain, split, strict=True
        ):
            rows.append(
                {
                    "activity": activity,
                    "feature": name,
                    "importance_gain": float(gain_value),
                    "importance_split": int(split_value),
                }
            )
    return rows


def run_official_evaluator(pred_file: Path, truth_file: Path) -> float | None:
    """Run the TA scorer on the CSVs and parse its Score line."""
    if not truth_file.exists():
        print(f"  [warn] truth file {truth_file} missing; skipping official evaluator")
        return None
    print(
        f"\nRunning official scripts/evaluate.py on {pred_file} against {truth_file}:"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--pred",
            str(pred_file),
            "--truth",
            str(truth_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    print(output, end="" if output.endswith("\n") else "\n")
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Score:"):
            token = stripped.split(":", 1)[1].strip().rstrip("%")
            try:
                return float(token) / 100.0
            except ValueError:
                return None
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        apply_dry_run_defaults(args)
    args.wandb_mode = resolve_wandb_mode(args)
    tags = build_wandb_tags(args)
    if args.wandb_mode == "online" and not (
        os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").exists()
    ):
        print("  [warn] no WANDB_API_KEY and no ~/.netrc: wandb may fail to authenticate")
        print("         export WANDB_API_KEY=... or use --wandb-mode offline")

    set_seed(args.seed)
    started = time.time()
    run_name = args.run_name or f"lightgbm-s{args.seed}"
    # The orchestrator passes an explicit --results-json that already encodes the
    # variant (e.g. outputs/lightgbm-pos-weights-auto_seed42/result.json). Use its
    # parent as the artifact dir, otherwise both LightGBM arms share
    # `lightgbm_seed<seed>/` and the second run overwrites the first's cached
    # predictions while leaving the first's result.json in place.
    if args.results_json:
        run_dir = Path(args.results_json).parent
    else:
        run_dir = Path(args.output_dir) / f"lightgbm_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_json = (
        Path(args.results_json) if args.results_json else run_dir / "result.json"
    )

    print("=" * 78)
    print(
        f"LightGBM baseline | seed {args.seed} | boosting rounds {args.epochs} | "
        f"device {lgbm_device(args.device)} | dry_run={args.dry_run}"
    )
    print(
        f"  features: data['Users'].x ({N_USER_FEATURES} columns) | "
        f"pos_weights={args.pos_weights} | threshold optimizer=CMA-ES"
    )
    print(f"  W&B group={args.wandb_group} | tags={tags} | mode={args.wandb_mode}")
    print(f"  run directory: {run_dir}")
    print(f"  result JSON:   {results_json}")
    print("=" * 78)

    val_cutoff = parse_cutoff_arg(args.val_cutoff)
    train_cutoff = (
        parse_cutoff_arg(args.cutoff)
        if args.cutoff
        else cast(pd.Timestamp, val_cutoff - pd.Timedelta(days=args.days))
    )
    if args.anchors != "none":
        cutoffs = anchor_cutoffs(
            args.anchors,
            start=parse_cutoff_arg(args.anchor_start),
            end=(
                parse_cutoff_arg(args.anchor_end)
                if args.anchor_end
                else train_cutoff
            ),
            stride=args.anchor_stride,
        )
    else:
        cutoffs = [train_cutoff]

    assert_before_test_week(val_cutoff, days=args.days, context="--val-cutoff")
    for cutoff in cutoffs:
        assert_before_test_week(cutoff, days=args.days, context="train cutoff")

    problems = check_window_usability(val_cutoff, args.days)
    if problems:
        print("  [warn] validation label window usability:")
        for problem in problems:
            print("    -", problem)
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

    if args.dry_run and args.max_anchors and len(cutoffs) > args.max_anchors:
        print(
            f"  [dry-run] building only the newest {args.max_anchors} of "
            f"{len(cutoffs)} anchors"
        )
        cutoffs = cutoffs[-args.max_anchors :]

    val_ds = load_tabular_anchor(
        tables,
        val_cutoff,
        args.days,
        "val",
        verify_graph_features=args.verify_graph_features,
    )
    if not val_ds.observed[val_ds.mask].all():
        raise SystemExit("validation anchor has unobserved label cells; pick a clean week")
    val_mask = val_ds.mask
    val_labels = val_ds.labels[val_mask]
    val_user_ids = val_ds.user_ids
    val_X = val_ds.features[val_mask]

    train_anchors: list[TabularAnchor] = []
    skipped: list[str] = []
    for cutoff in cutoffs:
        flags = verdicts.loc[cutoff].to_numpy(dtype=bool)
        if int(flags.sum()) < args.anchor_min_observed:
            skipped.append(
                f"{cutoff.date()} (only {int(flags.sum())}/6 activities usable)"
            )
            continue
        train_anchors.append(
            load_tabular_anchor(
                tables,
                cutoff,
                args.days,
                f"train {cutoff.date()}",
                observed_activity=flags,
                verify_graph_features=False,
            )
        )

    if args.max_anchors and len(train_anchors) > args.max_anchors:
        train_anchors = train_anchors[-args.max_anchors :]
    if skipped:
        print("\n  [skip] anchors the coverage audit cannot support:")
        for line in skipped:
            print("    -", line)
    if not train_anchors:
        raise SystemExit("no usable training anchors; lower --anchor-min-observed")

    print("\nAnchors:")
    for anchor in train_anchors:
        observed_share = anchor.n_observed / max(
            anchor.n_rows * len(ACTIVITY_NAMES), 1
        )
        print(
            f"  {anchor.cutoff.date()}  rows {anchor.n_rows:>6,}  observed cells "
            f"{anchor.n_observed:>7,} ({observed_share * 100:5.1f}%)"
        )

    train_rows = sum(anchor.n_rows for anchor in train_anchors)
    train_positives = np.sum([anchor.positives for anchor in train_anchors], axis=0)
    # Same `--pos-weights auto` denominator as the GNN: observed cells per
    # activity, so a masked cell is not counted as a negative.
    train_observed = np.sum(
        [anchor.observed[anchor.mask].sum(axis=0) for anchor in train_anchors], axis=0
    )
    pos_weights = resolve_pos_weights(
        args.pos_weights,
        np.asarray(train_positives),
        int(train_rows),
        observed_counts=np.asarray(train_observed),
    )
    print(f"  total labelled rows {train_rows:,} across {len(train_anchors)} anchors")
    print(
        "  pos_weight:",
        {
            activity: round(weight, 2)
            for activity, weight in zip(ACTIVITY_NAMES, pos_weights, strict=True)
        },
    )

    train_X = np.vstack([anchor.features[anchor.mask] for anchor in train_anchors])
    train_y = np.vstack([anchor.labels[anchor.mask] for anchor in train_anchors])
    train_obs = np.vstack([anchor.observed[anchor.mask] for anchor in train_anchors])
    print(
        f"  training matrix {train_X.shape} | observed cells "
        f"{int(train_obs.sum()):,}/{train_obs.size:,}"
    )

    models: list[Any] = []
    fallbacks: list[float] = []
    classifier_meta: list[dict[str, Any]] = []
    for activity_index, activity in enumerate(ACTIVITY_NAMES):
        rows = train_obs[:, activity_index]
        X_i = train_X[rows]
        y_i = train_y[rows, activity_index]
        positives = int(y_i.sum())
        if len(np.unique(y_i)) < 2:
            fallback = float(y_i.mean()) if len(y_i) else 0.0
            print(
                f"  [{activity:<18}] constant predictor: no class variation "
                f"({len(y_i):,} rows, {positives:,} positives) -> {fallback:.3f}"
            )
            models.append(None)
            fallbacks.append(fallback)
            classifier_meta.append(
                {
                    "activity": activity,
                    "n_train_rows": int(len(y_i)),
                    "n_positives": positives,
                    "scale_pos_weight": float(pos_weights[activity_index]),
                    "num_trees": 0,
                    "constant_fallback": True,
                }
            )
            continue

        params = lgbm_params(args, activity_index, pos_weights[activity_index])
        model = lgb.LGBMClassifier(**params)
        model.fit(X_i, y_i)
        models.append(model)
        fallbacks.append(0.0)
        classifier_meta.append(
            {
                "activity": activity,
                "n_train_rows": int(len(y_i)),
                "n_positives": positives,
                "scale_pos_weight": float(pos_weights[activity_index]),
                "num_trees": int(model.booster_.num_trees()),
                "constant_fallback": False,
            }
        )
        print(
            f"  [{activity:<18}] {len(y_i):,} rows, {positives:,} positives, "
            f"scale_pos_weight {pos_weights[activity_index]:.2f}, "
            f"trees {model.booster_.num_trees()}"
        )

    # Thresholds are fitted on TRAINING rows only, exactly like the GNN.  The
    # TA metric has no notion of an unobserved cell, so use rows where all six
    # labels are observed (the multi-anchor GNN protocol).
    full_rows = train_obs.all(axis=1)
    if not full_rows.any():
        raise SystemExit("no fully observed training rows for threshold fitting")
    fit_labels = train_y[full_rows]
    fit_probs = activity_probabilities(models, fallbacks, train_X[full_rows])
    theta, fit_metrics = optimize_thresholds_cmaes(
        fit_labels,
        fit_probs,
        sigma0=args.cma_sigma0,
        popsize=args.cma_popsize,
        maxfevals=args.cma_maxfevals,
        seed=args.seed,
    )

    val_probs = activity_probabilities(models, fallbacks, val_X)
    default_metrics = score_in_memory(
        val_labels, (val_probs >= 0.5).astype(np.int64)
    )
    tuned_metrics = score_in_memory(
        val_labels, (val_probs >= theta[None, :]).astype(np.int64)
    )
    print(
        f"Fit | rows {int(full_rows.sum()):,} | score {fit_metrics['score'] * 100:.2f}% | "
        f"val {default_metrics['score'] * 100:.2f}% @0.5 -> "
        f"{tuned_metrics['score'] * 100:.2f}% @tuned "
        f"(macro {tuned_metrics['macro_f1'] * 100:.2f}%, "
        f"micro {tuned_metrics['micro_f1'] * 100:.2f}%)"
    )

    # Artifacts: same row order and names as the GNN's validation outputs.
    val_pred_df = pd.DataFrame(
        (val_probs >= theta[None, :]).astype(np.int64),
        columns=np.asarray(ACTIVITY_NAMES, dtype=object),
    )
    val_pred_df.insert(0, "UserID", val_user_ids)
    val_pred_path = run_dir / "val_pred.csv"
    val_pred_df.to_csv(val_pred_path, index=False)

    val_truth_df = pd.DataFrame(
        val_labels, columns=np.asarray(ACTIVITY_NAMES, dtype=object)
    )
    val_truth_df.insert(0, "UserID", val_user_ids)
    truth_path = run_dir / "val_truth.csv"
    val_truth_df.to_csv(truth_path, index=False)

    np.save(run_dir / "val_probs.npy", val_probs)
    np.save(run_dir / "val_labels.npy", np.asarray(val_labels, dtype=np.int64))

    importance_rows = feature_importance_rows(models, train_anchors[0].feature_names)
    importance_path = run_dir / "feature_importance.csv"
    pd.DataFrame(importance_rows).to_csv(importance_path, index=False)

    official_score = run_official_evaluator(val_pred_path, truth_path)
    if official_score is not None:
        delta = abs(official_score - float(tuned_metrics["score"]))
        if delta > 1e-4:
            print(
                f"  [warn] official score {official_score:.6f} != in-memory "
                f"{float(tuned_metrics['score']):.6f} (delta {delta:.2e})"
            )

    n_parameters = int(sum(meta["num_trees"] for meta in classifier_meta))
    duration = time.time() - started
    result: dict[str, Any] = {
        "status": "ok",
        "variant": "lightgbm",
        "model": "lightgbm",
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "epochs": args.epochs,
        "device": lgbm_device(args.device),
        "macro_f1": float(tuned_metrics["macro_f1"]),
        "micro_f1": float(tuned_metrics["micro_f1"]),
        "score_default_0_5": float(default_metrics["score"]),
        "score_cmaes": float(tuned_metrics["score"]),
        "macro_f1_default_0_5": float(default_metrics["macro_f1"]),
        "micro_f1_default_0_5": float(default_metrics["micro_f1"]),
        "per_activity_f1_cmaes": tuned_metrics["per_event"],
        "per_activity_f1_default_0_5": default_metrics["per_event"],
        "thresholds": np.asarray(theta, dtype=np.float64).tolist(),
        "fit_score": float(fit_metrics["score"]),
        "fit_macro_f1": float(fit_metrics["macro_f1"]),
        "fit_micro_f1": float(fit_metrics["micro_f1"]),
        "cma_generations": int(fit_metrics["n_generations"]),
        "cma_evaluations": int(fit_metrics["n_evaluations"]),
        "official_eval_score": official_score,
        "val_cutoff": str(val_cutoff.date()),
        "val_rows": int(val_mask.sum()),
        "train_anchors": [str(anchor.cutoff.date()) for anchor in train_anchors],
        "n_train_anchors": len(train_anchors),
        "n_train_rows": int(train_rows),
        "n_observed_cells": int(sum(anchor.n_observed for anchor in train_anchors)),
        "train_positives": [int(value) for value in train_positives],
        "pos_weights_mode": str(args.pos_weights),
        "pos_weights": [float(value) for value in pos_weights],
        "feature_dim": N_USER_FEATURES,
        "feature_names": train_anchors[0].feature_names,
        "n_parameters": n_parameters,
        "parameter_kind": "trees",
        "classifiers": classifier_meta,
        "lgbm_params": lgbm_params(args, 0, float(pos_weights[0])),
        "cma_sigma0": args.cma_sigma0,
        "cma_popsize": args.cma_popsize,
        "cma_maxfevals": args.cma_maxfevals,
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
        "val_pred_csv": str(val_pred_path),
        "val_truth_csv": str(truth_path),
        "feature_importance_csv": str(importance_path),
    }
    write_result_json(results_json, result)

    wandb.init(
        project=args.project,
        name=run_name,
        group=args.wandb_group,
        tags=tags,
        mode=cast(WandbMode, args.wandb_mode),
        dir=WANDB_DIR,
        config={
            "model": "lightgbm",
            "epochs": args.epochs,
            "seed": args.seed,
            "device": lgbm_device(args.device),
            "feature_dim": N_USER_FEATURES,
            "feature_names": train_anchors[0].feature_names,
            "pos_weights_mode": str(args.pos_weights),
            "pos_weights": [float(value) for value in pos_weights],
            "lgbm_params": lgbm_params(args, 0, float(pos_weights[0])),
            "threshold_optimizer": "CMA-ES",
            "cma_sigma0": args.cma_sigma0,
            "cma_popsize": args.cma_popsize,
            "cma_maxfevals": args.cma_maxfevals,
            "val_cutoff": str(val_cutoff.date()),
            "train_cutoffs": [str(anchor.cutoff.date()) for anchor in train_anchors],
            "n_train_rows": int(train_rows),
            "n_val_rows": int(val_mask.sum()),
        },
    )
    log_dict: dict[str, Any] = {
        "train/score_theta_fit": float(fit_metrics["score"]),
        "train/macro_f1_theta_fit": float(fit_metrics["macro_f1"]),
        "train/micro_f1_theta_fit": float(fit_metrics["micro_f1"]),
        "val/score_default_0.5": float(default_metrics["score"]),
        "val/score_cmaes": float(tuned_metrics["score"]),
        "val/macro_f1": float(tuned_metrics["macro_f1"]),
        "val/micro_f1": float(tuned_metrics["micro_f1"]),
        "val/score_cmaes_optimistic": float(fit_metrics["score"])
        - float(tuned_metrics["score"]),
        "val/cma_generations": int(fit_metrics["n_generations"]),
        "val/cma_evaluations": int(fit_metrics["n_evaluations"]),
    }
    for index, activity in enumerate(ACTIVITY_NAMES):
        log_dict[f"val_f1/{activity}"] = float(tuned_metrics["per_class_f1"][index])
        log_dict[f"val_threshold/{activity}"] = float(theta[index])
    wandb.log(log_dict)
    wandb.summary["best_val_score"] = float(tuned_metrics["score"])
    wandb.summary["best_val/score_cmaes"] = float(tuned_metrics["score"])
    wandb.summary["best_val/score_default_0.5"] = float(default_metrics["score"])
    wandb.summary["best_val/macro_f1_cmaes"] = float(tuned_metrics["macro_f1"])
    wandb.summary["best_val/micro_f1_cmaes"] = float(tuned_metrics["micro_f1"])
    wandb.summary["best_thresholds"] = np.asarray(theta, dtype=np.float64).tolist()
    wandb.summary["n_estimators"] = args.epochs
    wandb.summary["n_parameters"] = n_parameters
    if official_score is not None:
        wandb.summary["official_eval_score"] = official_score
    wandb.finish()

    print("\n" + "=" * 78)
    print(f"LIGHTGBM RESULT | seed {args.seed}")
    print(f"  Macro-F1        {result['macro_f1'] * 100:6.2f}%")
    print(f"  Micro-F1        {result['micro_f1'] * 100:6.2f}%")
    print(f"  Score @0.5      {result['score_default_0_5'] * 100:6.2f}%")
    print(f"  Score CMA-ES    {result['score_cmaes'] * 100:6.2f}%")
    print(f"  Thresholds      {[round(float(value), 3) for value in result['thresholds']]}")
    print(f"  Result JSON     {results_json}")
    print(f"  Duration        {duration:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
