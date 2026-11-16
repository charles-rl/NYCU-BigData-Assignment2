"""Run the full experiment grid and build the summary tables.

The grid is:

    GNN ablations (4):   full, no-text, no-temporal-decay, no-residual-fusion
    tabular baseline:    lightgbm
    depth study (2):     depth-1layer, depth-2layer
    seeds:               [42, 123, 999]
    -> 7 variants x 3 seeds = 21 runs when executed without --dry-run

All GNN children share the base hyperparameters (epochs=310, warm-restarts,
calibrated doc pos_weights); only the named component or the layer count
differs.  The LightGBM baseline shares the temporal split, calibration, CMA-ES
threshold protocol, and metric, but uses tree-specific model hyperparameters.

Each run is an isolated subprocess (``python -m experiments.ablations.train_*``
or ``experiments.baselines.train_lightgbm``) so a crash or a CUDA OOM in one
arm cannot poison another.  Per-run logs and artifacts stay under
``experiments/ablations/``; the base pipeline's ``splits/``, ``checkpoints/``
and ``logs/`` are never touched.

Default output (full runs)::

    experiments/ablations/ablation_summary.csv
    experiments/ablations/ablation_summary.md

``--dry-run`` is the smoke-test mode: it forwards ``--dry-run`` to each child
(one epoch / one boosting round, one anchor, tiny CMA-ES), defaults to the seed
``42`` only, writes to ``smoke/``/``logs_dry_run/``, and writes
``ablation_summary_dry_run.{csv,md}`` so a smoke test cannot overwrite real
results.  Use ``--skip-existing`` to aggregate existing smoke results without
training them again.

Examples
--------
    # 7 smoke runs (one per variant, seed 42), CPU, no full training
    python -m experiments.ablations.run_all_ablations --dry-run --device cpu

    # the real 21-run grid (defaults: epochs 310, warm-restarts)
    python -m experiments.ablations.run_all_ablations --device cuda

    # optional HPO-comparable schedule
    python -m experiments.ablations.run_all_ablations --device cuda \
        --lr-schedule cosine

    # re-render the tables from existing result.json files, train nothing
    python -m experiments.ablations.run_all_ablations --skip-existing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

ABLATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ABLATION_ROOT.parents[1]
DEFAULT_SEEDS = [42, 123, 999]
ABLATION_GROUP = "github-activity-ablations"
BASELINE_GROUP = "github-activity-baselines"
DEPTH_GROUP = "github-activity-depth-study"
WANDB_GROUPS = {
    "ablations": ABLATION_GROUP,
    "baselines": BASELINE_GROUP,
    "depth": DEPTH_GROUP,
}

VARIANT_ORDER = [
    "full",
    "no-text",
    "no-temporal-decay",
    "no-residual-fusion",
    "lightgbm",
    "depth-1layer",
    "depth-2layer",
]
VARIANT_MODULES = {
    "full": "experiments.ablations.train_full",
    "no-text": "experiments.ablations.train_no_text",
    "no-temporal-decay": "experiments.ablations.train_no_temporal_decay",
    "no-residual-fusion": "experiments.ablations.train_no_residual_fusion",
    "lightgbm": "experiments.baselines.train_lightgbm",
    "depth-1layer": "experiments.ablations.train_gnn_depth",
    "depth-2layer": "experiments.ablations.train_gnn_depth",
}
VARIANT_EXTRA_ARGS = {
    "depth-1layer": ["--num_layers", "1"],
    "depth-2layer": ["--num_layers", "2"],
}
VARIANT_GROUPS = {
    "full": ABLATION_GROUP,
    "no-text": ABLATION_GROUP,
    "no-temporal-decay": ABLATION_GROUP,
    "no-residual-fusion": ABLATION_GROUP,
    "lightgbm": BASELINE_GROUP,
    "depth-1layer": DEPTH_GROUP,
    "depth-2layer": DEPTH_GROUP,
}
GNN_VARIANTS = {variant for variant in VARIANT_ORDER if variant != "lightgbm"}
VARIANT_LABELS = {
    "full": "Full Model",
    "no-text": "No text features",
    "no-temporal-decay": "No temporal decay",
    "no-residual-fusion": "No residual fusion",
    "lightgbm": "LightGBM tabular",
    "depth-1layer": "GNN depth 1",
    "depth-2layer": "GNN depth 2",
}

METRIC_COLUMNS = [
    "macro_f1",
    "micro_f1",
    "score_default_0_5",
    "score_cmaes",
]
METRIC_LABELS = {
    "macro_f1": "Macro-F1 (%)",
    "micro_f1": "Micro-F1 (%)",
    "score_default_0_5": "Score @0.5 (%)",
    "score_cmaes": "Score @CMA-ES (%)",
}
SUMMARY_COLUMNS = [
    "variant",
    "seed",
    "macro_f1",
    "micro_f1",
    "score_default_0_5",
    "score_cmaes",
    "status",
    "returncode",
    "duration_s",
    "epochs",
    "device",
    "val_cutoff",
    "n_train_anchors",
    "n_parameters",
    "wandb_group",
    "wandb_tags",
    "result_json",
    "log_path",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run 4 GNN ablations + LightGBM baseline + 1-vs-2 layer depth study "
            "x 3 seeds, then write CSV/Markdown summaries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help=(
            "comma list; default 42,123,999 for full runs and 42 for --dry-run "
            "(pass explicitly to override either)"
        ),
    )
    ap.add_argument(
        "--variants",
        default=",".join(VARIANT_ORDER),
        help="comma list of variants to run",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="forward --dry-run to each child (1 epoch, one anchor, tiny CMA-ES)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help=(
            "passed to GNN children only; default lets the GNN choose "
            "(dry-run -> cpu). LightGBM runs on CPU unless run directly."
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=310,
        help="passed to every child; LightGBM treats it as n_estimators",
    )
    ap.add_argument(
        "--lr-schedule",
        choices=("cosine", "warm-restarts", "none"),
        default="warm-restarts",
        help="GNN children only; the suite default is warm-restarts",
    )
    ap.add_argument("--db", default=None, help="passed to the child")
    ap.add_argument("--val-cutoff", default=None, help="passed to the child")
    ap.add_argument("--days", type=int, default=None, help="passed to the child")
    ap.add_argument("--anchors", default=None, help="passed to the child")
    ap.add_argument("--max-anchors", type=int, default=None, help="passed to the child")
    ap.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="passed to the child; default online for full runs, offline for dry-run",
    )
    ap.add_argument("--project", default=None, help="passed to the child")
    ap.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="torch/OpenMP/LightGBM threads for the child (0 = library default)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse <run-dir>/result.json instead of re-running that cell",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed run (the summary is still written)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="default: experiments/ablations/outputs (or smoke/ with --dry-run)",
    )
    ap.add_argument(
        "--log-dir",
        default=None,
        help="default: experiments/ablations/logs (or logs_dry_run/ with --dry-run)",
    )
    ap.add_argument("--summary-csv", default=None)
    ap.add_argument("--summary-md", default=None)
    return ap.parse_args(argv)


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def build_command(
    variant: str,
    module: str,
    seed: int,
    output_dir: Path,
    result_json: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        module,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--results-json",
        str(result_json),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    is_gnn = variant in GNN_VARIANTS
    if is_gnn and args.device:
        cmd += ["--device", args.device]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if is_gnn and args.lr_schedule:
        cmd += ["--lr-schedule", args.lr_schedule]
    if args.db:
        cmd += ["--db", args.db]
    if args.val_cutoff:
        cmd += ["--val-cutoff", args.val_cutoff]
    if args.days is not None:
        cmd += ["--days", str(args.days)]
    if args.anchors:
        cmd += ["--anchors", args.anchors]
    if args.max_anchors is not None:
        cmd += ["--max-anchors", str(args.max_anchors)]
    if args.wandb_mode:
        cmd += ["--wandb-mode", args.wandb_mode]
    if args.project:
        cmd += ["--project", args.project]
    if args.num_threads > 0:
        cmd += ["--num-threads", str(args.num_threads)]
    cmd += VARIANT_EXTRA_ARGS.get(variant, [])
    return cmd


def base_row(variant: str, seed: int) -> dict[str, Any]:
    return {
        "variant": variant,
        "seed": seed,
        "macro_f1": float("nan"),
        "micro_f1": float("nan"),
        "score_default_0_5": float("nan"),
        "score_cmaes": float("nan"),
        "status": "not_run",
        "returncode": "",
        "duration_s": float("nan"),
        "epochs": "",
        "device": "",
        "val_cutoff": "",
        "n_train_anchors": "",
        "n_parameters": "",
        "wandb_group": VARIANT_GROUPS.get(variant, ABLATION_GROUP),
        "wandb_tags": "",
        "result_json": "",
        "log_path": "",
    }


def row_from_result(
    data: dict[str, Any],
    *,
    variant: str,
    seed: int,
    status: str,
    returncode: int,
    duration: float,
    result_json: Path,
    log_path: Path,
) -> dict[str, Any]:
    row = base_row(variant, seed)
    row.update(
        {
            "macro_f1": data.get("macro_f1", float("nan")),
            "micro_f1": data.get("micro_f1", float("nan")),
            "score_default_0_5": data.get("score_default_0_5", float("nan")),
            "score_cmaes": data.get("score_cmaes", float("nan")),
            "status": status,
            "returncode": returncode,
            "duration_s": duration,
            "epochs": data.get("epochs", ""),
            "device": data.get("device", ""),
            "val_cutoff": data.get("val_cutoff", ""),
            "n_train_anchors": data.get("n_train_anchors", ""),
            "n_parameters": data.get("n_parameters", ""),
            "wandb_group": data.get(
                "wandb_group", VARIANT_GROUPS.get(variant, ABLATION_GROUP)
            ),
            "wandb_tags": ",".join(data.get("wandb_tags", [])),
            "result_json": str(result_json),
            "log_path": str(log_path),
        }
    )
    return row


def load_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] cannot read {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def run_one(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    module = VARIANT_MODULES[variant]
    run_dir = output_dir / f"{variant}_seed{seed}"
    result_json = run_dir / "result.json"
    log_path = log_dir / f"{variant}_seed{seed}.log"

    if args.skip_existing:
        data = load_result(result_json)
        if data is not None and data.get("status") == "ok":
            print(f"  [skip] {variant} seed {seed}: reusing {result_json}")
            if not log_path.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    f"[skip-existing] reused {result_json}\n", encoding="utf-8"
                )
            return row_from_result(
                data,
                variant=variant,
                seed=seed,
                status="ok",
                returncode=0,
                duration=float(data.get("duration_seconds", 0.0)),
                result_json=result_json,
                log_path=log_path,
            )

    cmd = build_command(variant, module, seed, output_dir, result_json, args)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if args.num_threads > 0:
        env["OMP_NUM_THREADS"] = str(args.num_threads)
        env["MKL_NUM_THREADS"] = str(args.num_threads)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {shlex.join(cmd)}")
    started = time.time()
    returncode = -1
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(cmd)}\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            log.write(f"[launch error] {exc}\n")
            print(f"  [error] could not launch: {exc}")
            return base_row(variant, seed) | {
                "status": "launch_error",
                "result_json": str(result_json),
                "log_path": str(log_path),
            }

        if proc.stdout is not None:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
        returncode = proc.wait()
    duration = time.time() - started

    data = load_result(result_json)
    if returncode == 0 and data is not None and data.get("status") == "ok":
        status = "ok"
    elif returncode == 0:
        status = "missing_result"
    else:
        status = "error"

    if data is None:
        row = base_row(variant, seed)
        row.update(
            {
                "status": status,
                "returncode": returncode,
                "duration_s": round(duration, 1),
                "device": args.device or "",
                "val_cutoff": args.val_cutoff or "",
                "result_json": str(result_json),
                "log_path": str(log_path),
            }
        )
        return row

    return row_from_result(
        data,
        variant=variant,
        seed=seed,
        status=status,
        returncode=returncode,
        duration=round(duration, 1),
        result_json=result_json,
        log_path=log_path,
    )


def fmt_pct(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number * 100:.2f}"


def fmt_plain(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    return str(value)


def fmt_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "—"
    if not math.isfinite(std):
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rows])
        for i in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(
            cell.ljust(widths[i]) for i, cell in enumerate(row)
        ) + " |"

    lines = [render(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def aggregate_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        group = df[df["variant"] == variant]
        ok = group[group["status"] == "ok"]
        row: dict[str, Any] = {"variant": variant, "n": len(ok)}
        for column in METRIC_COLUMNS:
            values: list[float] = []
            for raw in cast(pd.Series, ok[column]).tolist():
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    values.append(number)
            if values:
                mean = sum(values) / len(values)
                row[f"{column}_mean"] = float(mean)
                if len(values) > 1:
                    variance = sum((value - mean) ** 2 for value in values) / (
                        len(values) - 1
                    )
                    row[f"{column}_std"] = float(math.sqrt(variance))
                else:
                    row[f"{column}_std"] = float("nan")
            else:
                row[f"{column}_mean"] = float("nan")
                row[f"{column}_std"] = float("nan")
        rows.append(row)
    return rows


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as plain dicts without relying on pandas' typed ``to_dict`` stubs."""
    columns = [str(column) for column in df.columns]
    return [
        {columns[index]: value for index, value in enumerate(row)}
        for row in df.itertuples(index=False, name=None)
    ]


def render_markdown(
    df: pd.DataFrame,
    meta: dict[str, Any],
    aggregates: list[dict[str, Any]],
) -> str:
    lines = ["# Ablation experiment summary", ""]
    lines.extend(
        [
            f"- Generated (UTC): `{meta['generated_utc']}`",
            f"- Git revision: `{meta['git_revision']}`",
            f"- W&B groups: ablations=`{meta['wandb_groups']['ablations']}`, "
            f"baselines=`{meta['wandb_groups']['baselines']}`, "
            f"depth=`{meta['wandb_groups']['depth']}`",
            f"- Mode: **{'dry-run / smoke' if meta['dry_run'] else 'full'}**",
            f"- Variants: {', '.join(f'`{v}`' for v in meta['variants'])}",
            f"- Seeds: {', '.join(str(s) for s in meta['seeds'])}",
            f"- Epochs: {meta['epochs']}",
            f"- Device: {meta['device']}",
            f"- Runs: {meta['successful_runs']}/{meta['total_runs']} successful",
            "",
            "Scores are percentages in this table; the CSV stores fractions.",
            "",
            "## Per-run results",
            "",
        ]
    )

    headers = ["Variant", "Seed"] + [METRIC_LABELS[c] for c in METRIC_COLUMNS] + ["Status"]
    rows: list[list[str]] = []
    for record in dataframe_records(df):
        rows.append(
            [
                VARIANT_LABELS.get(str(record["variant"]), str(record["variant"])),
                str(record["seed"]),
                *[
                    fmt_pct(record[column]) for column in METRIC_COLUMNS
                ],
                str(record["status"]),
            ]
        )
    lines.append(markdown_table(headers, rows))

    lines.extend(["", "## Mean over seeds", ""])
    agg_headers = ["Variant", "Runs"] + [METRIC_LABELS[c] for c in METRIC_COLUMNS]
    agg_rows: list[list[str]] = []
    for row in aggregates:
        agg_rows.append(
            [
                VARIANT_LABELS.get(str(row["variant"]), str(row["variant"])),
                str(row["n"]),
                *[
                    fmt_mean_std(row[f"{column}_mean"], row[f"{column}_std"])
                    for column in METRIC_COLUMNS
                ],
            ]
        )
    lines.append(markdown_table(agg_headers, agg_rows))

    full = next((row for row in aggregates if row["variant"] == "full"), None)
    if full is not None and math.isfinite(float(full["score_cmaes_mean"])):
        lines.extend(["", "## Effect vs Full Model", ""])
        effect_headers = ["Variant"] + [
            f"Δ {METRIC_LABELS[c].replace(' (%)', '')}" for c in METRIC_COLUMNS
        ]
        effect_rows: list[list[str]] = []
        for row in aggregates:
            deltas: list[str] = []
            for column in METRIC_COLUMNS:
                base = float(full[f"{column}_mean"])
                value = float(row[f"{column}_mean"])
                if not math.isfinite(value) or not math.isfinite(base):
                    deltas.append("—")
                else:
                    delta = (value - base) * 100.0
                    deltas.append(f"{delta:+.2f}")
            effect_rows.append(
                [VARIANT_LABELS.get(str(row["variant"]), str(row["variant"])), *deltas]
            )
        lines.append(markdown_table(effect_headers, effect_rows))
        lines.append("")
        lines.append("Deltas are percentage points; positive means better than Full Model.")

    failed = cast(pd.DataFrame, df[df["status"] != "ok"])
    if not failed.empty:
        lines.extend(["", "## Failed / missing runs", ""])
        for record in dataframe_records(failed):
            lines.append(
                f"- `{record['variant']}` seed `{record['seed']}`: "
                f"**{record['status']}** (log: `{record['log_path']}`)"
            )

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    variants = parse_list(args.variants)
    unknown = [v for v in variants if v not in VARIANT_MODULES]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; choose from {VARIANT_ORDER}")
    variants = [v for v in VARIANT_ORDER if v in variants]

    if args.seeds is not None:
        try:
            seeds = [int(part) for part in parse_list(args.seeds)]
        except ValueError as exc:
            raise SystemExit(f"--seeds must be integers: {exc}") from exc
    elif args.dry_run:
        seeds = [DEFAULT_SEEDS[0]]
    else:
        seeds = list(DEFAULT_SEEDS)
    if not seeds:
        raise SystemExit("no seeds selected")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ABLATION_ROOT / ("smoke" if args.dry_run else "outputs")
    ).resolve()
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else ABLATION_ROOT / ("logs_dry_run" if args.dry_run else "logs")
    ).resolve()
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else ABLATION_ROOT
        / ("ablation_summary_dry_run.csv" if args.dry_run else "ablation_summary.csv")
    ).resolve()
    summary_md = (
        Path(args.summary_md)
        if args.summary_md
        else ABLATION_ROOT
        / ("ablation_summary_dry_run.md" if args.dry_run else "ablation_summary.md")
    ).resolve()

    plan = [(variant, seed) for variant in variants for seed in seeds]
    print("=" * 78)
    print(f"Ablation grid | {len(plan)} run(s) | dry_run={args.dry_run}")
    print(f"  variants: {', '.join(variants)}")
    print(f"  seeds:    {', '.join(str(s) for s in seeds)}")
    print(f"  output:   {output_dir}")
    print(f"  logs:     {log_dir}")
    print(f"  summary:  {summary_csv}")
    print("=" * 78)

    rows: list[dict[str, Any]] = []
    for index, (variant, seed) in enumerate(plan, start=1):
        print(
            f"\n[{index}/{len(plan)}] {variant} | seed {seed} | "
            f"module {VARIANT_MODULES[variant]}"
        )
        row = run_one(variant, seed, args, output_dir, log_dir)
        rows.append(row)
        status = row["status"]
        message = (
            f"  -> {status}: macro_f1={fmt_pct(row['macro_f1'])} "
            f"micro_f1={fmt_pct(row['micro_f1'])} "
            f"score@0.5={fmt_pct(row['score_default_0_5'])} "
            f"score@cma={fmt_pct(row['score_cmaes'])}"
        )
        print(message)
        if args.fail_fast and status != "ok":
            print("  [fail-fast] stopping after the first failed run")
            break

    by_key = {(row["variant"], row["seed"]): row for row in rows}
    ordered = [by_key.get((variant, seed), base_row(variant, seed)) for variant, seed in plan]
    df = pd.DataFrame(ordered, columns=SUMMARY_COLUMNS)

    aggregates = aggregate_rows(df)
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "wandb_groups": WANDB_GROUPS,
        "dry_run": bool(args.dry_run),
        "variants": variants,
        "seeds": seeds,
        "epochs": "1 (dry-run profile)" if args.dry_run else args.epochs,
        "device": (
            args.device
            or (
                "cpu (dry-run default)"
                if args.dry_run
                else "auto (GNN cuda/cpu; LightGBM cpu)"
            )
        ),
        "successful_runs": int(len(df[df["status"] == "ok"])),
        "total_runs": len(plan),
    }

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False, float_format="%.6f")
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(render_markdown(df, meta, aggregates), encoding="utf-8")

    print("\n" + "=" * 78)
    print("Ablation summary")
    print("=" * 78)
    display_rows: list[dict[str, str | int]] = []
    for record in dataframe_records(df):
        display_rows.append(
            {
                "variant": str(record["variant"]),
                "seed": int(record["seed"]),
                **{
                    column: fmt_pct(record[column])
                    for column in METRIC_COLUMNS
                },
                "status": str(record["status"]),
            }
        )
    print(pd.DataFrame(display_rows).to_string(index=False))
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {summary_md}")

    success = bool((df["status"] == "ok").all())
    if not success:
        failed = df[df["status"] != "ok"]
        print(f"\n[warn] {len(failed)} of {len(df)} runs did not complete successfully")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
