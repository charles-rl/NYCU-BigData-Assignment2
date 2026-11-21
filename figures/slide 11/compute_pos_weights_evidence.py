"""Extract the numbers behind the slide 11 bar chart from the repo's own artifacts.

Nothing on this slide is typed in by hand: the two bars, their spread, the gain
and the weight-inflation ratios are all *parsed* out of files the ablation grid
already wrote, so the figure can be re-derived and re-checked.

Sources (read-only; this script never writes outside ``figures/slide 11/``):

* ``experiments/ablations/ablation_summary.csv`` -- one row per (variant, seed)
  with ``score_default_0_5`` and ``score_cmaes``.  The two arms shown on slide 11
  are ``full`` (the production config with ``--pos-weights doc``) and
  ``pos-weights-auto`` (identical graph/architecture/anchors, only the BCE
  positive weights change).
* ``experiments/ablations/RESULTS.md`` Experiment C -- the per-activity table
  ``doc constants | auto (this run) | doc ÷ auto``, which is what says how badly
  the static weights are inflated.
* ``scripts/run_training.py:90`` ``DOC_POS_WEIGHTS`` and
  ``scripts/run_training.py:553-600`` ``resolve_pos_weights``, whose docstring
  states the auto rule ``(N_observed - N_pos) / N_pos``.
* ``src/utils/metrics.py:159`` -- ``score = (macro_f1 + micro_f1) / 2``, the
  definition of the y axis ("Score @ 0.5 Threshold").

Run it to print the table and write ``pos_weights_evidence.json`` next to it.

    conda run -n bda_asg2 python "figures/slide 11/compute_pos_weights_evidence.py"
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent  # figures/slide 11
REPO = HERE.parent.parent  # project root
SUMMARY_CSV = REPO / "experiments" / "ablations" / "ablation_summary.csv"
RESULTS_MD = REPO / "experiments" / "ablations" / "RESULTS.md"
OUT = HERE / "pos_weights_evidence.json"

# the two arms the slide compares
DOC_ARM = "full"
AUTO_ARM = "pos-weights-auto"

ACTIVITIES = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)

# "| open_pr | 7.28 | 3.46 | 2.10× |" -- the trailing multiplication sign keeps
# this apart from the per-activity F1 table further down the same section.
RATIO_ROW = re.compile(
    r"^\|\s*(" + "|".join(ACTIVITIES) + r")\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*×\s*\|\s*$",
    re.MULTILINE,
)


def read_scores(path: Path) -> dict[str, dict[str, float]]:
    """variant -> seed -> score_default_0_5, plus the protocol fields."""
    if not path.is_file():
        raise SystemExit(f"missing ablation artifact: {path}")
    out: dict[str, dict[str, float]] = {}
    protocol: dict[str, object] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row.get("variant")
            if variant not in (DOC_ARM, AUTO_ARM):
                continue
            if row.get("status", "ok") != "ok":
                continue
            seed = str(row["seed"])
            out.setdefault(variant, {})[seed] = float(row["score_default_0_5"]) * 100.0
            cma = row.get("score_cmaes")
            if cma:
                out.setdefault(variant + "@cmaes", {})[seed] = float(cma) * 100.0
            protocol.setdefault("val_cutoff", row.get("val_cutoff"))
            protocol.setdefault("n_train_anchors", row.get("n_train_anchors"))
            protocol.setdefault("epochs", row.get("epochs"))
            protocol.setdefault("device", row.get("device"))
    for variant in (DOC_ARM, AUTO_ARM):
        if variant not in out:
            raise SystemExit(f"{variant} has no rows in {path}")
    return out


def summarize(values: dict[str, float]) -> dict[str, object]:
    """Mean / sample sd of the seed replicates, and the raw per-seed values."""
    series = [values[key] for key in sorted(values, key=int)]
    seeds = sorted(values, key=int)
    return {
        "seeds": seeds,
        "values_pct": [round(value, 4) for value in series],
        "mean_pct": round(mean(series), 4),
        "sd_pct": round(stdev(series), 4) if len(series) > 1 else None,
        "n_seeds": len(series),
    }


def paired_delta(doc: dict[str, float], auto: dict[str, float]) -> dict[str, object]:
    """Per-seed differences, which is how the grid counts wins (3/3 seeds)."""
    seeds = sorted(set(doc) & set(auto), key=int)
    deltas = [auto[seed] - doc[seed] for seed in seeds]
    return {
        "seeds": seeds,
        "delta_pts": [round(delta, 4) for delta in deltas],
        "mean_delta_pts": round(mean(deltas), 4) if deltas else None,
        "wins": sum(1 for delta in deltas if delta > 0),
        "n_seeds": len(seeds),
    }


def read_ratios(path: Path) -> dict[str, dict[str, float]]:
    """The Experiment C table: activity -> {doc, auto, ratio}."""
    if not path.is_file():
        raise SystemExit(f"missing ablation write-up: {path}")
    text = path.read_text(encoding="utf-8")
    ratios: dict[str, dict[str, float]] = {}
    for match in RATIO_ROW.finditer(text.replace("\u00a0", " ")):
        activity, doc, auto, ratio = match.groups()
        ratios[activity] = {
            "doc_weight": float(doc),
            "auto_weight": float(auto),
            "doc_over_auto": float(ratio),
        }
    missing = [name for name in ACTIVITIES if name not in ratios]
    if missing:
        raise SystemExit(f"Experiment C ratio table incomplete, no rows for {missing}")
    return ratios


def main() -> None:
    scores = read_scores(SUMMARY_CSV)
    doc, auto = scores[DOC_ARM], scores[AUTO_ARM]
    ratios = read_ratios(RESULTS_MD)
    ratio_values = [row["doc_over_auto"] for row in ratios.values()]

    payload = {
        "source_csv": str(SUMMARY_CSV.relative_to(REPO)),
        "source_md": str(RESULTS_MD.relative_to(REPO)),
        "score_definition": "score = (macro_f1 + micro_f1) / 2 (src/utils/metrics.py:159)",
        "protocol": {
            "val_cutoff": None,
            "n_train_anchors": None,
            "epochs": None,
            "device": None,
        },
        "doc_arm": {"variant": DOC_ARM, "pos_weights": "doc", **summarize(doc)},
        "auto_arm": {"variant": AUTO_ARM, "pos_weights": "auto", **summarize(auto)},
        "delta_at_0_5": paired_delta(doc, auto),
        "doc_over_auto_ratios": ratios,
        "ratio_range": {
            "min": min(ratio_values),
            "max": max(ratio_values),
            "median": round(sorted(ratio_values)[len(ratio_values) // 2], 3),
        },
    }
    cma_doc, cma_auto = scores.get(DOC_ARM + "@cmaes"), scores.get(AUTO_ARM + "@cmaes")
    if cma_doc and cma_auto:
        payload["cmaes_reference"] = {
            "doc_mean_pct": round(mean(cma_doc.values()), 4),
            "auto_mean_pct": round(mean(cma_auto.values()), 4),
            "delta_at_cmaes_pts": round(mean(cma_auto.values()) - mean(cma_doc.values()), 4),
            "note": (
                "the tuned-score gain, for contrast with the @0.5 gain; "
                "RESULTS.md calls anything under ~0.2 seed noise"
            ),
        }

    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for key in ("val_cutoff", "n_train_anchors", "epochs", "device"):
        value = payload["protocol"].get(key)
        if value is not None:
            payload["protocol"][key] = value
    print(f"wrote {OUT}")
    print(
        f"  doc  ({DOC_ARM:>16}) @0.5 "
        f"{payload['doc_arm']['mean_pct']:.2f} +- {payload['doc_arm']['sd_pct']:.2f} "
        f"over seeds {', '.join(payload['doc_arm']['seeds'])}"
    )
    print(
        f"  auto ({AUTO_ARM:>16}) @0.5 "
        f"{payload['auto_arm']['mean_pct']:.2f} +- {payload['auto_arm']['sd_pct']:.2f} "
        f"over seeds {', '.join(payload['auto_arm']['seeds'])}"
    )
    delta = payload["delta_at_0_5"]
    print(
        f"  gain @0.5  {delta['mean_delta_pts']:+.2f} pts "
        f"(per seed {', '.join(f'{d:+.2f}' for d in delta['delta_pts'])}; "
        f"{delta['wins']}/{delta['n_seeds']} seeds win)"
    )
    if "cmaes_reference" in payload:
        ref = payload["cmaes_reference"]
        print(
            f"  for contrast @CMA-ES: {ref['doc_mean_pct']:.2f} -> "
            f"{ref['auto_mean_pct']:.2f} ({ref['delta_at_cmaes_pts']:+.2f} pts)"
        )
    print(
        "  doc weights ÷ this run's auto weights: "
        + ", ".join(f"{name} {ratios[name]['doc_over_auto']:.2f}x" for name in ACTIVITIES)
        + f"  (range {payload['ratio_range']['min']:.2f}-{payload['ratio_range']['max']:.2f})"
    )


if __name__ == "__main__":
    main()
