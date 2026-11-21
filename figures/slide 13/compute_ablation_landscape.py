"""Measure the ablation numbers the slide 13 chart draws, straight from the grid.

Slide 13 is the ablation landscape: six bars in three matched pairs on the
official Score @ CMA-ES axis.  The slide spec quoted some of these numbers from
memory and one of them was a guess ("e.g., 63.70%"), so nothing here is typed
in: every value is read from ``experiments/ablations/ablation_summary.csv``, the
27-row summary the grid itself writes (3 seeds x 9 variants).

Why the pairs have to be read as pairs
--------------------------------------
The grid changes exactly one thing per arm, and the arms are not all built on the
same baseline:

* ``depth-1layer`` / ``depth-2layer`` pin ``--num_layers 1`` / ``2`` and both use
  the driver's default ``--pos-weights doc``
  (``experiments/ablations/run_all_ablations.py:116-129``).  ``full`` is also a
  1-layer doc-weights model, which is why it shares ``depth-1layer``'s parameter
  count - it doubles as a replication of it.
* ``no-temporal-decay`` removes the edge-decay term from ``full``; both are
  doc-weights, 2,304,630 parameters.
* ``lightgbm`` / ``lightgbm-pos-weights-auto`` differ only in ``scale_pos_weight``.
* ``pos-weights-auto`` is the **submission recipe** (1 layer, auto weights) and is
  the best mean in the grid.  It is drawn as a reference line, not as one bar of a
  matched pair: pairing e.g. its 65.84 against ``no-temporal-decay``'s 65.77 would
  compare two arms that differ in loss weights as well as in the mechanism under
  test.  No 2x2 of decay x calibration was ever run.

Noise floor, from the grid itself: ``full`` alone scores 65.63 / 65.80 / 65.70
across seeds (sd 0.07).  ``experiments/ablations/RESULTS.md:83-84`` says to treat
any delta below ~0.2 as seed noise, so the chart also carries each bar's per-seed
spread and every pair's delta.

Writes ``figures/slide 13/ablation_landscape.json``.

    conda run -n bda_asg2 python "figures/slide 13/compute_ablation_landscape.py"
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent  # figures/slide 13
REPO = HERE.parent.parent

SUMMARY_CSV = REPO / "experiments" / "ablations" / "ablation_summary.csv"
OUT = HERE / "ablation_landscape.json"

# the six bars, as matched pairs: (baseline, variant-under-test) per group
GROUPS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("Depth & Efficiency", ("depth-1layer", "depth-2layer")),
    ("Temporal Mechanism", ("full", "no-temporal-decay")),
    ("Tabular Baseline Parity", ("lightgbm", "lightgbm-pos-weights-auto")),
)
REFERENCE = "pos-weights-auto"  # the submission recipe, drawn as a reference line


@dataclass(frozen=True)
class Arm:
    variant: str
    score_cmaes: float  # mean over seeds, %
    score_cmaes_sd: float
    score_default: float
    score_default_sd: float
    macro_f1: float
    micro_f1: float
    minutes_per_cell: float
    n_parameters: int
    per_seed: dict[str, float]  # seed -> score @CMA-ES, %
    spread: tuple[float, float]
    seeds: tuple[str, ...]


def mean_sd(values: list[float], scale: float = 100.0) -> tuple[float, float]:
    scaled = [v * scale for v in values]
    return statistics.mean(scaled), (statistics.stdev(scaled) if len(scaled) > 1 else 0.0)


def read_arms(path: Path) -> tuple[dict[str, Arm], dict[str, object]]:
    rows: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"].strip().lower() != "ok":
                raise SystemExit(f"{path.name}: non-ok cell {row['variant']} seed {row['seed']}")
            rows.setdefault(row["variant"], []).append(row)

    arms: dict[str, Arm] = {}
    for variant, cells in rows.items():
        if len(cells) < 2:
            raise SystemExit(f"{path.name}: need >= 2 seeds for {variant}")
        cmaes = [float(c["score_cmaes"]) for c in cells]
        default = [float(c["score_default_0_5"]) for c in cells]
        per_seed = {c["seed"]: round(float(c["score_cmaes"]) * 100, 2) for c in cells}
        score, score_sd = mean_sd(cmaes)
        unc, unc_sd = mean_sd(default)
        arms[variant] = Arm(
            variant=variant,
            score_cmaes=round(score, 4),
            score_cmaes_sd=round(score_sd, 4),
            score_default=round(unc, 4),
            score_default_sd=round(unc_sd, 4),
            macro_f1=round(mean_sd([float(c["macro_f1"]) for c in cells])[0], 4),
            micro_f1=round(mean_sd([float(c["micro_f1"]) for c in cells])[0], 4),
            minutes_per_cell=round(statistics.mean(float(c["duration_s"]) for c in cells) / 60, 1),
            n_parameters=int(statistics.mode(int(c["n_parameters"]) for c in cells)),
            per_seed=dict(sorted(per_seed.items())),
            spread=(round(min(per_seed.values()), 2), round(max(per_seed.values()), 2)),
            seeds=tuple(sorted((c["seed"] for c in cells), key=int)),
        )
    return arms, protocol_of(rows)


def protocol_of(rows: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    """Grid-wide protocol fields, read from the cells so nothing is typed in here.

    ``device`` legitimately differs (the LightGBM arms are CPU-only, the GNN arms
    are CUDA), so these are reported as the set of values seen.  The split fields
    the chart quotes must be uniform, and that is asserted.
    """
    cells = [cell for group in rows.values() for cell in group]

    def distinct(column: str) -> list[str]:
        return sorted({cell[column].strip() for cell in cells if cell[column].strip()})

    protocol = {
        "val_cutoff": distinct("val_cutoff"),
        "n_train_anchors": distinct("n_train_anchors"),
        "device": distinct("device"),
        "epochs": distinct("epochs"),
        "cells": len(cells),
    }
    for column in ("val_cutoff", "n_train_anchors"):
        if len(protocol[column]) != 1:  # type: ignore[arg-type]
            raise SystemExit(f"grid is not uniform on {column}: {protocol[column]}")
    return protocol


def wins(base: Arm, arm: Arm) -> int:
    """Per-seed wins of `arm` over `base`, the tie-breaker RESULTS.md:83-84 uses."""
    return sum(1 for seed, value in arm.per_seed.items() if value > base.per_seed.get(seed, -1))


def main() -> int:
    arms, protocol = read_arms(SUMMARY_CSV)
    wanted = [variant for _, pair in GROUPS for variant in pair] + [REFERENCE]
    missing = [variant for variant in wanted if variant not in arms]
    if missing:
        raise SystemExit(f"{SUMMARY_CSV} has no cells for {missing}")

    pairs = []
    for group, (base_name, arm_name) in GROUPS:
        base, arm = arms[base_name], arms[arm_name]
        pairs.append(
            {
                "group": group,
                "baseline": base.variant,
                "variant": arm.variant,
                "delta_cmaes": round(arm.score_cmaes - base.score_cmaes, 4),
                "delta_default": round(arm.score_default - base.score_default, 4),
                "per_seed_wins": wins(base, arm),
                "seed_count": len(arm.seeds),
                "param_delta_pct": round(
                    100.0 * (arm.n_parameters - base.n_parameters) / base.n_parameters, 1
                ),
                "minutes_delta_pct": round(
                    100.0 * (arm.minutes_per_cell - base.minutes_per_cell) / base.minutes_per_cell,
                    1,
                ),
            }
        )

    base_full, ref = arms["full"], arms[REFERENCE]
    lgbm_auto = arms["lightgbm-pos-weights-auto"]
    payload = {
        "source": str(SUMMARY_CSV.relative_to(REPO)),
        "seeds": list(ref.seeds),
        "protocol": protocol,
        "reference": {
            "variant": REFERENCE,
            **asdict(ref),
            "meaning": "submission recipe: 1 HGT layer + auto pos weights; best mean in the grid",
            "delta_vs_full": round(ref.score_cmaes - base_full.score_cmaes, 4),
            "delta_vs_lightgbm_auto": round(ref.score_cmaes - lgbm_auto.score_cmaes, 4),
        },
        "groups": [{"group": g, "arms": [asdict(arms[b]), asdict(arms[a])]} for g, (b, a) in GROUPS],
        "pair_deltas": pairs,
        "noise_floor": {
            "full_per_seed": base_full.per_seed,
            "full_sd": base_full.score_cmaes_sd,
            "rule": "treat any delta below ~0.2 as seed noise (experiments/ablations/RESULTS.md:83-84)",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"source: {payload['source']} · seeds {payload['seeds']} · protocol {protocol}")
    for entry in payload["groups"]:
        print(f"\n{entry['group']}")
        for arm in entry["arms"]:
            print(
                f"  {arm['variant']:26} @CMA {arm['score_cmaes']:6.2f} ± {arm['score_cmaes_sd']:.2f}"
                f"  @0.5 {arm['score_default']:6.2f}  {arm['minutes_per_cell']:6.1f} min/cell"
                f"  {arm['n_parameters']:,} params  spread {arm['spread'][0]:.2f}-{arm['spread'][1]:.2f}"
            )
    print()
    for pair in pairs:
        print(
            f"  Δ {pair['variant']} vs {pair['baseline']}: "
            f"{pair['delta_cmaes']:+.2f} @CMA ({pair['per_seed_wins']}/{pair['seed_count']} seeds), "
            f"{pair['delta_default']:+.2f} @0.5, params {pair['param_delta_pct']:+.1f}%, "
            f"wall {pair['minutes_delta_pct']:+.1f}%"
        )
    ref_line = payload["reference"]
    print(
        f"\nreference {REFERENCE}: @CMA {ref_line['score_cmaes']:.2f} ± {ref_line['score_cmaes_sd']:.2f}"
        f" · @0.5 {ref_line['score_default']:.2f} · vs full {ref_line['delta_vs_full']:+.2f}"
        f" · vs {lgbm_auto.variant} {ref_line['delta_vs_lightgbm_auto']:+.2f}"
    )
    floor = payload["noise_floor"]
    print(f"noise floor: full per seed {floor['full_per_seed']} · sd {floor['full_sd']:.2f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
