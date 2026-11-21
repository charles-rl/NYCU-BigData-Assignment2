"""Audit hyperparameter parity between the base pipeline and every experiment.

Run from the repository root::

    python -m experiments.ablations.audit_parity

It compares ``scripts/run_training.py`` (the production driver) with the copied
driver used by the full model, the three ablations, and the depth study.  Every
shared hyperparameter must match exactly.  The one intentional, suite-wide
override is ``--epochs 310`` (the base default is 150); it is reported as
``SUITE OVERRIDE`` rather than silently ignored.

It also checks that the LightGBM baseline uses the same temporal split,
calibration, CMA-ES budget, and project defaults, that the pos-weight
calibration arms pin ``pos_weights="auto"`` while every other arm obeys the CLI,
and that the three W&B groups and all nine orchestrator modules exist.

Exit code 0 = PASS, 1 = mismatch.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from scripts.run_training import (
    DOC_POS_WEIGHTS as BASE_POS_WEIGHTS,
)
from scripts.run_training import (
    parse_args as base_parse_args,
)

from ..baselines.train_lightgbm import BASELINE_GROUP
from ..baselines.train_lightgbm import parse_args as lgbm_parse_args
from .run_all_ablations import (
    BASELINE_GROUP as ORCH_BASELINE_GROUP,
)
from .run_all_ablations import (
    DEPTH_GROUP as ORCH_DEPTH_GROUP,
)
from .run_all_ablations import (
    VARIANT_MODULES,
    VARIANT_ORDER,
    WANDB_GROUPS,
)
from .run_training_ablation import (
    DEFAULT_GROUP,
    DEPTH_GROUP,
    VARIANTS,
)
from .run_training_ablation import (
    DOC_POS_WEIGHTS as ABLATION_POS_WEIGHTS,
)
from .run_training_ablation import (
    parse_args as ablation_parse_args,
)

# Shared hyperparameters that must be identical in the base and copied drivers.
PARITY_ARGS = [
    "db",
    "val_cutoff",
    "cutoff",
    "days",
    "anchors",
    "anchor_start",
    "anchor_end",
    "anchor_stride",
    "anchor_observed",
    "anchor_min_observed",
    "max_anchors",
    "no_anchor_shuffle",
    "val_mode",
    "val_fraction",
    "lr",
    "lr_schedule",
    "lr_t0",
    "lr_t_mult",
    "weight_decay",
    "hidden_dim",
    "latent_dim",
    "head_hidden_dim",
    "num_heads",
    "num_gnn_layers",
    "dropout",
    "val_interval",
    "grad_clip",
    "cma_maxfevals",
    "cma_popsize",
    "cma_sigma0",
    "seed",
    "project",
]

# The LightGBM model hyperparameters are tree-specific; these split/label/
# threshold/metric settings are the ones that must match the GNN protocol.
LIGHTGBM_SHARED_ARGS = [
    "db",
    "val_cutoff",
    "cutoff",
    "days",
    "anchors",
    "anchor_start",
    "anchor_end",
    "anchor_stride",
    "anchor_observed",
    "anchor_min_observed",
    "max_anchors",
    "epochs",
    "seed",
    "pos_weights",
    "cma_maxfevals",
    "cma_popsize",
    "cma_sigma0",
    "project",
    "wandb_mode",
]

SUITE_EPOCHS = 310
# The base driver ships the ablation-validated calibration mode as its default
# (``auto``: +4.26 score @0.5 on 3/3 seeds). The copied ablation driver stays on
# ``doc`` because that is the setting its 21 original cells were measured with;
# the arms that study calibration pin ``pos_weights`` in ``VariantSpec`` instead
# of relying on a default. So this is a deliberate divergence, not drift.
BASE_POS_WEIGHTS_MODE = "auto"
ABLATION_POS_WEIGHTS_MODE = "doc"


def _fmt(value: Any) -> str:
    return repr(value)


def compare(
    name: str,
    left: Any,
    right: Any,
    rows: list[tuple[str, str, str, str]],
) -> bool:
    match = left == right
    rows.append((name, _fmt(left), _fmt(right), "MATCH" if match else "MISMATCH"))
    return match


def print_table(
    rows: list[tuple[str, str, str, str]],
    headers: tuple[str, str, str, str] = (
        "argument",
        "run_training.py",
        "ablation copy",
        "status",
    ),
) -> None:
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rows])
        for i in range(4)
    ]
    print("  " + "  ".join(headers[i].ljust(widths[i]) for i in range(4)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(row[i].ljust(widths[i]) for i in range(4)))


def main() -> int:
    print("=" * 78)
    print("Hyperparameter parity audit")
    print("=" * 78)

    base = base_parse_args([])
    ablation = ablation_parse_args([])

    rows: list[tuple[str, str, str, str]] = []
    all_ok = True
    for name in PARITY_ARGS:
        all_ok &= compare(name, getattr(base, name), getattr(ablation, name), rows)
    print("\nGNN ablations vs scripts/run_training.py")
    print_table(rows)
    print(f"\n  parity arguments matched: {sum(1 for row in rows if row[3] == 'MATCH')}/{len(rows)}")

    epochs_ok = bool(ablation.epochs == SUITE_EPOCHS)
    print(
        f"  epochs: base={base.epochs} ablation={ablation.epochs} "
        f"-> {'SUITE OVERRIDE (intentional)' if epochs_ok else 'MISMATCH'}"
    )
    all_ok &= epochs_ok

    pos_mode_ok = bool(
        base.pos_weights == BASE_POS_WEIGHTS_MODE
        and ablation.pos_weights == ABLATION_POS_WEIGHTS_MODE
    )
    print(
        f"  pos_weights: base={base.pos_weights!r} ablation={ablation.pos_weights!r} "
        f"-> {'INTENTIONAL DIVERGENCE (base ships the validated auto mode)' if pos_mode_ok else 'MISMATCH'}"
    )
    all_ok &= pos_mode_ok
    print(
        "  device / wandb_mode: dynamic in the ablation copy "
        "(dry-run -> cpu/offline); excluded from the parity table by design"
    )
    print(
        "  artifact paths (run_name/checkpoint/val_pred_path): suite-specific; "
        "excluded by design"
    )

    pos_weights_ok = BASE_POS_WEIGHTS == ABLATION_POS_WEIGHTS
    print(f"\n  DOC_POS_WEIGHTS: {'MATCH' if pos_weights_ok else 'MISMATCH'} {ABLATION_POS_WEIGHTS}")
    all_ok &= pos_weights_ok

    # Variant flags: only the named factor may differ; depth arms differ only
    # in the number of layers.
    print("\nVariant specifications")
    spec_rows: list[tuple[str, str, str, str]] = []
    expected_non_depth = {
        "full": (False, True, True),
        "no-text": (True, True, True),
        "no-temporal-decay": (False, False, True),
        "no-residual-fusion": (False, True, False),
        # calibration arm: same graph/architecture as `full`; only the loss
        # weights differ, which is why it pins pos_weights instead of a flag.
        "pos-weights-auto": (False, True, True),
    }
    # None means "obey --pos-weights" (every pre-existing arm); the calibration
    # arm pins its mode so it cannot silently drift back to the doc constants.
    expected_pos_weights = {"pos-weights-auto": "auto"}
    for name, spec in VARIANTS.items():
        if spec.num_layers is None:
            expected = expected_non_depth.get(name)
            ok = expected == (spec.mask_text, spec.edge_decay, spec.residual_fusion)
            ok &= spec.wandb_group == DEFAULT_GROUP
            ok &= spec.pos_weights == expected_pos_weights.get(name)
            status = "OK" if ok else "MISMATCH"
            all_ok &= ok
            spec_rows.append(
                (
                    name,
                    f"layers=- mask={spec.mask_text} decay={spec.edge_decay} "
                    f"fusion={spec.residual_fusion} pos_weights={spec.pos_weights}",
                    spec.wandb_group,
                    status,
                )
            )
        else:
            ok = (
                spec.mask_text is False
                and spec.edge_decay is True
                and spec.residual_fusion is True
                and spec.wandb_group == DEPTH_GROUP
                and spec.pos_weights is None
            )
            status = "OK" if ok else "MISMATCH"
            all_ok &= ok
            spec_rows.append(
                (
                    name,
                    f"layers={spec.num_layers} mask={spec.mask_text} decay={spec.edge_decay} fusion={spec.residual_fusion}",
                    spec.wandb_group,
                    status,
                )
            )
    for row in spec_rows:
        print(f"  {row[0]:<20} {row[1]:<58} {row[2]:<30} {row[3]}")

    depth_ok = (
        VARIANTS.get("depth-1layer") is not None
        and VARIANTS["depth-1layer"].num_layers == 1
        and VARIANTS.get("depth-2layer") is not None
        and VARIANTS["depth-2layer"].num_layers == 2
    )
    print(f"\n  depth arms (1 vs 2 layers): {'OK' if depth_ok else 'MISMATCH'}")
    all_ok &= depth_ok

    # Orchestrator coverage: eight variants, three distinct W&B groups.
    print("\nOrchestrator coverage")
    print(f"  variants ({len(VARIANT_ORDER)}): {', '.join(VARIANT_ORDER)}")
    expected_variants = [
        "full",
        "no-text",
        "no-temporal-decay",
        "no-residual-fusion",
        "pos-weights-auto",
        "lightgbm",
        "lightgbm-pos-weights-auto",
        "depth-1layer",
        "depth-2layer",
    ]
    coverage_ok = expected_variants == VARIANT_ORDER
    print(f"  expected 9-variant order: {'OK' if coverage_ok else 'MISMATCH'}")
    all_ok &= coverage_ok
    for module in sorted(set(VARIANT_MODULES.values())):
        exists = importlib.util.find_spec(module) is not None
        print(f"  module {module:<50} {'OK' if exists else 'MISSING'}")
        all_ok &= exists

    groups_ok = (
        WANDB_GROUPS.get("ablations") == DEFAULT_GROUP
        and WANDB_GROUPS.get("baselines") == BASELINE_GROUP == ORCH_BASELINE_GROUP
        and WANDB_GROUPS.get("depth") == DEPTH_GROUP == ORCH_DEPTH_GROUP
    )
    print(
        "  W&B groups: "
        f"ablations={WANDB_GROUPS.get('ablations')} "
        f"baselines={WANDB_GROUPS.get('baselines')} "
        f"depth={WANDB_GROUPS.get('depth')} "
        f"-> {'OK' if groups_ok else 'MISMATCH'}"
    )
    all_ok &= groups_ok

    # LightGBM split/threshold parity.
    lgbm = lgbm_parse_args([])
    lgbm_rows: list[tuple[str, str, str, str]] = []
    for name in LIGHTGBM_SHARED_ARGS:
        all_ok &= compare(
            name, getattr(ablation, name), getattr(lgbm, name), lgbm_rows
        )
    print("\nLightGBM baseline vs GNN split/threshold protocol")
    print_table(
        lgbm_rows,
        headers=("argument", "GNN protocol", "LightGBM", "status"),
    )
    print(
        f"\n  shared arguments matched: "
        f"{sum(1 for row in lgbm_rows if row[3] == 'MATCH')}/{len(lgbm_rows)}"
    )

    print("\n" + "=" * 78)
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
