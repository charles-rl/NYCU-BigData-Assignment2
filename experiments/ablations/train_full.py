"""Full-model control arm for the ablation suite.

This is the reference arm every ablation is compared against: temporal edge
decay ON, raw-user residual fusion ON, and all six structural/textual pattern
features present.

Usage (from the repository root)::

    python -m experiments.ablations.train_full --dry-run --device cpu --seed 42

    python -m experiments.ablations.train_full --seed 42 --epochs 310 \
        --lr-schedule cosine --device cuda

All arguments are inherited from ``experiments.ablations.run_training_ablation``;
this module only pins the variant.
"""

from __future__ import annotations

import sys

from .run_training_ablation import main

if __name__ == "__main__":
    sys.exit(main(variant="full"))
