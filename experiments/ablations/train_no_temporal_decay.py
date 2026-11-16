"""Ablation: disable exponential temporal edge weighting.

The full arm weights every event edge by ``exp(-min(dt_days, max_days) / tau)``
with the per-activity taus measured in ``docs/VALIDATION_DATA.md``.  This arm
sets every edge weight to exactly ``1.0``, leaving the graph topology, the
model parameters, and the training loop unchanged.

Static ``Repositories -> Users owner`` edges are unit-weighted in both arms
because they carry no event time.

Usage (from the repository root)::

    python -m experiments.ablations.train_no_temporal_decay --dry-run \
        --device cpu --seed 42

    python -m experiments.ablations.train_no_temporal_decay --seed 42 \
        --device cuda

All arguments are inherited from ``experiments.ablations.run_training_ablation``;
this module only pins the variant.
"""

from __future__ import annotations

import sys

from .run_training_ablation import main

if __name__ == "__main__":
    sys.exit(main(variant="no-temporal-decay"))
