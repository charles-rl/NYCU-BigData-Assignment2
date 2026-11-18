"""Ablation: derive ``pos_weight`` from this run's training anchors.

Every other arm uses the static constants in ``docs/VALIDATION_DATA.md``
section 5 (``DOC_POS_WEIGHTS``), which were fitted on a 252-anchor panel that
spans 2025-12-02 .. 2026-09-30 — i.e. it includes days after the 2026-02-15
validation week and after any single run's anchors.  Those constants are 2.0–2.6x
the prevalence ratio of the 8 anchors a run actually trains on, which is why the
CMA-ES thresholds land at 0.76–0.88 for five activities: the model over-predicts
and theta has to correct it.

This arm replaces them with the textbook weight measured where the model is
actually fitted::

    pos_weight[a] = (N_observed[a] - N_pos[a]) / N_pos[a]

``N_observed[a]`` is the number of label cells for activity ``a`` that the
coverage audit says are observable across this run's anchors — masked cells are
excluded, because a cell that never enters the loss is not a negative.

Nothing else changes: same graph builder, same 21 node features, same
architecture, same anchors, same CMA-ES protocol (theta still fitted on training
rows only), same validation week.

Usage (from the repository root)::

    python -m experiments.ablations.train_pos_weights_auto --dry-run --device cpu --seed 42

    python -m experiments.ablations.train_pos_weights_auto --seed 42 --device cuda

All arguments are inherited from ``experiments.ablations.run_training_ablation``;
this module only pins the variant.  ``--pos-weights auto`` on its own also works,
but the variant pins the mode so the arm cannot drift if a flag is forgotten.
"""

from __future__ import annotations

import sys

from .run_training_ablation import main

if __name__ == "__main__":
    sys.exit(main(variant="pos-weights-auto"))
