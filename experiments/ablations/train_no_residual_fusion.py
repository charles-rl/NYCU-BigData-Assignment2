"""Ablation: remove the raw-user residual fusion from the prediction heads.

The full model concatenates the post-GNN user embedding with the raw
``data['Users'].x`` feature block before each of the six heads::

    combined = [gnn_user_embedding ; raw_users_x]   # latent_dim + 21

This arm passes only the post-GNN embedding::

    combined = gnn_user_embedding                  # latent_dim

The graph, the edge weights, and all six text features are unchanged.

Usage (from the repository root)::

    python -m experiments.ablations.train_no_residual_fusion --dry-run \
        --device cpu --seed 42

    python -m experiments.ablations.train_no_residual_fusion --seed 42 \
        --device cuda

All arguments are inherited from ``experiments.ablations.run_training_ablation``;
this module only pins the variant.
"""

from __future__ import annotations

import sys

from .run_training_ablation import main

if __name__ == "__main__":
    sys.exit(main(variant="no-residual-fusion"))
