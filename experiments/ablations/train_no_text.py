"""Ablation: mask the six structural/textual pattern features.

The six features removed from the node feature matrices are:

* ``Issues``: ``log_issue_len``, ``has_markdown_structure``, ``has_prefix``
* ``PullRequests``: ``branch_depth``
* ``PRReviewComments``: ``log_diff_len``, ``path_depth``

``--text-mode zero`` (default) writes zeros and keeps the feature width, so the
parameter count and architecture stay identical to the full model.
``--text-mode drop`` slices the columns out instead.

Usage (from the repository root)::

    python -m experiments.ablations.train_no_text --dry-run --device cpu --seed 42

    python -m experiments.ablations.train_no_text --seed 42 --text-mode zero \
        --device cuda

All arguments are inherited from ``experiments.ablations.run_training_ablation``;
this module only pins the variant.
"""

from __future__ import annotations

import sys

from .run_training_ablation import main

if __name__ == "__main__":
    sys.exit(main(variant="no-text"))
