"""GNN layer-depth study: 1-layer vs 2-layer HGT.

This is the entry point for Experiment B.  It reuses the copied base driver
``run_training_ablation.py`` (itself a copy of ``scripts/run_training.py``), so
every hyperparameter except the number of HGT layers is identical to the base
pipeline and to the other ablation arms:

    python -m experiments.ablations.train_gnn_depth --num_layers 1 ...
    python -m experiments.ablations.train_gnn_depth --num_layers 2 ...

The default is one layer, matching ``scripts/run_training.py``.  The explicit
``--num_layers`` value selects the ``depth-1layer`` or ``depth-2layer`` variant,
which changes only ``--num-gnn-layers`` and files the run under the
``github-activity-depth-study`` W&B group.

All other CLI arguments (``--seed``, ``--epochs`` default 310, ``--device``,
``--dry-run``, split and loss options) are inherited from the shared driver.
"""

from __future__ import annotations

import argparse
import sys

from .run_training_ablation import main as _shared_main


def _preparse_num_layers(argv: list[str]) -> int:
    """Read --num_layers/--num-layers without consuming the shared CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--num_layers",
        "--num-layers",
        dest="num_layers",
        type=int,
        choices=(1, 2),
        default=1,
    )
    known, _rest = parser.parse_known_args(argv)
    return int(known.num_layers)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    num_layers = _preparse_num_layers(args)
    variant = f"depth-{num_layers}layer"
    return _shared_main(args, variant=variant)


if __name__ == "__main__":
    sys.exit(main())
