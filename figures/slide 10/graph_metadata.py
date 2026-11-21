"""Count what the graph the repo actually builds contains, for the slide 10 figure.

The slide 10 pipeline diagram claims "8 Entity Nodes" and "26 Directed Edges
(13 FK + 13 rev_*)". This probe does not re-implement anything: it imports the
repo's own pipeline (`load_tables` -> `mask_future_leakage` ->
`build_hetero_graph`) at the submission run's last training anchor and dumps the
resulting PyTorch Geometric metadata, so the numbers printed on the slide are
read off the real graph rather than off a docstring.

Nothing under src/ is modified; the only output is `graph_metadata.json` in this
folder.

    /train-data-5-hdd/charles/miniconda3/envs/bda_asg2/bin/python \
        "figures/slide 10/graph_metadata.py"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd  # pyright: ignore[reportMissingImports]  # pyrightconfig.json venvPath is stale on this box; pandas resolves and runs in env bda_asg2

HERE = Path(__file__).resolve().parent  # figures/slide 10
REPO = HERE.parent.parent  # repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.split_io import DEFAULT_DB, load_tables  # noqa: E402
from src.data.temporal_split import mask_future_leakage  # noqa: E402
from src.features.build_graph import build_hetero_graph  # noqa: E402

# The submission run's last legal training anchor (docs/FINAL_RUN.md): the feature
# view is every row strictly before it.  The graded-week graph is built at the later
# test cutoff 2026-11-21 16:00, which the corpus covers only up to 05:49 UTC.
CUTOFF = pd.Timestamp("2026-11-14 00:00:00")
OUT = HERE / "graph_metadata.json"


def main() -> None:
    tables = load_tables(DEFAULT_DB)
    features = mask_future_leakage(tables, cast(pd.Timestamp, CUTOFF))
    graph = build_hetero_graph(features, cast(pd.Timestamp, CUTOFF))

    node_rows = {nt: int(graph[nt].num_nodes) for nt in graph.node_types}
    node_dims = {nt: int(graph[nt].x.shape[1]) for nt in graph.node_types}
    edge_rows = [
        {
            "src": edge[0],
            "relation": edge[1],
            "dst": edge[2],
            "edges": int(graph[edge].edge_index.shape[1]),
            "reverse": edge[1].startswith("rev_"),
        }
        for edge in graph.edge_types
    ]
    forward = [e for e in edge_rows if not e["reverse"]]
    reverse = [e for e in edge_rows if e["reverse"]]

    payload: dict[str, Any] = {
        "cutoff": str(CUTOFF),
        "db": DEFAULT_DB,
        "tables_loaded": sorted(tables),
        "rows_visible_at_cutoff": {
            name: int(len(df)) for name, df in sorted(features.items())
        },
        "num_node_types": len(graph.node_types),
        "num_edge_types": len(graph.edge_types),
        "node_types": node_rows,
        "node_feature_dims": node_dims,
        "forward_relations": forward,
        "reverse_relations": reverse,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"cutoff {CUTOFF} · db {DEFAULT_DB}")
    print(f"node types: {len(graph.node_types)} -> {node_rows}")
    print(f"feature dims: {node_dims}")
    print(f"edge types: {len(graph.edge_types)} "
          f"({len(forward)} forward + {len(reverse)} rev_*)")
    for entry in sorted(forward, key=lambda e: (e["src"], e["relation"])):
        print(f"  {entry['src']:<17} -{entry['relation']:<14}-> "
              f"{entry['dst']:<17} {entry['edges']:>9,}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
