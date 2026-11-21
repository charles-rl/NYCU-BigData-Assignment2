"""Slide 10 figure: end-to-end pipeline as 5 phase-coded minimalist blocks.

High-density, large-type aesthetic version preserving all symbols imported by
`check_layout.py`: BADGE_INSET, BLOCK_W, BLOCKS, CARD_Y0, CARD_Y1, N_BLOCKS,
Probe, badge_stack, block_x, and build.
"""

from __future__ import annotations

import json
import matplotlib

matplotlib.use("Agg")

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
OUT = HERE / "framework_pipeline_minimal.png"
META_PATH = HERE / "graph_metadata.json"

# --- canvas -------------------------------------------------------------------
FIG_W, FIG_H, DPI = 12.0, 4.0, 300
XMAX, YMAX = 120.0, 40.0  # 10 data units per inch

# --- palette (minimalist flat design with #E2E8F0 borders & phase accents) ----
INK = "#0F172A"          # primary titles
BODY = "#1E293B"         # badge primary labels & inter-block arrows
MUTED = "#475569"        # secondary technical details
SUBTLE = "#64748B"       # micro-connectors & footnote
CANVAS = "#FFFFFF"
CARD_FILL = "#FFFFFF"
CARD_EDGE = "#E2E8F0"    # spec border
CARD_SHADOW = "#F1F5F9"  # subtle flat elevation
DIVIDER = "#E2E8F0"

# Phase-coded top accent rules + whisper-soft tinted badge surfaces
STAGE_STYLES: list[dict[str, str]] = [
    {   # 1. Data & Split Layer (Cool Slate)
        "accent": "#475569",
        "badge_bg": "#F8FAFC",
        "badge_edge": "#E2E8F0",
    },
    {   # 2. HeteroData Graph (Ocean Cyan)
        "accent": "#0284C7",
        "badge_bg": "#F4FAFE",
        "badge_edge": "#E2E8F0",
    },
    {   # 3. HGT Backbone & Residual Skip (Royal Indigo)
        "accent": "#4F46E5",
        "badge_bg": "#F6F7FF",
        "badge_edge": "#E2E8F0",
    },
    {   # 4. Multi-Label Head (Deep Iris)
        "accent": "#7C3AED",
        "badge_bg": "#FAF7FF",
        "badge_edge": "#E2E8F0",
    },
    {   # 5. Decision & Inference (Emerald Teal)
        "accent": "#0D9488",
        "badge_bg": "#F3FCFA",
        "badge_edge": "#E2E8F0",
    },
]

# --- tight layout: 5 wide blocks, compact arrow gaps, minimal outer margins ---
MARGIN_X = 1.4
GAP = 2.8
N_BLOCKS = 5
BLOCK_W = (XMAX - 2.0 * MARGIN_X - (N_BLOCKS - 1) * GAP) / N_BLOCKS  # 21.20 units

CARD_Y0, CARD_Y1 = 3.1, 38.6   # 35.5 units tall (~87% canvas coverage)
HEADER_H = 5.4
BADGE_INSET = 0.65
BADGE_GAP = 1.95
BADGE_LINE = 3.10
BADGE_PAD_Y = 0.85

CARD_EDGE_LW = 1.05
BADGE_EDGE_LW = 0.9
ARROW_LW = 1.5

HEADER_SIZE = 10.4
BADGE_SIZE = 10.2
BADGE_MIN_SIZE = 6.8
NOTE_SIZE = 7.4


def _load_graph_summary() -> str:
    """Read live counts from graph_metadata.json if present."""
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
            total_nodes = sum(int(v) for v in meta["node_types"].values())
            cutoff_date = str(meta["cutoff"]).split()[0]
            return f"{total_nodes:,} nodes ({cutoff_date})"
        except Exception:
            pass
    return "1,070,946 nodes (2026-11-14)"


# Each badge line is (text, is_code_style)
BadgeLine = tuple[str, bool]

BLOCKS: list[tuple[str, list[tuple[BadgeLine, ...]]]] = [
    (
        "1. Data & Split Layer",
        [
            (("Raw GH Archive", False), ("hourly .json.gz stream", True)),
            (("Coverage Mask", False), ("verify_raw_data.py", True)),
            (("Strict Temporal Cutoff", False), ("rows keep t < t_cutoff", True)),
        ],
    ),
    (
        "2. HeteroData Graph",
        [
            (("8 Entity Nodes", False), (_load_graph_summary(), True)),
            (("26 Directed Edge Types", False), ("13 forward + 13 rev_*", True)),
        ],
    ),
    (
        "3. HGT Backbone & Residual Skip",
        [
            (("NodeMLPEncoder", False), ("Linear → LN → Mish (×4)", False)),
            (("1-Layer HGTConv", False), ("4 heads + residual LN", False)),
            (("Direct Residual Concat", False), ("128-d latent ⊕ 21-d raw", True)),
        ],
    ),
    (
        "4. Multi-Label Head",
        [
            (("6 Task-Specific MLP Heads", False), ("2×(Mish+LN) → 1 logit", False)),
            (("Masked BCEWithLogits", False), ("auto pos_weight", True)),
        ],
    ),
    (
        "5. Decision & Inference",
        [
            (("Thresholds (θ) per Activity", False), ("CMA-ES (val) · F1-sweep (final)", False)),
            (("scripts/evaluate.py", True), ("schema check · F1 score", False)),
            (("submission/submission.csv", True), ("output_{studentID}.csv", True)),
        ],
    ),
]

INTRA_CONNECTORS: list[list[str]] = [
    ["down", "down"],
    ["bi"],
    ["down", "down"],
    ["down"],
    ["down", "down"],
]

FOOTNOTE = (
    "code paths: scripts/get_raw_data.py · scripts/verify_raw_data.py · "
    "src/data/temporal_split.py · src/features/build_graph.py · "
    "src/models/hetero_model.py · src/models/train.py · scripts/evaluate.py · "
    "graph counts: figures/slide 10/graph_metadata.json"
)

Box = tuple[float, float, float, float]


class Probe(TypedDict):
    kind: str
    text: Text
    box: Box | None
    block: int
    badge: int


def block_x(i: int) -> float:
    return MARGIN_X + i * (BLOCK_W + GAP)


def badge_height(lines: int, extra_pad: float = 0.0) -> float:
    return (BADGE_PAD_Y + extra_pad) * 2.0 + BADGE_LINE * lines


def badge_stack(card_x: float, lines_per_badge: Sequence[int]) -> list[tuple[float, float]]:
    # Give 2-badge cards slightly taller badges & gap so they fill the card body harmoniously
    n_badges = len(lines_per_badge)
    extra_pad = 0.90 if n_badges == 2 else 0.0
    gap = BADGE_GAP + (0.95 if n_badges == 2 else 0.0)

    heights = [badge_height(n, extra_pad=extra_pad) for n in lines_per_badge]
    total = sum(heights) + gap * (n_badges - 1)
    body_bottom = CARD_Y0 + 0.75
    body_top = CARD_Y1 - HEADER_H - 0.65
    available = body_top - body_bottom
    if total > available:
        raise SystemExit(
            f"badges need {total:.2f} units but the card body has {available:.2f}"
        )
    top = body_bottom + (available + total) / 2.0
    boxes: list[tuple[float, float]] = []
    for height in heights:
        y1 = top if not boxes else boxes[-1][0] - gap
        boxes.append((y1 - height, y1))
    return boxes


def measure(renderer: RendererBase, text: Text) -> tuple[float, float, float, float]:
    box = text.get_window_extent(renderer=renderer)
    return box.x0, box.y0, box.x1, box.y1


def fit_lines(renderer: RendererBase, text: Text, limit_px: float, min_size: float) -> float:
    size = float(text.get_fontsize())
    while size > min_size:
        x0, _, x1, _ = measure(renderer, text)
        if x1 - x0 <= limit_px:
            break
        size -= 0.1
        text.set_fontsize(size)
    return size


def card(
    renderer: RendererBase, ax, x0: float, label: str, style: dict[str, str]
) -> tuple[Text, float]:
    card_h = CARD_Y1 - CARD_Y0

    # 1. Whisper-soft flat elevation shadow
    ax.add_patch(FancyBboxPatch(
        (x0 + 0.22, CARD_Y0 - 0.22), BLOCK_W, card_h,
        boxstyle="round,pad=0,rounding_size=0.95",
        facecolor=CARD_SHADOW, edgecolor="none", zorder=1,
    ))

    # 2. Main card body with crisp #E2E8F0 border
    ax.add_patch(FancyBboxPatch(
        (x0, CARD_Y0), BLOCK_W, card_h,
        boxstyle="round,pad=0,rounding_size=0.95",
        facecolor=CARD_FILL, edgecolor=CARD_EDGE, linewidth=CARD_EDGE_LW, zorder=2,
    ))

    # 3. Phase-coded top accent bar inset inside top corners
    accent_inset = 0.75
    accent_h = 0.48
    ax.add_patch(FancyBboxPatch(
        (x0 + accent_inset, CARD_Y1 - accent_h - 0.12),
        BLOCK_W - 2.0 * accent_inset,
        accent_h,
        boxstyle="round,pad=0,rounding_size=0.18",
        facecolor=style["accent"],
        edgecolor="none",
        zorder=3,
    ))

    # 4. Crisp hairline header divider
    header_div_y = CARD_Y1 - HEADER_H
    ax.add_line(Line2D(
        [x0 + 0.7, x0 + BLOCK_W - 0.7],
        [header_div_y, header_div_y],
        color=DIVIDER,
        linewidth=0.85,
        zorder=3,
    ))

    # 5. Bold high-contrast stage header
    text = ax.text(
        x0 + BLOCK_W / 2.0,
        CARD_Y1 - HEADER_H / 2.0 - 0.18,
        label,
        ha="center",
        va="center",
        fontsize=HEADER_SIZE,
        fontweight="bold",
        color=INK,
        zorder=4,
    )
    header_limit_px = (BLOCK_W - 1.1) / XMAX * FIG_W * 72.0
    fit_lines(renderer, text, header_limit_px, HEADER_SIZE - 2.6)
    badge_limit_px = (BLOCK_W - 2.0 * BADGE_INSET) / XMAX * FIG_W * 72.0
    return text, badge_limit_px


def badge(
    renderer: RendererBase,
    ax,
    x0: float,
    y0: float,
    y1: float,
    lines: tuple[BadgeLine, ...],
    limit_px: float,
    style: dict[str, str],
) -> list[Text]:
    ax.add_patch(FancyBboxPatch(
        (x0 + BADGE_INSET, y0),
        BLOCK_W - 2.0 * BADGE_INSET,
        y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.72",
        facecolor=style["badge_bg"],
        edgecolor=style["badge_edge"],
        linewidth=BADGE_EDGE_LW,
        zorder=3,
    ))
    centre = y0 + (y1 - y0) / 2.0
    span = BADGE_LINE * (len(lines) - 1)
    artists: list[Text] = []
    for i, (line_text, is_code) in enumerate(lines):
        y = centre + span / 2.0 - BADGE_LINE * i
        is_primary = i == 0
        size = BADGE_SIZE if is_primary else BADGE_SIZE - 1.35
        if is_code and not is_primary:
            size -= 0.25
        color = BODY if is_primary else MUTED
        weight = "semibold" if is_primary else "normal"
        family = "DejaVu Sans Mono" if is_code else "DejaVu Sans"

        text = ax.text(
            x0 + BLOCK_W / 2.0,
            y,
            line_text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            fontfamily=family,
            color=color,
            zorder=4,
        )
        fit_lines(renderer, text, limit_px - 3.0, BADGE_MIN_SIZE)
        artists.append(text)
    return artists


def intra_connectors(
    ax, x0: float, stack: list[tuple[float, float]], modes: Sequence[str], style: dict[str, str]
) -> None:
    """Draw clean micro-arrows between stacked sub-badges inside a card."""
    cx = x0 + BLOCK_W / 2.0
    pad = 0.24
    for j, mode in enumerate(modes):
        top_y = stack[j][0] - pad
        bot_y = stack[j + 1][1] + pad
        if top_y <= bot_y:
            continue
        arrowstyle = (
            "<->,head_length=2.8,head_width=2.0"
            if mode == "bi"
            else "-|>,head_length=3.0,head_width=2.0"
        )
        ax.add_patch(FancyArrowPatch(
            (cx, top_y),
            (cx, bot_y),
            arrowstyle=arrowstyle,
            linewidth=1.0,
            color=style["accent"] if mode == "bi" else SUBTLE,
            alpha=0.88,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=4,
        ))


def arrows(ax, blocks_x: Sequence[float]) -> None:
    """Crisp dark-slate directional arrows in the gaps between blocks."""
    mid = (CARD_Y0 + CARD_Y1) / 2.0
    for i in range(len(blocks_x) - 1):
        x_start = blocks_x[i] + BLOCK_W + 0.38
        x_end = blocks_x[i + 1] - 0.38
        ax.add_patch(FancyArrowPatch(
            (x_start, mid),
            (x_end, mid),
            arrowstyle="-|>,head_length=4.5,head_width=2.9",
            linewidth=ARROW_LW,
            color=BODY,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        ))


def build() -> tuple[Figure, list[Probe]]:
    fig = Figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(CANVAS)
    agg = FigureCanvasAgg(fig)
    canvas = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    canvas.set_xlim(0.0, XMAX)
    canvas.set_ylim(0.0, YMAX)
    canvas.set_aspect("equal")
    canvas.set_axis_off()
    agg.draw()
    renderer = agg.get_renderer()

    xs = [block_x(i) for i in range(N_BLOCKS)]
    probes: list[Probe] = []

    for i, (label, badges) in enumerate(BLOCKS):
        x0 = xs[i]
        style = STAGE_STYLES[i]
        header, badge_limit = card(renderer, canvas, x0, label, style)
        probes.append({
            "kind": "header",
            "block": i,
            "badge": -1,
            "text": header,
            "box": (x0 + 0.45, CARD_Y1 - HEADER_H, x0 + BLOCK_W - 0.45, CARD_Y1),
        })
        line_counts = [len(b) for b in badges]
        stack = badge_stack(x0, line_counts)
        for j, ((y0, y1), lines) in enumerate(zip(stack, badges, strict=True)):
            artists = badge(renderer, canvas, x0, y0, y1, lines, badge_limit, style)
            for artist in artists:
                probes.append({
                    "kind": "badge",
                    "block": i,
                    "badge": j,
                    "text": artist,
                    "box": (x0 + BADGE_INSET, y0, x0 + BLOCK_W - BADGE_INSET, y1),
                })
        intra_connectors(canvas, x0, stack, INTRA_CONNECTORS[i], style)

    arrows(canvas, xs)

    note = fig.text(
        0.5,
        0.025,
        FOOTNOTE,
        ha="center",
        va="center",
        fontsize=NOTE_SIZE,
        fontfamily="DejaVu Sans",
        color=SUBTLE,
    )
    fit_lines(renderer, note, 0.97 * FIG_W * 72.0, NOTE_SIZE - 1.2)
    probes.append({
        "kind": "footnote",
        "block": -1,
        "badge": -1,
        "text": note,
        "box": None,
    })

    return fig, probes


def main() -> None:
    fig, _probes = build()
    fig.savefig(OUT, dpi=DPI, facecolor=fig.get_facecolor())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()