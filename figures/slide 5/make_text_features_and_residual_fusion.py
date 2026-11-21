#!/usr/bin/env python3
"""Slide 5 figure: deterministic text markers + direct residual fusion (publication figure).

Produces ``figures/slide 5/text_features_and_residual_fusion.png`` — a two-panel diagram:

  panel 1 (left)   deterministic textual & structural markers, no NLP: a masked raw string
                   and the three flags/counts extracted from it by character rules;
  panel 2 (right)  the residual fusion path of GitHubHeteroHGT: Users.x splits into the
                   MLP+HGT branch and an identity highway over it, recombined by concat.

Every dimension quoted here is read off the implementation, not invented:

  Users.x = 21 dims ................ src/features/build_graph.py:23-26
      6 activities x (days_since, has_ever, count_7d) + is_bot + is_cohort + user_age_days
  latent = 128 ..................... scripts/run_training.py:292 (--latent-dim default)
  z_user = 128 + 21 = 149 .......... src/models/hetero_model.py:124, 152-153
      combined_dim = latent_dim + self.user_raw_dim
      user_combined = torch.cat([user_gnn_emb, raw_user_features], dim=-1)
  6 heads ......................... src/models/hetero_model.py:127-129 (num_classes=6)
      each one is a PredictionHead, whose own docstring says "Task-specific MLP prediction
      head" (hetero_model.py:47-66): Linear(149→hidden) → LayerNorm → Mish →
      Linear(hidden→hidden) → LayerNorm → Mish → Linear(hidden→1).  It is NOT a single
      linear layer, so the card text says MLP Heads.
  text markers .................... src/features/build_graph.py
      iss_has_prefix :148  ("scope word, optional (scope) or !, then a colon")
      iss_has_md     :143  (markdown body markers)
      pr_branch_depth :128 (head_ref slash count)   prc_path_depth :178 (path slash count)

Card text is set from measured glyph widths (``fits()``): any label wider than its card
is shrunk until it clears the padding, so nothing spills over a box edge.

Run:  python "figures/slide 5/make_text_features_and_residual_fusion.py"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent  # figures/slide 5

# --------------------------------------------------------------------------- theme --
INK = "#0f172a"  # near-black text
BODY = "#334155"  # regular text
MUTED = "#64748b"  # captions / provenance
CANVAS = "#ffffff"  # figure background
PANEL = "#f1f5f9"  # panel fill (the specified light border tone)
PANEL_EDGE = "#cbd5e1"  # panel outline
CARD_EDGE = "#cbd5e1"

BLUE = "#2563eb"
BLUE_BG = "#dbeafe"
AMBER = "#b45309"
AMBER_BG = "#fef3c7"
TEAL = "#0f766e"
TEAL_BG = "#ccf1ec"
SLATE = "#475569"
SLATE_BG = "#e2e8f0"

FIG_W, FIG_H, DPI = 14.0, 6.0, 300

# Canvas coordinate system: 10 units per inch on both axes, so patches stay square.
XMAX, YMAX = FIG_W * 10, FIG_H * 10  # 140 x 60

LEFT_PANEL = (1.6, 2.6, 55.4, 57.4)  # x0, y0, x1, y1
RIGHT_PANEL = (59.4, 2.6, 138.4, 57.4)


# ------------------------------------------------------------------------- helpers --
def panel(ax, x0: float, y0: float, x1: float, y1: float, num: str, title: str) -> None:
    """Panel card with a numbered header."""
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=1.4",
            facecolor=PANEL,
            edgecolor=PANEL_EDGE,
            linewidth=0.9,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 1.5, y1 - 5.0),
            3.1,
            3.1,
            boxstyle="round,pad=0,rounding_size=0.7",
            facecolor=INK,
            edgecolor="none",
            zorder=3,
        )
    )
    ax.text(
        x0 + 3.05,
        y1 - 3.45,
        num,
        fontsize=8.8,
        color=CANVAS,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=4,
    )
    ax.text(
        x0 + 5.8,
        y1 - 2.95,
        title,
        fontsize=10.6,
        color=INK,
        fontweight="bold",
        va="center",
        zorder=4,
    )


def card(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    face: str = CANVAS,
    edge: str = CARD_EDGE,
    accent: str | None = None,
    rounding: float = 0.9,
    lw: float = 0.9,
    z: int = 2,
) -> None:
    """Rounded card; ``accent`` draws a solid rule down its left edge."""
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            zorder=z,
        )
    )
    if accent is not None:
        ax.add_artist(
            Line2D(
                [x0 + 0.8, x0 + 0.8],
                [y0 + 1.0, y1 - 1.0],
                color=accent,
                lw=2.8,
                solid_capstyle="butt",
                zorder=z + 1,
            )
        )


def arrow(
    ax,
    tail: tuple[float, float],
    head: tuple[float, float],
    *,
    color: str = SLATE,
    lw: float = 1.3,
    scale: float = 9.5,
    z: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            tail,
            head,
            arrowstyle="-|>",
            mutation_scale=scale,
            shrinkA=0,
            shrinkB=0,
            color=color,
            linewidth=lw,
            connectionstyle="arc3",
            zorder=z,
        )
    )


def segment(
    ax, pts: list[tuple[float, float]], *, color: str, lw: float = 2.2, ls: Any = "-", z: int = 4
) -> None:
    """Poly-line used for the identity highway and routed bus lines."""
    ax.add_artist(
        Line2D(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            lw=lw,
            linestyle=ls,
            solid_capstyle="round",
            zorder=z,
        )
    )


def label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    min_size: float = 5.6,
    color: str = BODY,
    weight: str = "normal",
    family: str = "sans-serif",
    ha: str = "left",
    va: str = "center",
    rotation: float = 0.0,
    max_units: float | None = None,
    z: int = 5,
) -> float:
    """Draw one text run, shrinking it until it fits ``max_units`` of card width.

    Returns the size actually used, so a caller can keep a label and its sub-label on
    the same footing instead of one silently overflowing its box.
    """
    txt = ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        family=family,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=z,
    )
    if max_units is None or rotation != 0.0:
        return size
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    size = min_size if size < min_size else size
    while size > min_size:
        bbox = txt.get_window_extent(renderer=renderer)
        x_lo, _ = inv.transform((bbox.x0, bbox.y0))
        x_hi, _ = inv.transform((bbox.x1, bbox.y1))
        if x_hi - x_lo <= max_units:
            break
        size -= 0.15
        txt.set_fontsize(size)
    return size


def mono_string(ax, text: str, cx: float, cy: float, *, fontsize: float, hi: dict[str, str]) -> None:
    """Draw ``text`` centred at (cx, cy) in monospace, one glyph per measured cell.

    Cell pitch is measured from the rendered probe string, so placement stays exact for
    whatever monospace family the environment resolves; ``hi`` maps a character to a
    highlight background colour (those glyphs take bold weight).
    """
    probe = ax.text(
        cx,
        cy,
        text,
        fontsize=fontsize,
        family="monospace",
        ha="center",
        va="center",
        color=INK,
        zorder=6,
    )
    fig = ax.figure
    fig.canvas.draw()
    bbox = probe.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    x_lo, _ = inv.transform((bbox.x0, bbox.y0))
    x_hi, _ = inv.transform((bbox.x1, bbox.y1))
    probe.remove()

    pitch = (x_hi - x_lo) / len(text)
    start = cx - pitch * len(text) / 2.0
    for i, ch in enumerate(text):
        gx = start + pitch * (i + 0.5)
        if ch in hi:
            ax.add_patch(
                Rectangle(
                    (gx - pitch / 2, cy - 2.35),
                    pitch,
                    4.7,
                    facecolor=hi[ch],
                    edgecolor="none",
                    zorder=5,
                )
            )
        ax.text(
            gx,
            cy,
            ch,
            fontsize=fontsize,
            family="monospace",
            ha="center",
            va="center",
            color=INK,
            fontweight="bold" if ch in hi else "normal",
            zorder=6,
        )


# ------------------------------------------------------------------- panel 1 (left) --
def draw_text_markers(ax) -> None:
    x0, y0, x1, y1 = LEFT_PANEL
    panel(ax, x0, y0, x1, y1, "1", "Deterministic Textual & Structural Markers — No NLP")

    # ------------------------------------------------------- masked raw-string card --
    c_x0, c_y0, c_x1, c_y1 = 3.6, 40.2, 53.4, 51.8
    card(ax, c_x0, c_y0, c_x1, c_y1, face=CANVAS)
    label(
        ax,
        c_x0 + 1.8,
        c_y1 - 1.8,
        "raw title / body text, character-masked",
        size=7.4,
        color=MUTED,
        va="top",
        max_units=c_x1 - c_x0 - 3.6,
    )
    mono_string(
        ax,
        "aaaa: aaaaa (aaa-00) # aaaa",
        (c_x0 + c_x1) / 2,
        (c_y0 + c_y1) / 2 - 1.2,
        fontsize=16.5,
        hi={":": BLUE_BG, "(": BLUE_BG, ")": BLUE_BG, "#": AMBER_BG},
    )

    # ---------------------------------------------------------------- extraction bus --
    # one spine, one arrow per marker: the fan-out stays orthogonal instead of crossing
    spine_x = 7.2
    badges = [
        (BLUE, "Conventional Prefix Flag", 'Detected  ":"  with optional "()" or "!"'),
        (AMBER, "Markdown Template Flag", 'Detected  heading "#"  and  code "`"'),
        (TEAL, "Hierarchy Slash Depth", 'Branch & File Path slashes  "/"'),
    ]
    b_x0, b_x1, bh, gap = 11.2, 53.4, 9.2, 2.4
    top = 37.6
    mid_y = [top - i * (bh + gap) - bh / 2 for i in range(len(badges))]
    segment(ax, [(spine_x, c_y0), (spine_x, mid_y[-1])], color=SLATE, lw=1.5)
    label(
        ax,
        3.9,
        24.6,
        "character / regex extraction",
        size=7.4,
        color=MUTED,
        rotation=90,
        ha="center",
        va="center",
    )

    for i, (accent, title, detail) in enumerate(badges):
        by1 = top - i * (bh + gap)
        by0 = by1 - bh
        card(ax, b_x0, by0, b_x1, by1, face=CANVAS, accent=accent)
        label(
            ax,
            b_x0 + 2.5,
            by1 - 2.8,
            title,
            size=10.4,
            min_size=7.6,
            color=INK,
            weight="bold",
            max_units=b_x1 - b_x0 - 4.6,
        )
        label(
            ax,
            b_x0 + 2.5,
            by0 + 2.9,
            detail,
            size=9.6,
            min_size=7.0,
            color=BODY,
            max_units=b_x1 - b_x0 - 4.6,
        )
        arrow(ax, (spine_x, mid_y[i]), (b_x0 - 0.35, mid_y[i]), color=SLATE, lw=1.4)

    label(
        ax,
        3.6,
        3.5,
        "detectors: build_graph.py — iss_has_prefix :148 · iss_has_md :143 · pr_branch_depth :128",
        size=6.6,
        color=MUTED,
        va="bottom",
    )


# ------------------------------------------------------------------ panel 2 (right) --
def draw_residual_fusion(ax) -> None:
    x0, y0, x1, y1 = RIGHT_PANEL
    panel(ax, x0, y0, x1, y1, "2", "Direct Residual Fusion Architecture")

    # -------------------------------------------------------------- Users.x source --
    u_x0, u_y0, u_x1, u_y1 = 61.4, 16.6, 79.6, 40.6
    card(ax, u_x0, u_y0, u_x1, u_y1, face=SLATE_BG, accent=SLATE, lw=1.1)
    label(
        ax,
        u_x0 + 2.4,
        u_y1 - 2.8,
        "Users.x",
        size=12.0,
        color=INK,
        weight="bold",
        max_units=u_x1 - u_x0 - 4.4,
    )
    label(
        ax,
        u_x0 + 2.4,
        u_y1 - 6.6,
        "21-dim Raw Features",
        size=9.6,
        color=BODY,
        max_units=u_x1 - u_x0 - 4.4,
    )
    label(
        ax,
        u_x0 + 2.4,
        u_y0 + 2.8,
        "recency · counts · profile",
        size=7.4,
        color=MUTED,
        max_units=u_x1 - u_x0 - 4.4,
    )

    split = (u_x1 + 1.2, (u_y0 + u_y1) / 2)  # 80.8, 28.6

    # --------------------------------------------- Path B: identity highway (upper) --
    hy_y, hy_x1 = 45.4, 132.9
    segment(
        ax,
        [(u_x1, split[1]), (split[0], split[1]), (split[0], hy_y), (hy_x1, hy_y)],
        color=TEAL,
        lw=2.4,
    )
    ax.add_patch(
        FancyArrowPatch(
            (hy_x1, hy_y),
            (hy_x1, 40.0),
            arrowstyle="-|>",
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
            color=TEAL,
            linewidth=2.4,
            zorder=4,
        )
    )
    label(
        ax,
        107.0,
        hy_y + 3.2,
        "Identity Highway: Preserves Raw Tabular Recency",
        size=9.4,
        color=TEAL,
        weight="bold",
        ha="center",
        max_units=66.0,
    )
    label(
        ax,
        107.0,
        hy_y - 3.0,
        "bypasses the encoder and the HGT backbone",
        size=7.6,
        color=MUTED,
        ha="center",
        max_units=66.0,
    )

    # ------------------------------------------------- Path A: encoder + backbone --
    a_y0, a_y1 = 12.2, 33.8
    a_cy = (a_y0 + a_y1) / 2
    arrow(ax, split, (84.0, a_cy), color=SLATE, lw=1.7)
    card(ax, 84.0, a_y0, 100.0, a_y1, face=CANVAS, accent=BLUE)
    label(
        ax,
        86.0,
        a_cy + 2.6,
        "NodeMLPEncoder",
        size=9.6,
        min_size=6.0,
        color=INK,
        weight="bold",
        max_units=100.0 - 86.0 - 1.4,
    )
    label(
        ax,
        86.0,
        a_cy - 3.4,
        "21 → 128 latent",
        size=7.8,
        color=MUTED,
        max_units=100.0 - 86.0 - 1.4,
    )

    arrow(ax, (100.0, a_cy), (102.8, a_cy), color=SLATE, lw=1.7)
    card(ax, 102.8, a_y0, 116.4, a_y1, face=CANVAS, accent=BLUE)
    label(
        ax,
        104.8,
        a_cy + 2.6,
        "HGTConv",
        size=10.4,
        color=INK,
        weight="bold",
        max_units=116.4 - 104.8 - 1.4,
    )
    label(
        ax,
        104.8,
        a_cy - 3.4,
        "Backbone",
        size=10.4,
        color=INK,
        weight="bold",
        max_units=116.4 - 104.8 - 1.4,
    )

    arrow(ax, (116.4, a_cy), (118.4, a_cy), color=BLUE, lw=1.8)
    card(ax, 118.6, 18.0, 127.0, 28.0, face=BLUE_BG, edge=BLUE, rounding=0.8)
    label(
        ax,
        122.8,
        25.0,
        "h_user",
        size=10.0,
        color=INK,
        weight="bold",
        ha="center",
        max_units=127.0 - 118.6 - 1.6,
    )
    label(
        ax,
        122.8,
        21.2,
        "128-dim",
        size=8.4,
        color=BODY,
        ha="center",
        max_units=127.0 - 118.6 - 1.6,
    )
    arrow(ax, (127.0, a_cy), (128.6, a_cy), color=BLUE, lw=1.8)

    label(
        ax,
        98.0,
        a_y0 - 3.0,
        "message passing over 13 relation types",
        size=8.0,
        color=BLUE,
        weight="bold",
        ha="center",
        max_units=30.0,
    )

    # ---------------------------------------------------------------- fusion block --
    f_x0, f_y0, f_x1, f_y1 = 128.8, 12.4, 137.0, 40.0
    f_cx = (f_x0 + f_x1) / 2
    f_w = f_x1 - f_x0 - 1.6
    card(ax, f_x0, f_y0, f_x1, f_y1, face=CANVAS, accent=INK, lw=1.1, rounding=1.0)
    label(ax, f_cx, f_y1 - 3.2, "Fusion", size=10.0, color=INK, weight="bold", ha="center", max_units=f_w)
    label(ax, f_cx, f_y1 - 7.6, "Concat", size=9.0, color=BODY, ha="center", max_units=f_w)
    label(
        ax,
        f_cx,
        f_y1 - 12.2,
        "[h_user",
        size=8.6,
        color=INK,
        family="monospace",
        ha="center",
        max_units=f_w,
    )
    label(
        ax,
        f_cx,
        f_y1 - 15.2,
        "‖",
        size=8.0,
        color=MUTED,
        ha="center",
        max_units=f_w,
    )
    label(
        ax,
        f_cx,
        f_y1 - 18.2,
        "Users.x]",
        size=8.6,
        color=INK,
        family="monospace",
        ha="center",
        max_units=f_w,
    )
    label(
        ax,
        f_cx,
        f_y1 - 22.2,
        "z_user",
        size=9.2,
        color=INK,
        weight="bold",
        ha="center",
        max_units=f_w,
    )
    label(
        ax,
        f_cx,
        f_y1 - 25.2,
        "149-dim",
        size=7.8,
        color=BODY,
        ha="center",
        max_units=f_w,
    )

    # ------------------------------------------------------------------- 6 heads --
    h_x0, h_y0, h_x1, h_y1 = 114.5, 3.8, 137.0, 11.4
    arrow(ax, (f_cx, f_y0), (f_cx, h_y1), color=INK, lw=1.8)
    card(ax, h_x0, h_y0, h_x1, h_y1, face=SLATE_BG, edge="#94a3b8", accent=INK, rounding=1.0)
    label(
        ax,
        (h_x0 + h_x1) / 2 + 0.5,
        h_y1 - 2.9,
        "6 Task-Specific MLP Heads",
        size=9.2,
        color=INK,
        weight="bold",
        ha="center",
        max_units=h_x1 - h_x0 - 4.4,
    )
    label(
        ax,
        (h_x0 + h_x1) / 2 + 0.5,
        h_y0 + 2.9,
        "open_pr · open_issue · issue_comment",
        size=6.6,
        color=BODY,
        ha="center",
        max_units=h_x1 - h_x0 - 4.4,
    )
    label(
        ax,
        (h_x0 + h_x1) / 2 + 0.5,
        h_y0 + 1.4,
        "pr_review · pr_review_comment · push",
        size=6.6,
        color=BODY,
        ha="center",
        max_units=h_x1 - h_x0 - 4.4,
    )

    label(
        ax,
        x0 + 2.0,
        3.5,
        "hetero_model.py — combined_dim :124 · torch.cat([user_gnn_emb, raw_user_features]) :153",
        size=6.6,
        color=MUTED,
        va="bottom",
    )


def build() -> Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(CANVAS)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    ax.set_facecolor(CANVAS)

    draw_text_markers(ax)
    draw_residual_fusion(ax)
    return fig


def main() -> None:
    fig = build()
    out = HERE / "text_features_and_residual_fusion.png"
    fig.savefig(out, dpi=DPI, facecolor=CANVAS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
