#!/usr/bin/env python3
"""Layout audit for survival_vs_sinusoidal_minimal.png (the slide 7 figure).

Every Text artist is measured through the Agg renderer and checked for

  a. leaving the canvas,
  b. overlapping another text box,
  c. the provenance line spilling past the width of the axes,
  d. the legend spilling outside the axes box.

Run:  python "figures/slide 7/check_layout.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_survival_vs_sinusoidal_minimal import (  # noqa: E402
    AX_RECT,
    DPI,
    FIG_H,
    FIG_W,
    build,
)

TOLERANCE = 0.5  # px of slack at the canvas edge
Box = tuple[str, float, float, float, float, bool]  # text, x0, y0, x1, y1, figure_level


def renderer_for(fig: Figure) -> RendererBase:
    """A drawn Agg renderer, so legend and label positions are final."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def text_boxes(fig: Figure, renderer: RendererBase) -> list[Box]:
    """(text, x0, y0, x1, y1, figure_level) for every visible Text artist, in pixels.

    figure_level is True only for text placed in figure coordinates (the
    provenance line), which is what the footnote-band rules apply to.
    """
    boxes = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible():
            continue
        label = artist.get_text().strip()
        if not label:
            continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0.0 or box.height <= 0.0:
            continue
        figure_level = artist.get_transform() is fig.transFigure
        boxes.append((label.replace("\n", " "), box.x0, box.y0, box.x1, box.y1, figure_level))
    return boxes


def overlap(a: Box, b: Box) -> float:
    """Intersection area of two boxes; 0 when they merely touch."""
    dx = min(a[3], b[3]) - max(a[1], b[1])
    dy = min(a[4], b[4]) - max(a[2], b[2])
    return dx * dy if dx > 0.0 and dy > 0.0 else 0.0


def area(box: Box) -> float:
    return max(box[3] - box[1], 0.0) * max(box[4] - box[2], 0.0)


def main() -> int:
    fig = build()
    renderer = renderer_for(fig)
    width = FIG_W * DPI
    height = FIG_H * DPI
    boxes = text_boxes(fig, renderer)

    problems = 0
    for label, x0, y0, x1, y1, _figure_level in boxes:
        if x0 < -TOLERANCE or y0 < -TOLERANCE or x1 > width + TOLERANCE or y1 > height + TOLERANCE:
            problems += 1
            print(f"  off-canvas  {label!r}  box=({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")

    # c/d: the bands that figure-level annotations must stay inside
    axes_left, axes_bottom = AX_RECT[0] * width, AX_RECT[1] * height
    axes_right, axes_top = (AX_RECT[0] + AX_RECT[2]) * width, (AX_RECT[1] + AX_RECT[3]) * height
    for label, x0, y0, x1, y1, figure_level in boxes:
        if not figure_level:
            continue
        if y1 < axes_bottom and (x0 < axes_left - TOLERANCE or x1 > axes_right + TOLERANCE):
            problems += 1
            print(f"  footnote spills the axes width  {label!r}  ({x0:.0f}..{x1:.0f})")
        if y0 > axes_top:
            problems += 1
            print(f"  above the axes  {label!r}  ({y0:.0f}..{y1:.0f})")

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            cross = overlap(boxes[i], boxes[j])
            if cross <= 1.0:
                continue
            smaller = min(area(boxes[i]), area(boxes[j]))
            if cross / max(smaller, 1.0) < 0.04:
                continue
            problems += 1
            print(f"  overlap {cross:.0f}px  {boxes[i][0]!r} <> {boxes[j][0]!r}")

    print(f"canvas {width:.0f}x{height:.0f}px · measured {len(boxes)} text artists · problems: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
