"""Layout proof for the slide 11 bar chart.

The figure script returns, for every text artist it draws, the pixel box that
artist must live inside (a category cell under the axis, the column above a bar,
the bracket band, the footnote band).  This checker redraws the figure, measures
real pixel extents, and fails on:

* anything drawn outside the canvas;
* text escaping its declared box;
* any two text artists overlapping by more than 2 % of the smaller one (tick
  labels included, because a value label colliding with a y tick is exactly the
  failure mode a bar chart has);
* a text artist sitting on a drawn stroke (the bracket's drop lines, the error
  whiskers and caps) - nothing there is text, so the overlap rules miss it.

Exits non-zero when a problem is found, so it can gate a build.

    conda run -n bda_asg2 python "figures/slide 11/check_layout.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg

HERE = Path(__file__).resolve().parent  # figures/slide 11
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_pos_weights_calibration_minimal import (  # noqa: E402  # pyright: ignore[reportMissingImports]  # sibling module in this folder; pyright's index predates it
    DPI,
    FIG_H,
    FIG_W,
    build,
)

TOLERANCE = 0.6  # px of slack allowed against a box edge
OVERLAP_FRACTION = 0.02  # share of the smaller box before an overlap is a problem
MIN_FONT = 4.6  # below this the text is unreadable on a slide, whatever fits


def overlap(a: list[float], b: list[float]) -> float:
    """Fraction of the smaller box covered by the intersection of two boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    area = width * height
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return area / smaller if smaller > 0 else 0.0


def main() -> int:
    fig, probes, strokes = build()  # build() attaches an Agg canvas and draws
    # re-draw and measure against that renderer explicitly: an artist caches its
    # last-draw extent, and measuring a stale cache is how a label that clearly
    # sits over its bar reports x = 0
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    width_px, height_px = int(FIG_W * DPI), int(FIG_H * DPI)

    texts: list[tuple[str, list[float]]] = []
    problems: list[str] = []
    sizes: list[tuple[str, float, str]] = []

    for probe in probes:
        text = probe["text"]
        label = text.get_text()
        if not label.strip():
            continue
        extent = text.get_window_extent(renderer=renderer)
        box = [extent.x0, extent.y0, extent.x1, extent.y1]
        kind = str(probe["kind"])
        texts.append((f"{kind}: {label}", box))
        sizes.append((kind, float(text.get_fontsize()), label))

        if box[0] < -TOLERANCE or box[1] < -TOLERANCE or box[2] > width_px + TOLERANCE:
            problems.append(
                f"{kind}: '{label}' leaves the canvas "
                f"({box[0]:.1f}..{box[2]:.1f} x {box[1]:.1f}..{box[3]:.1f} "
                f"vs canvas {width_px}x{height_px})"
            )
        declared = probe["box"]
        if declared is not None:
            x0, y0, x1, y1 = declared
            # y grows upwards in display space, so the box's y0 is its bottom
            outside = (
                box[0] < x0 - TOLERANCE
                or box[2] > x1 + TOLERANCE
                or box[1] < y0 - TOLERANCE
                or box[3] > y1 + TOLERANCE
            )
            if outside:
                problems.append(
                    f"{kind}: '{label}' escapes its box "
                    f"(text {box[0]:.1f}..{box[2]:.1f} x {box[1]:.1f}..{box[3]:.1f}, "
                    f"box {x0:.1f}..{x1:.1f} x {y0:.1f}..{y1:.1f})"
                )

    # tick labels are not probes but still must not collide with anything
    for tick in list(fig.axes[0].get_yticklabels()) + list(fig.axes[0].get_xticklabels()):
        if not tick.get_text().strip():
            continue
        extent = tick.get_window_extent(renderer=renderer)
        texts.append((f"tick: {tick.get_text()}", [extent.x0, extent.y0, extent.x1, extent.y1]))

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            fraction = overlap(texts[i][1], texts[j][1])
            if fraction > OVERLAP_FRACTION:
                problems.append(
                    f"overlap {fraction * 100:.1f}%: {texts[i][0]} vs {texts[j][0]}"
                )

    for kind, size, label in sizes:
        if size < MIN_FONT:
            problems.append(f"{kind}: '{label}' shrank to {size:.2f} pt (unreadable)")

    # a label sitting on a drawn stroke (bracket drop, error whisker) is the
    # classic bar-chart defect: nothing here is text, so overlap rules miss it
    for label, box in texts:
        for kind, stroke in strokes:
            fraction = overlap(box, list(stroke))
            if fraction > OVERLAP_FRACTION:
                problems.append(f"{label} sits on the {kind} stroke ({fraction * 100:.1f}%)")

    print(
        f"canvas {width_px}x{height_px}px · measured {len(texts)} text artists · "
        f"probes {len(probes)} · strokes {len(strokes)}"
    )
    print("font sizes: " + ", ".join(f"{kind} {size:.1f}" for kind, size, _ in sizes))
    # the footnote is the one line allowed to be dropped rather than squeezed, so
    # report which variant survived and at what size
    for kind, size, label in sizes:
        if kind == "footnote":
            print(f'footnote drawn: "{label}" at {size:.1f} pt')
    if problems:
        print("problems:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("problems: 0")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
