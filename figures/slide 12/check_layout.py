"""Layout proof for the slide 12 terminal card.

The figure script hands back every text artist with the pixel box it must stay
inside (the card's inner width, at its own row band), plus the drawn marks that
are not text (the three window dots, the title rule).  This checker redraws the
card and fails on:

* anything drawn outside the canvas;
* text escaping its row band or the card's inner width - a terminal line that
  runs past the card border is the exact defect this card can have;
* two text artists overlapping by more than 2 % of the smaller one (row bands
  make that impossible unless the pitch was squeezed);
* text sitting on a drawn mark: a glyph crossing the title rule, or a title
  colliding with a traffic-light dot;
* a monospace row pair that no longer shares one size, which would break the
  terminal's column grid.

Exits non-zero when a problem is found, so it can gate a build.

    conda run -n bda_asg2 python "figures/slide 12/check_layout.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg

HERE = Path(__file__).resolve().parent  # figures/slide 12
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_evaluate_terminal_minimal import (  # noqa: E402  # pyright: ignore[reportMissingImports]  # sibling module in this folder; pyright's index predates it
    DPI,
    FIG_H,
    FIG_W,
    MONO_FAMILY,
    ROWS,
    build,
)

TOLERANCE = 0.6  # px of slack allowed against a box edge
OVERLAP_FRACTION = 0.02  # share of the smaller box before an overlap is a problem
MIN_FONT = 6.5  # below this a monospace line stops reading as terminal text


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
    fig, probes, shapes = build()
    # measure against a fresh renderer: an artist caches its last-draw extent and
    # a stale cache is how a line that clearly crosses the border reports 0 width
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    width_px, height_px = int(FIG_W * DPI), int(FIG_H * DPI)

    texts: list[tuple[str, list[float]]] = []
    problems: list[str] = []
    body_sizes: list[float] = []

    for probe in probes:
        text = probe["text"]
        label = text.get_text()
        if not label.strip():
            continue
        extent = text.get_window_extent(renderer=renderer)
        box = [extent.x0, extent.y0, extent.x1, extent.y1]
        kind = str(probe["kind"])
        texts.append((f"{kind}: {label}", box))
        if kind != "title":
            body_sizes.append(float(text.get_fontsize()))

        if (
            box[0] < -TOLERANCE
            or box[1] < -TOLERANCE
            or box[2] > width_px + TOLERANCE
            or box[3] > height_px + TOLERANCE
        ):
            problems.append(
                f"{kind}: '{label}' leaves the canvas "
                f"({box[0]:.1f}..{box[2]:.1f} x {box[1]:.1f}..{box[3]:.1f} "
                f"vs canvas {width_px}x{height_px})"
            )
        declared = probe["box"]
        if declared is not None:
            x0, y0, x1, y1 = declared
            if (
                box[0] < x0 - TOLERANCE
                or box[2] > x1 + TOLERANCE
                or box[1] < y0 - TOLERANCE
                or box[3] > y1 + TOLERANCE
            ):
                problems.append(
                    f"{kind}: '{label}' escapes its row band "
                    f"(text {box[0]:.1f}..{box[2]:.1f} x {box[1]:.1f}..{box[3]:.1f}, "
                    f"band {x0:.1f}..{x1:.1f} x {y0:.1f}..{y1:.1f})"
                )
        family = text.get_fontfamily()[0].strip("'\"").lower().replace("-", " ")
        if family != MONO_FAMILY.lower():
            problems.append(
                f"{kind}: '{label}' resolved to family '{family}', not {MONO_FAMILY}"
            )

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            fraction = overlap(texts[i][1], texts[j][1])
            if fraction > OVERLAP_FRACTION:
                problems.append(f"overlap {fraction * 100:.1f}%: {texts[i][0]} vs {texts[j][0]}")

    for label, box in texts:
        for shape in shapes:
            fraction = overlap(box, list(shape["box"]))
            if fraction > OVERLAP_FRACTION:
                problems.append(f"{label} sits on the {shape['kind']} mark ({fraction * 100:.1f}%)")

    if body_sizes and max(body_sizes) - min(body_sizes) > 0.05:
        problems.append(
            f"terminal rows are not one size: {sorted(set(body_sizes))} pt - the column grid breaks"
        )
    for size in body_sizes:
        if size < MIN_FONT:
            problems.append(f"a terminal row shrank to {size:.2f} pt (unreadable)")

    drawn = sum(1 for row in ROWS if row.text.strip())
    print(
        f"canvas {width_px}x{height_px}px · terminal rows {drawn} of {len(ROWS)} · "
        f"measured {len(texts)} text artists · marks {len(shapes)}"
    )
    print(f"row size: {body_sizes[0]:.1f} pt · title {texts[0][0]!r}" if texts else "no text")
    if problems:
        print("problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("problems: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
