"""Layout proof for the slide 10 pipeline figure.

    /train-data-5-hdd/charles/miniconda3/envs/bda_asg2/bin/python \
        "figures/slide 10/check_layout.py"

This figure is a layout problem, not an aesthetics problem, so the checker is
stricter than the slide 3/7/8 ones. `build()` hands back probes that pair every
text artist with the box it must live in, and this script measures the actual
rendered pixel extents against them. It fails on:

  * any text escaping its badge or its block-label band,
  * a badge rect leaving its card, or two badges overlapping inside a card,
  * two cards overlapping (the arrow gap must survive),
  * text-vs-text overlap anywhere,
  * anything drawn off the canvas.

Exit code is non-zero when a problem is found, so it can gate a commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # figures/slide 10
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_framework_pipeline_minimal import (  # noqa: E402  # pyright: ignore[reportMissingImports]  # sibling module in this folder; pyright's index predates it
    BADGE_INSET,
    BLOCK_W,
    BLOCKS,
    CARD_Y0,
    CARD_Y1,
    N_BLOCKS,
    Probe,
    badge_stack,
    block_x,
    build,
)

TOLERANCE = 0.6  # px of slack allowed at a box edge
MIN_GAP_PX = 1.5  # required clear space between stacked badges and between cards
OVERLAP_FRACTION = 0.02  # text-vs-text overlap allowed, as a share of the smaller box

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


def px(trans, x: float, y: float) -> tuple[float, float]:
    point = trans.transform((x, y))
    return float(point[0]), float(point[1])


def artist_box(artist) -> Box:
    box = artist.get_window_extent()
    return box.x0, box.y0, box.x1, box.y1


def data_rect(trans, x0: float, y0: float, x1: float, y1: float) -> Box:
    lo = px(trans, x0, y0)
    hi = px(trans, x1, y1)
    return min(lo[0], hi[0]), min(lo[1], hi[1]), max(lo[0], hi[0]), max(lo[1], hi[1])


def overlap(a: Box, b: Box) -> float:
    """Intersection area of two boxes (0 when they only touch)."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0.0 and height > 0.0 else 0.0


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def inside(inner: Box, outer: Box) -> bool:
    return (inner[0] >= outer[0] - TOLERANCE and inner[1] >= outer[1] - TOLERANCE
            and inner[2] <= outer[2] + TOLERANCE and inner[3] <= outer[3] + TOLERANCE)


def describe(probe: Probe) -> str:
    return (f"[{probe['kind']} block={probe['block']} badge={probe['badge']}] "
            f"{probe['text'].get_text()!r}")


def main() -> int:
    fig, probes = build()
    fig.canvas.draw()
    width_px = int(fig.get_figwidth() * fig.dpi)
    height_px = int(fig.get_figheight() * fig.dpi)

    # Every block text lives on the full-figure axes, so one transform covers all
    # of them; the footnote is figure-level and measured in pixels directly.
    first_axes = probes[0]["text"].axes
    if first_axes is None:
        print("first probe is figure-level, so data units cannot be converted")
        return 1
    trans = first_axes.transData

    problems: list[str] = []
    card_rects: dict[int, Box] = {}

    for probe in probes:
        box = artist_box(probe["text"])
        if probe["box"] is not None:
            target = data_rect(trans, *probe["box"])
            if not inside(box, target):
                problems.append(
                    f"{describe(probe)} escapes its box: text "
                    f"({box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}) vs box "
                    f"({target[0]:.1f},{target[1]:.1f},{target[2]:.1f},{target[3]:.1f})"
                )
            if probe["kind"] == "header":
                i = probe["block"]
                card_rects[i] = data_rect(trans, block_x(i), CARD_Y0,
                                          block_x(i) + BLOCK_W, CARD_Y1)
        else:
            # The footnote must stay under the cards.
            card_bottom_px = px(trans, 0.0, CARD_Y0)[1]
            if box[3] >= card_bottom_px:
                problems.append(f"{describe(probe)} intrudes on the cards above it")

    # Card/badge geometry recomputed from the same constants the drawing uses.
    for i in range(N_BLOCKS):
        card = card_rects.get(i)
        if card is None:
            problems.append(f"block {i}: no header probe, cannot check its card")
            continue
        x0 = block_x(i)
        x1 = x0 + BLOCK_W
        stack = badge_stack(x0, [len(b) for b in BLOCKS[i][1]])
        rects = [data_rect(trans, x0 + BADGE_INSET, y0, x1 - BADGE_INSET, y1)
                 for y0, y1 in stack]
        for j, rect in enumerate(rects):
            if not inside(rect, card):
                problems.append(f"block {i} badge {j}: rect leaves the card")
        for j in range(len(rects) - 1):
            gap = rects[j][1] - rects[j + 1][3]
            if gap < MIN_GAP_PX:
                problems.append(
                    f"block {i}: badges {j}/{j + 1} gap {gap:.1f}px < {MIN_GAP_PX}px")
        if i + 1 < N_BLOCKS:
            clearance = px(trans, block_x(i + 1), CARD_Y0)[0] - px(trans, x1, CARD_Y0)[0]
            if clearance < MIN_GAP_PX:
                problems.append(
                    f"blocks {i}/{i + 1}: only {clearance:.1f}px of arrow gap")

    # No text-on-text collisions anywhere on the canvas.
    boxes = [artist_box(probe["text"]) for probe in probes]
    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            hit = overlap(boxes[i], boxes[j])
            if hit > OVERLAP_FRACTION * min(area(boxes[i]), area(boxes[j])):
                problems.append(
                    f"text overlap {hit:.1f}px^2: "
                    f"{probes[i]['text'].get_text()!r} vs {probes[j]['text'].get_text()!r}")

    # Everything must be inside the canvas.
    for probe in probes:
        box = artist_box(probe["text"])
        if (box[0] < -TOLERANCE or box[1] < -TOLERANCE
                or box[2] > width_px + TOLERANCE or box[3] > height_px + TOLERANCE):
            problems.append(
                f"off canvas: {probe['text'].get_text()!r} -> "
                f"({box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f})")

    print(f"canvas {width_px}x{height_px}px · measured {len(probes)} text artists · "
          f"cards {len(card_rects)}/{N_BLOCKS}")
    for problem in problems:
        print(f"  PROBLEM {problem}")
    print(f"problems: {len(problems)}")

    if card_rects:
        left = min(r[0] for r in card_rects.values())
        right = max(r[2] for r in card_rects.values())
        top = max(r[3] for r in card_rects.values())
        bottom = min(r[1] for r in card_rects.values())
        share = 100.0 * (right - left) * (top - bottom) / (width_px * height_px)
        print(f"margins: left {left:.0f}px right {width_px - right:.0f}px "
              f"top {height_px - top:.0f}px bottom {bottom:.0f}px · "
              f"the 5 blocks cover {share:.0f}% of the canvas area")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
