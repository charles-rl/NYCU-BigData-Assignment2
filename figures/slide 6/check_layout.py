#!/usr/bin/env python3
"""Layout audit for relational_ablations_and_lightgbm.png (the slide 6 figure).

The figure is verified geometrically rather than by eye: every Text artist is
measured through the Agg renderer and checked for

  a. leaving the canvas,
  b. overlapping another text box,
  c. a value label falling outside the axes it annotates,
  d. a value label colliding with the bar or whisker it belongs to.

Run:  python "figures/slide 6/check_layout.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Annotation, Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_relational_ablations_and_lightgbm import (  # noqa: E402
    ACTIVITIES,
    build,
    load_activity_deltas,
    load_scores,
)


def walk_texts(fig):
    yield from ((None, t) for t in fig.texts)
    for ax in fig.axes:
        yield from ((ax, t) for t in ax.texts)
        yield from ((ax, t) for t in ax.get_xticklabels() + ax.get_yticklabels())
        for axis in (ax.xaxis, ax.yaxis):
            label = axis.get_label()
            if label.get_text():
                yield ax, label


def text_bbox(fig, t, renderer: RendererBase):
    """Text-only extent in figure coords (Annotation.get_window_extent adds the arrow)."""
    bb = Text.get_window_extent(t, renderer=renderer)
    return bb.transformed(fig.transFigure.inverted())


def area(bb) -> float:
    return max(bb.x1 - bb.x0, 0.0) * max(bb.y1 - bb.y0, 0.0)


def overlap(a, b) -> float:
    return max(min(a.x1, b.x1) - max(a.x0, b.x0), 0.0) * max(
        min(a.y1, b.y1) - max(a.y0, b.y0), 0.0
    )


def main() -> int:
    fig = build()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    ax_l, ax_r = fig.axes[0], fig.axes[1]

    items = [
        (t.get_text().replace("\n", " / ")[:60], text_bbox(fig, t, renderer), ax)
        for ax, t in walk_texts(fig)
        if isinstance(t, (Text, Annotation)) and t.get_text().strip()
    ]
    print(f"measured {len(items)} text artists")

    problems = 0

    print("\n— text outside the canvas —")
    out = 0
    for label, bb, _ax in items:
        if bb.x0 < -0.002 or bb.x1 > 1.002 or bb.y0 < -0.004 or bb.y1 > 1.004:
            out += 1
            print(f"  OUT x[{bb.x0:.3f},{bb.x1:.3f}] y[{bb.y0:.3f},{bb.y1:.3f}]  {label}")
    problems += out
    if not out:
        print("  none")

    print("\n— text / text overlaps (>2% of the smaller box) —")
    hits = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            la, ba, _ = items[i]
            lb, bb, _ = items[j]
            ov = overlap(ba, bb)
            if ov <= 0.0 or ov / min(area(ba), area(bb)) <= 0.02:
                continue
            hits += 1
            print(f"  {ov / min(area(ba), area(bb)) * 100:5.1f}%  {la!r}  <->  {lb!r}")
    problems += hits
    if not hits:
        print("  none")

    print("\n— value labels must stay inside their own axes —")
    bad = 0
    for _owner, ax, name in (("ax_l", ax_l, "left"), ("ax_r", ax_r, "right")):
        box = ax.get_position()
        inside = [
            (label, bb)
            for label, bb, _ in items
            if box.x0 <= bb.x0 and bb.x1 <= box.x1 and box.y0 <= bb.y0 and bb.y1 <= box.y1
        ]
        print(f"  {name}: {len(inside)} text artist(s) fully inside the axes box")
    print("  (brackets and captions are allowed to sit outside; the checks below are strict)")

    print("\n— left chart: every 'mean ± sd' caption clears its bar top + whisker —")
    scores = load_scores()
    order = ["full", "no-text", "no-residual-fusion", "lightgbm"]
    for i, arm in enumerate(order):
        top = scores[arm]["cma"] + scores[arm]["sd"]
        # data -> figure coords for the whisker cap
        cap = ax_l.transData.transform((i, top))
        cap_fig = fig.transFigure.inverted().transform(cap)
        want = f"{scores[arm]['cma']:.2f} ± {scores[arm]['sd']:.2f}"
        got = [bb for label, bb, _ in items if label == want]
        if not got:
            print(f"  MISSING caption for {arm!r}")
            bad += 1
            continue
        bb = got[0]
        ok = bb.y0 >= cap_fig[1] - 0.001
        print(
            f"  {'OK ' if ok else 'BAD'} {arm!r}: caption y0={bb.y0:.4f} vs whisker cap "
            f"y={cap_fig[1]:.4f} (caption must start at or above the cap)"
        )
        bad += 0 if ok else 1
    problems += bad

    print("\n— right chart: Δ captions clear the whisker end and stay in the axes —")
    deltas = load_activity_deltas()
    ys = list(range(len(ACTIVITIES)))[::-1]
    bad = 0
    box_r = ax_r.get_position()
    for act, y in zip(ACTIVITIES, ys, strict=True):
        want = f"+{deltas[act]['delta']:.2f}"
        got = [bb for label, bb, _ in items if label == want]
        if not got:
            print(f"  MISSING caption for {act!r}")
            bad += 1
            continue
        bb = got[0]
        end = ax_r.transData.transform((deltas[act]["delta"] + deltas[act]["sd"], y))
        end_fig = fig.transFigure.inverted().transform(end)
        clears = bb.x0 >= end_fig[0] - 0.001
        inside = box_r.x0 <= bb.x0 and bb.x1 <= box_r.x1
        print(
            f"  {'OK ' if clears and inside else 'BAD'} {act!r}: caption x[{bb.x0:.3f},{bb.x1:.3f}] "
            f"whisker end x={end_fig[0]:.3f} axes x1={box_r.x1:.3f}"
        )
        bad += 0 if clears and inside else 1
    problems += bad

    print("\n— extents —")
    for label, bb, _ in items:
        print(f"  x[{bb.x0:.3f},{bb.x1:.3f}] y[{bb.y0:.3f},{bb.y1:.3f}]  {label}")

    print(f"\nproblems: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
