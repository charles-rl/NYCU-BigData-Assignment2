#!/usr/bin/env python3
"""Layout audit for corpus_data_quality_audit.png (the slide 3 figure).

This session cannot view images, so the figure is verified geometrically instead:
every Text artist is measured through the Agg renderer and checked for

  a. leaving the canvas,
  b. overlapping another text box,
  c. spilling outside the bar/band it is supposed to sit in,
  d. colliding with the density curves it annotates.

Run:  python "figures/slide 3/check_layout.py"
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]  # pyrightconfig.json venvPath is stale on this box; pandas resolves and runs in env bda_asg2
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Annotation, Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_corpus_audit_figure import (  # noqa: E402
    AX_D,
    AX_T,
    BAR_H,
    BAR_Y,
    DENS_MAX,
    PHASES,
    PUSH_MAX,
    L,
    R,
    build,
    load_daily,
    ts,
    xfrac,
)


def walk_texts(fig):
    yield from ((None, t) for t in fig.texts)
    for ax in fig.axes:
        yield from ((ax, t) for t in ax.texts)
        yield from ((ax, t) for t in ax.get_xticklabels() + ax.get_yticklabels())
        legend = ax.get_legend()
        if legend is not None:
            yield from ((ax, t) for t in legend.get_texts())
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


def bar_box(start: str, end: str) -> tuple[float, float, float, float]:
    """Track-1 status bar segment, in figure coords."""
    y0 = AX_T[1] + (BAR_Y - BAR_H / 2) * AX_T[3]
    y1 = AX_T[1] + (BAR_Y + BAR_H / 2) * AX_T[3]
    return xfrac(start), xfrac(end) + (xfrac("2025-12-02") - xfrac("2025-12-01")), y0, y1


def main() -> int:
    fig = build()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    items = [
        (t.get_text().replace("\n", " / ")[:56], text_bbox(fig, t, renderer), t)
        for ax, t in walk_texts(fig)
        if isinstance(t, (Text, Annotation)) and t.get_text().strip()
    ]
    print(f"measured {len(items)} text artists")

    problems = 0

    print("\n— text outside the canvas —")
    out = 0
    for label, bb, _ in items:
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

    print("\n— in-bar captions must stay inside their bar segment —")
    expectations = [
        ("CONTIGUOUS", PHASES[0]),
        ("INTERMITTENT", PHASES[1]),
    ]
    bad = 0
    for needle, phase in expectations:
        x0, x1, y0, y1 = bar_box(phase[0], phase[1])
        for label, bb, _ in items:
            if not label.startswith(needle):
                continue
            ok = bb.x0 >= x0 and bb.x1 <= x1 and bb.y0 >= y0 and bb.y1 <= y1
            print(
                f"  {'OK ' if ok else 'BAD'} {needle!r}: text x[{bb.x0:.3f},{bb.x1:.3f}] "
                f"y[{bb.y0:.3f},{bb.y1:.3f}]  bar x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}]"
            )
            bad += 0 if ok else 1
    problems += bad

    print("\n— any caption drawn inside the density panel must clear the curves —")
    daily = load_daily()
    axD_y0, axD_h = AX_D[1], AX_D[3]
    span_days = ts("2026-12-06") - ts("2025-11-27")
    origin = datetime(2025, 11, 27)
    lo_d, lo_p = -0.06 * DENS_MAX, -0.06 * PUSH_MAX
    series: dict[str, pd.Series] = {
        "PR+Issue": pd.Series((daily["pr_issue"] - lo_d) / (DENS_MAX - lo_d), index=daily.index),
        "Push": pd.Series((daily["push"] - lo_p) / (PUSH_MAX - lo_p), index=daily.index),
    }
    bad = 0
    checked = 0
    for label, bb, _ in items:
        # only free-floating captions inside the density panel (ticks/axis labels excluded)
        if not (AX_D[0] < bb.x0 and bb.x1 < AX_D[2] and AX_D[1] < bb.y0 and bb.y1 < AX_D[3]):
            continue
        checked += 1
        t0 = origin + timedelta(days=(bb.x0 - L) / (R - L) * span_days)
        t1 = origin + timedelta(days=(bb.x1 - L) / (R - L) * span_days)
        worst = []
        for name, frac in series.items():
            window = frac.loc[t0:t1]
            if len(window) == 0:
                continue
            lo = axD_y0 + float(window.min()) * axD_h
            hi = axD_y0 + float(window.max()) * axD_h
            # the caption may not intersect the vertical range the curve sweeps in its span
            if bb.y0 < hi and bb.y1 > lo:
                worst.append(
                    f"{name} sweeps y[{lo:.3f},{hi:.3f}] over {t0:%Y-%m-%d}..{t1:%Y-%m-%d}"
                )
        print(f"  {'CLEAN' if not worst else 'HIT '} {label!r}: {'; '.join(worst) or 'clear'}")
        bad += len(worst)
    print(f"  ({checked} caption(s) inside the density panel)")
    problems += bad

    print("\n— extents —")
    for label, bb, _ in items:
        print(f"  x[{bb.x0:.3f},{bb.x1:.3f}] y[{bb.y0:.3f},{bb.y1:.3f}]  {label}")

    print(f"\nproblems: {problems}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
