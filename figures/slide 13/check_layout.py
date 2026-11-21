"""Layout and accuracy proof for the slide 13 ablation landscape.

Two jobs, because this chart is a claims chart:

Layout - redraw the figure, measure real pixel extents, and fail on text outside
the canvas, text escaping the box it was promised (a bar tag must stay inside its
bar, a value label inside its column, a header inside its band), any two text
artists overlapping, text sitting on a drawn mark (error whisker/cap, group
divider, the dashed reference line), or any label shrunk under MIN_FONT.

Accuracy - every value label must equal the number in
``ablation_landscape.json`` (which ``compute_ablation_landscape.py`` reads from
``experiments/ablations/ablation_summary.csv``), and every value label must start
clear of its own error whisker.  A chart that re-renders a stale JSON is exactly
how a slide ends up with a number the grid never produced.

    conda run -n bda_asg2 python "figures/slide 13/check_layout.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg

HERE = Path(__file__).resolve().parent  # figures/slide 13
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_ablation_landscape_minimal import (  # noqa: E402  # pyright: ignore[reportMissingImports]  # sibling module in this folder; pyright's index predates it
    DPI,
    EVIDENCE,
    FIG_H,
    FIG_W,
    VALUE_GAP,
    X_LIM,
    Evidence,
    build,
    read_evidence,
)

TOLERANCE = 0.6  # px of slack allowed against a box edge
OVERLAP_FRACTION = 0.02  # share of the smaller box before an overlap is a problem
MIN_FONT = 5.0  # below this the text is unreadable on a slide
WHISKER_CLEARANCE = 1.0  # px a value label must clear its own whisker


def box_of(artist: object, renderer: object) -> list[float]:
    extent = artist.get_window_extent(renderer=renderer)  # type: ignore[attr-defined]
    return [float(extent.x0), float(extent.y0), float(extent.x1), float(extent.y1)]


def overlap(a: list[float], b: list[float]) -> float:
    """Fraction of the smaller box covered by the intersection of two boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    area = width * height
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return area / smaller if smaller > 0 else 0.0


def arms_by_variant(evidence: Evidence) -> dict[str, float]:
    """Every score the figure is allowed to print, keyed by variant name."""
    scores: dict[str, float] = {}
    for _, arms in evidence.groups:
        for arm in arms:
            scores[arm.variant] = arm.score_cmaes
    reference = evidence.reference
    scores[reference.variant] = reference.score_cmaes
    return scores


def main() -> int:
    fig, probes, strokes = build()
    evidence = read_evidence(EVIDENCE)
    expected = arms_by_variant(evidence)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    width_px, height_px = int(FIG_W * DPI), int(FIG_H * DPI)
    problems: list[str] = []
    texts: list[tuple[str, list[float]]] = []

    for probe in probes:
        text = probe["text"]
        # fresh extents: an artist caches the box from its last draw
        extent = text.get_window_extent(renderer=renderer)
        box = [float(extent.x0), float(extent.y0), float(extent.x1), float(extent.y1)]
        label = text.get_text()
        kind = str(probe["kind"])
        texts.append((f"{kind}: {label[:36]}", box))

        if (
            box[0] < -TOLERANCE
            or box[1] < -TOLERANCE
            or box[2] > width_px + TOLERANCE
            or box[3] > height_px + TOLERANCE
        ):
            problems.append(f"{kind}: '{label[:36]}' runs off the canvas {box}")

        declared = probe["box"]
        if declared is not None:
            if box[0] < declared["x0"] - TOLERANCE:
                problems.append(
                    f"{kind}: '{label[:30]}' escapes its left edge "
                    f"({box[0]:.1f} vs {declared['x0']:.1f})"
                )
            if box[2] > declared["x1"] + TOLERANCE:
                problems.append(
                    f"{kind}: '{label[:30]}' escapes its right edge "
                    f"({box[2]:.1f} vs {declared['x1']:.1f})"
                )
            if box[1] < declared["y0"] - TOLERANCE or box[3] > declared["y1"] + TOLERANCE:
                problems.append(
                    f"{kind}: '{label[:30]}' escapes its band "
                    f"(y {box[1]:.1f}..{box[3]:.1f} vs {declared['y0']:.1f}..{declared['y1']:.1f})"
                )

        size = float(text.get_fontsize())
        if size < MIN_FONT:
            problems.append(f"{kind}: '{label[:36]}' at {size:.2f} pt is unreadable")

        note = str(probe["note"])
        if kind == "value_label":
            variant = note.split("arm=", 1)[1].split()[0] if "arm=" in note else ""
            want = expected.get(variant)
            if want is None:
                problems.append(f"value_label has no arm in its note: '{note}'")
            elif label != f"{want:.2f}":
                problems.append(
                    f"value_label for {variant} reads {label} but the grid says {want:.2f}"
                )

    for tick in list(fig.axes[0].get_yticklabels()) + list(fig.axes[0].get_xticklabels()):
        if not tick.get_text().strip():
            continue
        texts.append((f"tick: {tick.get_text()}", box_of(tick, renderer)))
        if float(tick.get_fontsize()) < MIN_FONT:
            problems.append(f"tick '{tick.get_text()}' at {tick.get_fontsize():.2f} pt")

    # axis labels are not probes either, and the x label is the neighbour that
    # most often slips down into the footnote band
    for name, artist in (
        ("x_label", fig.axes[0].xaxis.get_label()),
        ("y_label", fig.axes[0].yaxis.get_label()),
    ):
        if artist.get_text().strip():
            texts.append((f"{name}: {artist.get_text()}", box_of(artist, renderer)))
            if float(artist.get_fontsize()) < MIN_FONT:
                problems.append(f"{name} at {artist.get_fontsize():.2f} pt")

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            fraction = overlap(texts[i][1], texts[j][1])
            if fraction > OVERLAP_FRACTION:
                problems.append(
                    f"overlap {fraction * 100:.1f}%: {texts[i][0]} vs {texts[j][0]}"
                )

    stroke_boxes: list[tuple[str, list[float]]] = []
    for mark in strokes:
        mark_box = mark["box"]
        stroke_boxes.append(
            (str(mark["kind"]), [mark_box["x0"], mark_box["y0"], mark_box["x1"], mark_box["y1"]])
        )
    for label, box in texts:
        for kind, stroke_box in stroke_boxes:
            fraction = overlap(box, stroke_box)
            if fraction > OVERLAP_FRACTION:
                problems.append(f"{label} sits on the {kind} mark ({fraction * 100:.1f}%)")

    # a value label must start clear of its own whisker, not merely not overlap it
    whiskers = [b for kind, b in stroke_boxes if kind == "error_whisker"]
    for probe in probes:
        if probe["kind"] != "value_label" or probe["box"] is None:
            continue
        start = probe["box"]["x0"]
        extent = probe["text"].get_window_extent(renderer=renderer)
        for whisker in whiskers:
            if (
                whisker[1] - TOLERANCE <= extent.y0
                and extent.y1 <= whisker[3] + TOLERANCE
                and start < whisker[2] + WHISKER_CLEARANCE
            ):
                problems.append(
                    f"value label for {probe['note']} starts at {start:.1f}, "
                    f"inside the whisker ending {whisker[2]:.1f}"
                )

    print(
        f"canvas {width_px}x{height_px}px · probes {len(probes)} · measured "
        f"{len(texts)} text artists · marks {len(strokes)} · x window "
        f"{X_LIM[0]:g}..{X_LIM[1]:g} (truncated, stated in the footnote)"
    )
    modes = [str(p["note"]) for p in probes if p["kind"] == "bar_tag"]
    print("bar tags: " + ", ".join(m.split("=", 1)[1] for m in modes))
    print(
        "scores drawn: "
        + ", ".join(
            f"{p['text'].get_text()} ({str(p['note']).split('arm=', 1)[1]})"
            for p in probes
            if p["kind"] == "value_label"
        )
    )
    print(f"VALUE_GAP {VALUE_GAP} score-units between bar end and value label")
    if problems:
        print("problems:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("problems: 0")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
