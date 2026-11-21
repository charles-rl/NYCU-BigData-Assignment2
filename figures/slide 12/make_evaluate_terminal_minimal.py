"""Slide 12 figure: a dark-mode terminal card for ``scripts/evaluate.py``.

Every string on this card is a real string from this repo, not a mock-up:

* the two commands are the two documented invocations of ``scripts/evaluate.py``
  (``scripts/evaluate.py:3-5``, ``README.md:105-108``): the format check against the
  official test list, then scoring a held-out validation week;
* the hand-in name carries **your** student id, ``output_314540049.csv``.  The id in
  the repo's own example is 314551051 (``scripts/evaluate.py:7``) and appears nowhere
  else in the repo, so it is not a claim about you;
* ``submission/test_input.csv`` is the official test list: 4,465 unique UserIDs, the
  same set as ``submission/sample_submission.csv``, and every one of them is inside the
  9,000 ids in ``scripts/cohort.txt``.  So ``Format check: no errors (4465 users)`` is
  the verbatim print at ``scripts/evaluate.py:100`` and the output block documented in
  ``docs/FINAL_RUN.md:138-143``;
* the validation paths are what the driver actually writes: predictions to
  ``splits/val_best_pred_<val-cutoff>.csv`` (``scripts/run_training.py:1455``)
  and truth to ``splits/val_truth_<val-cutoff>.csv`` (``scripts/run_training.py:1421``)
  at the protocol's cutoff 2026-02-15 (``experiments/ablations/RESULTS.md:6``,
  ``docs/HPO.md:99``).  The stale ``splits/answer.csv`` is deliberately not shown:
  ``docs/VALIDATION_DATA.md:23`` calls it stale and invalid, and the driver writes a
  fresh truth file "so the check never silently reads a stale splits/answer.csv"
  (``scripts/run_training.py:1420``);
* the three metric lines are the exact strings ``scripts/evaluate.py:109-111``
  prints, ``Score:`` with its four spaces included, using the numbers of the
  ``pos-weights-auto`` arm's 3-seed mean at CMA-ES thresholds
  (``experiments/ablations/ablation_summary.csv``, seeds 42/123/999): macro-F1 63.16,
  micro-F1 68.51, and Score 65.84, which is (macro + micro) / 2 exactly as
  ``evaluate.py`` defines it.  **These are not the ``full`` (doc-weights) arm**, whose
  line is 63.06 / 68.36 / 65.71; ``auto`` is the arm the submission run uses because
  ``--pos-weights`` defaults to ``auto`` (``scripts/run_training.py:319-320``).  The
  ablation protocol is 8 training anchors and the 2026-02-15 validation week
  (``experiments/ablations/RESULTS.md:6``), so no number on this card is from the
  graded test week - nothing can score that week, there is no truth file for it.

Three things the card leaves out, stated here rather than crowding the card:

* ``--truth`` prints six per-activity ``F1`` lines before the three summary lines
  (``scripts/evaluate.py:107-108``); the card notes them in one muted shell comment
  instead of turning the badge into a log dump.
* a ``--truth`` run also prints its **own** ``Format check`` line, and at this cutoff
  that count would be 5,873 users, not 4,465: ``val_user_ids`` is the cohort subset of
  the feature view (``scripts/run_training.py:433``, ``mask = is_cohort & present``)
  and at most 5,873 of the 9,000 cohort users have ``first_seen_at`` before
  2026-02-15.  The card deliberately shows exactly one Format-check line, under the
  test-list command it belongs to.
* Long commands wrap with ``\\`` continuations, the way you would type them, so a
  single monospace size fits the card width.  The size is chosen by measuring the
  widest row against the card's inner width and stepping down until it fits, which
  keeps the monospace grid intact instead of shrinking one line only.

    conda run -n bda_asg2 python "figures/slide 12/make_evaluate_terminal_minimal.py"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent  # figures/slide 12
OUT = HERE / "evaluate_terminal_minimal.png"

# 8 x 4 in at 300 DPI = 2400 x 1200 px; 10 units per inch on both axes so the
# window corners and the traffic-light dots stay circular
FIG_W, FIG_H, DPI = 8.0, 4.0, 300
XMAX, YMAX = 80.0, 40.0
TRANSPARENT = False  # the spec's two options were transparent or pure white

# the card itself, with the wide margin that makes it read as a badge
CARD_X0, CARD_Y0, CARD_X1, CARD_Y1 = 4.4, 2.6, 75.6, 37.4
ROUNDING = 1.5
TITLE_H = 4.2
DOT_R = 0.78
DOT_XS = (7.9, 10.5, 13.1)
TITLE_TEXT = "bash - scripts/evaluate.py"

BODY_X = 7.9  # left edge of every terminal row, aligned with the first dot
PAD_X = 2.3  # right safety margin inside the card
PAD_TOP = 1.7  # gap under the title rule
PAD_BOTTOM = 1.9  # gap above the card's bottom edge
ROW_PITCH_MAX = 3.10  # units; the stack is centred, so this is a cap not a target
LINE_FILL = 0.62  # glyph height may not exceed this share of one row pitch
MONO_SIZE_MAX, MONO_SIZE_MIN = 12.0, 7.5
MONO_FAMILY = "DejaVu Sans Mono"

CANVAS = "#ffffff"
CARD_BG = "#0f172a"  # spec background
CARD_EDGE = "#334155"  # spec border
TITLE_RULE = "#1e293b"
TITLE_INK = "#94a3b8"
DOT_RED, DOT_YELLOW, DOT_GREEN = "#f87171", "#fbbf24", "#34d399"
PROMPT = "#67e8f9"  # spec: prompt in cyan
OUTPUT = "#4ade80"  # spec: format-check line in bright green
METRIC = "#f8fafc"  # spec: metric output in white
SCORE = "#fde68a"  # spec: ... with the score picked out in yellow
COMMENT = "#64748b"


@dataclass(frozen=True)
class Row:
    """One terminal line.  An empty text is a deliberate blank between runs."""

    text: str
    color: str
    kind: str


ROWS: list[Row] = [
    Row("$ python scripts/evaluate.py --pred output_314540049.csv \\", PROMPT, "prompt"),
    Row("      --input submission/test_input.csv", PROMPT, "prompt"),
    Row("Format check: no errors (4465 users)", OUTPUT, "output"),
    Row("", COMMENT, "blank"),
    Row("$ python scripts/evaluate.py --pred splits/val_best_pred_2026-02-15.csv \\", PROMPT, "prompt"),
    Row("      --truth splits/val_truth_2026-02-15.csv", PROMPT, "prompt"),
    Row("macro-F1: 63.16%", METRIC, "metric"),
    Row("micro-F1: 68.51%", METRIC, "metric"),
    Row("Score:    65.84%", SCORE, "score"),
    Row("# + 6 per-activity F1 lines and a Format check line of its own", COMMENT, "comment"),
    Row("# metrics: 3-seed mean at CMA-ES thresholds · val cutoff 2026-02-15", COMMENT, "comment"),
]


class Probe(TypedDict):
    """A text artist plus the pixel box it must stay inside."""

    kind: str
    text: Text
    box: tuple[float, float, float, float] | None
    note: str


class Shape(TypedDict):
    """A drawn non-text mark, so the checker can catch text sitting on it."""

    kind: str
    box: tuple[float, float, float, float]


def unit_px() -> float:
    """Canvas pixels per data unit (the axes spans the whole figure)."""
    return (FIG_W * DPI) / XMAX


def data_box(ax, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    """Convert a data-unit rectangle to canvas pixels."""
    low = ax.transData.transform((x0, y0))
    high = ax.transData.transform((x1, y1))
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def choose_size(
    fig: Figure,
    ax,
    canvas: FigureCanvasAgg,
    limit_px: float,
    pitch: float,
    top: float,
) -> tuple[list[tuple[Row, float, Text]], float]:
    """Largest single monospace size at which every row fits the card.

    One size for all rows on purpose: a terminal is a monospaced grid, and
    shrinking only the long lines would break the column alignment the card is
    selling.
    """
    size = MONO_SIZE_MAX
    while True:
        placed: list[tuple[Row, float, Text]] = []
        for i, row in enumerate(ROWS):
            y = top - i * pitch
            if row.text:
                placed.append(
                    (
                        row,
                        y,
                        ax.text(
                            BODY_X,
                            y,
                            row.text,
                            ha="left",
                            va="center",
                            fontsize=size,
                            family=MONO_FAMILY,
                            color=row.color,
                        ),
                    )
                )
        canvas.draw()
        renderer = canvas.get_renderer()
        extents = [text.get_window_extent(renderer=renderer) for _, _, text in placed]
        widest = max(float(e.width) for e in extents)
        tallest = max(float(e.height) for e in extents)
        if widest <= limit_px and tallest <= LINE_FILL * pitch * unit_px():
            return placed, size
        for _, _, text in placed:
            text.remove()
        size -= 0.1
        if size < MONO_SIZE_MIN:
            raise SystemExit(
                f"no monospace size fits: widest {widest:.0f}px vs limit {limit_px:.0f}px "
                f"at {MONO_SIZE_MIN} pt - widen the card or shorten a row"
            )


def build() -> tuple[Figure, list[Probe], list[Shape]]:
    """Draw the card and return it with the artists and marks to be measured."""
    fig = Figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=CANVAS)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.0 if TRANSPARENT else 1.0)

    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, YMAX)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (CARD_X0, CARD_Y0),
            CARD_X1 - CARD_X0,
            CARD_Y1 - CARD_Y0,
            boxstyle=f"round,pad=0,rounding_size={ROUNDING}",
            facecolor=CARD_BG,
            edgecolor=CARD_EDGE,
            linewidth=1.6,
        )
    )

    rule_y = CARD_Y1 - TITLE_H
    dot_y = CARD_Y1 - TITLE_H / 2
    ax.plot(
        [CARD_X0 + 0.9, CARD_X1 - 0.9],
        [rule_y, rule_y],
        color=TITLE_RULE,
        linewidth=1.1,
        solid_capstyle="butt",
    )
    for dot_x, color in zip(DOT_XS, (DOT_RED, DOT_YELLOW, DOT_GREEN), strict=True):
        ax.add_patch(
            Circle((dot_x, dot_y), DOT_R, facecolor=color, edgecolor="none", alpha=0.92)
        )
    title = ax.text(
        (CARD_X0 + CARD_X1) / 2,
        dot_y,
        TITLE_TEXT,
        ha="center",
        va="center",
        fontsize=9.6,
        family=MONO_FAMILY,
        color=TITLE_INK,
    )

    body_top = rule_y - PAD_TOP
    body_bottom = CARD_Y0 + PAD_BOTTOM
    # fit the whole stack inside the card, then centre it in the space left over
    pitch = min(ROW_PITCH_MAX, (body_top - body_bottom) / max(len(ROWS) - 1, 1))
    stack = (len(ROWS) - 1) * pitch
    top = body_bottom + (body_top - body_bottom + stack) / 2
    # the width budget is measured from where the rows actually start (BODY_X), not
    # from the card edge plus PAD_X: the left margin is wider than the right one, so
    # budgeting from the card edge lets a long line cross the inner right padding
    limit_px = (CARD_X1 - PAD_X - BODY_X) * unit_px()
    placed, _size = choose_size(fig, ax, canvas, limit_px, pitch, top)

    band = 0.5 * pitch * unit_px()
    box_x0, box_x1 = (CARD_X0 + 1.5) * unit_px(), (CARD_X1 - PAD_X) * unit_px()
    probes: list[Probe] = [
        Probe(
            kind="title",
            text=title,
            box=(box_x0, (dot_y * unit_px()) - band, box_x1, (dot_y * unit_px()) + band),
            note="window title, between the dots and the card edge",
        )
    ]
    for row, y, text in placed:
        probes.append(
            Probe(
                kind=row.kind,
                text=text,
                box=(box_x0, y * unit_px() - band, box_x1, y * unit_px() + band),
                note=f"row pitch {pitch:.2f}u",
            )
        )

    shapes: list[Shape] = [
        Shape(
            kind=f"dot_{name}",
            box=(x - DOT_R, dot_y - DOT_R, x + DOT_R, dot_y + DOT_R),
        )
        for name, x in zip(("red", "yellow", "green"), DOT_XS, strict=True)
    ]
    shapes.append(
        Shape(kind="title_rule", box=data_box(ax, CARD_X0 + 0.9, rule_y - 0.08, CARD_X1 - 0.9, rule_y + 0.08))
    )

    canvas.draw()
    return fig, probes, shapes


def main() -> None:
    fig, probes, _shapes = build()
    fig.savefig(OUT, dpi=DPI, facecolor=fig.get_facecolor(), transparent=TRANSPARENT)
    sizes = sorted({float(probe["text"].get_fontsize()) for probe in probes})
    print(
        f"wrote {OUT} · {len(ROWS)} terminal rows · monospace sizes "
        + ", ".join(f"{size:.1f}" for size in sizes)
    )


if __name__ == "__main__":
    main()
