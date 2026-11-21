from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Transform  # noqa: E402

HERE = Path(__file__).resolve().parent  # figures/slide 11
EVIDENCE = HERE / "pos_weights_evidence.json"
OUT = HERE / "pos_weights_calibration_minimal.png"

FIG_W, FIG_H, DPI = 6.0, 4.5, 300
TRANSPARENT = False  # pure white; flip to True for a dark slide background

# ---- geometry ------------------------------------------------------------
AX_RECT = (0.150, 0.300, 0.685, 0.545)  # 405 px below the axes for the notes
X_LIM = (-0.5, 1.5)  # the two categories sit at x = 0 and x = 1
Y_LIM = (50.0, 65.0)  # the spec's 50-65 % window
Y_TICKS = [50.0, 55.0, 60.0, 65.0]
BAR_W = 0.42  # data units; each category cell is 1.0 wide

# annotation geometry, in score units
BRACKET_DROP = 1.10  # bracket line sits this far above the taller error bar
BRACKET_TEXT_GAP = 0.80  # label sits this far above the bracket line
VALUE_INSET = 2.20  # value labels live inside the bars, this far below the top

# ---- palette -------------------------------------------------------------
INK = "#111827"
BODY = "#374151"
MUTED = "#6b7280"
GRID = "#e5e7eb"
SPINE = "#9ca3af"
# the two bars are the ablation grid's own pair, named as the grid names them:
#   `full` = the production config, which runs `--pos-weights doc` (that flag's
#            default is "doc", experiments/ablations/run_training_ablation.py:367)
#   `pos-weights-auto` = same graph, features, architecture and anchors, only the
#            BCE positive weights change (experiments/ablations/RESULTS.md:70-71)
DOC_BAR = "#94a3b8"  # muted gray, per the spec
DOC_BAR_EDGE = "#64748b"
AUTO_BAR = "#1e3a8a"  # vibrant deep blue, per the spec
AUTO_BAR_EDGE = "#1e3a8a"

X_LABEL = "Pos-weight source"
Y_LABEL = "Score @ 0.5 Threshold (%)"
CATEGORY_LABELS = ("Static Doc Weights", "Auto-Calibrated Weights")
DOC_SUB_NOTE = "252-anchor constants · 1.55–2.55× high"
AUTO_SUB_NOTE = "(N_obs - N_pos) / N_pos · per activity"
BRACKET_TEXT = "+4.26 pts Gain"

# font sizes; the notes are additionally shrunk until they measure inside their band
CAT_SIZE, CAT_MIN = 9.6, 7.0
SUB_SIZE, SUB_MIN = 7.4, 5.2
VALUE_SIZE, VALUE_MIN = 10.5, 8.0
BRACKET_SIZE, BRACKET_MIN = 9.4, 7.0
AXIS_LABEL_SIZE, AXIS_LABEL_MIN = 9.5, 7.0
TICK_SIZE = 8.5
FOOTNOTE_SIZE, FOOTNOTE_MIN = 7.4, 6.0
FOOTNOTE_Y = 0.050  # one line only: the rest of the provenance is in the code comments

# containment bands in axes fractions: the two category cells below the axis, the
# caption band under them, the columns above each bar, and the bracket band
CELL_BAND = (-0.225, -0.004)
XLABEL_BAND = (-0.315, -0.225)
COLUMN_BAND = (0.0, 1.0)
BRACKET_BAND = (0.45, 1.0)

# kind -> the label used in check_layout.py output
KINDS = ("category_label", "category_note", "value_label", "bracket_label")


@dataclass(frozen=True)
class Evidence:
    """The measured numbers this figure draws."""

    doc_mean: float
    doc_sd: float
    auto_mean: float
    auto_sd: float
    gain: float
    seeds: tuple[str, ...]
    cmaes_gain: float | None


class Probe(dict):
    """One text artist plus the pixel box it must stay inside (None = any)."""


def read_evidence(path: Path) -> Evidence:
    """Pull the measured numbers out of the artifact the compute step writes."""
    if not path.is_file():
        raise SystemExit(f"missing {path} - run compute_pos_weights_evidence.py first")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read the evidence artifact {path}: {exc}") from exc
    doc, auto = payload["doc_arm"], payload["auto_arm"]
    delta = payload["delta_at_0_5"]
    cmaes = payload.get("cmaes_reference")
    sd_doc, sd_auto = doc.get("sd_pct"), auto.get("sd_pct")
    if sd_doc is None or sd_auto is None:
        raise SystemExit("evidence file has no seed spread, but the figure draws error bars")
    gain = delta["mean_delta_pts"]
    if gain is None:
        raise SystemExit("evidence file has no mean gain")
    return Evidence(
        doc_mean=float(doc["mean_pct"]),
        doc_sd=float(sd_doc),
        auto_mean=float(auto["mean_pct"]),
        auto_sd=float(sd_auto),
        gain=float(gain),
        seeds=tuple(str(seed) for seed in doc["seeds"]),
        cmaes_gain=None if cmaes is None else float(cmaes["delta_at_cmaes_pts"]),
    )


def draw_width(fig: Figure, text: Text) -> float:
    """Width of a text artist in points, after a redraw."""
    fig.canvas.draw()
    return float(text.get_window_extent().width)


def fit_width(fig: Figure, text: Text, limit_points: float, min_size: float) -> bool:
    """Shrink a text artist until it fits a width budget, or report failure."""
    size = float(text.get_fontsize())
    while size >= min_size:
        text.set_fontsize(size)
        if draw_width(fig, text) <= limit_points:
            return True
        size -= 0.15
    return False


def ax_band(ax: Axes, x0: float, x1: float, y0: float, y1: float) -> tuple[float, ...]:
    """Convert an axes-fraction rectangle to canvas pixels."""
    low = ax.transAxes.transform((x0, y0))
    high = ax.transAxes.transform((x1, y1))
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def footnote(fig: Figure, evidence: Evidence) -> list[Probe]:
    """One provenance line: what the bars average, and what 'score' means."""
    count = len(evidence.seeds)
    seeds = "/".join(evidence.seeds)
    # longest first: the widest variant that stays readable wins, the rest are
    # fallbacks so the line is either legible or absent, never micro-text
    variants = [
        f"mean of {count} seeds ({seeds}) ± 1 sd · score = ½(macro-F1 + micro-F1)",
        f"mean of {count} seeds ± 1 sd · score = ½(macro-F1 + micro-F1)",
        f"mean of {count} seeds ± 1 sd",
    ]
    # widths are compared in canvas pixels, so the budget uses DPI, not 72 pt/in
    limit = AX_RECT[2] * FIG_W * DPI * 0.92
    for line in variants:
        text = fig.text(
            0.5,
            FOOTNOTE_Y,
            line,
            ha="center",
            va="center",
            fontsize=FOOTNOTE_SIZE,
            color=MUTED,
        )
        if fit_width(fig, text, limit, FOOTNOTE_MIN):
            return [Probe(kind="footnote", text=text, box=None, note=f"y={FOOTNOTE_Y:g}")]
        text.remove()
    raise SystemExit(f"no footnote variant fits at {FOOTNOTE_MIN} pt")


def build() -> tuple[Figure, list[Probe], list[tuple[str, tuple[float, ...]]]]:
    """Draw the chart, returning its text probes and the line obstacles to dodge."""
    evidence = read_evidence(EVIDENCE)
    means = [evidence.doc_mean, evidence.auto_mean]
    sds = [evidence.doc_sd, evidence.auto_sd]
    xs = [0.0, 1.0]

    fig = Figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#ffffff")
    fig.patch.set_alpha(0.0 if TRANSPARENT else 1.0)
    ax = fig.add_axes(AX_RECT)
    ax.set_facecolor("none")

    # limits first: every text artist below is placed in data or axes coordinates,
    # so the transforms must already be final before anything is measured
    ax.set_xticks(xs)
    ax.set_xticklabels([])  # drawn by hand so both notes are measurable artists
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels([f"{tick:g}" for tick in Y_TICKS])
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.7, color=GRID)
    ax.set_axisbelow(True)

    probes: list[Probe] = []
    # drawn strokes a label must never sit on top of: the bracket's drop lines and
    # the seed-spread whiskers, kept in data coords and converted once at the end
    strokes: list[tuple[str, float, float, float, float]] = []

    def place(
        kind: str,
        xy: tuple[float, float],
        label: str,
        size: float,
        box: tuple[float, ...] | None,
        note: str,
        *,
        min_size: float,
        color: str = INK,
        transform: Transform | None = None,
        weight: str = "normal",
    ) -> Text:
        """Add one text artist and register the pixel box it must stay inside."""
        # transform must be *omitted*, not passed as None: Text.__init__ takes the
        # missing kwarg to mean "use the axes data transform", while an explicit
        # None falls back to the identity transform and anchors the text at x = 0
        text = ax.text(
            xy[0],
            xy[1],
            label,
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            fontweight=weight,
            zorder=6,
            clip_on=False,
        )
        if transform is not None:
            text.set_transform(transform)
        if not fit_width(fig, text, (box[2] - box[0]) if box else 1e9, min_size):
            raise SystemExit(f"{kind} does not fit at {min_size} pt: {label}")
        probes.append(Probe(kind=kind, text=text, box=box, note=note))
        return text

    # --- bars + seed spread ----------------------------------------------
    bars = ax.bar(
        xs,
        means,
        width=BAR_W,
        color=[DOC_BAR, AUTO_BAR],
        edgecolor=[DOC_BAR_EDGE, AUTO_BAR_EDGE],
        linewidth=0.8,
        zorder=3,
    )
    for x, mean, sd, edge in zip(xs, means, sds, (DOC_BAR_EDGE, AUTO_BAR_EDGE), strict=True):
        ax.errorbar(
            [x],
            [mean],
            yerr=[[sd], [sd]],
            fmt="none",
            ecolor=edge,
            elinewidth=1.1,
            capsize=3.5,
            capthick=1.1,
            zorder=4,
        )
        strokes.append(("error_whisker", x, mean - sd, x, mean + sd))
        strokes.append(("error_cap", x - 0.06, mean + sd, x + 0.06, mean + sd))
        strokes.append(("error_cap", x - 0.06, mean - sd, x + 0.06, mean - sd))

    # --- significance bracket --------------------------------------------
    top = max(mean + sd for mean, sd in zip(means, sds, strict=True))
    bracket_y = top + BRACKET_DROP
    text_y = bracket_y + BRACKET_TEXT_GAP
    if text_y + 0.55 > Y_LIM[1]:
        raise SystemExit(f"the bracket needs headroom above {top:.2f} inside {Y_LIM}")
    for x, mean, sd in zip(xs, means, sds, strict=True):
        ax.plot(
            [x, x],
            [bracket_y, mean + sd + 0.45],
            color=BODY,
            lw=0.9,
            solid_capstyle="butt",
            zorder=5,
        )
        strokes.append(("bracket_drop", x, bracket_y, x, mean + sd + 0.45))
    ax.plot([xs[0], xs[1]], [bracket_y, bracket_y], color=BODY, lw=0.9, zorder=5)
    strokes.append(("bracket_beam", xs[0], bracket_y, xs[1], bracket_y))

    # --- value labels (inside the bars) + bracket label -------------------
    # above the bars the bracket's drop lines run, so the numbers live inside the
    # bars instead: nothing crosses them and the top stays clean for the bracket
    for index, (bar, mean) in enumerate(zip(bars, means, strict=True)):
        label = f"{mean:.2f}"
        place(
            "value_label",
            (float(bar.get_x()) + BAR_W / 2.0, mean - VALUE_INSET),
            label,
            VALUE_SIZE,
            ax_band(ax, 0.0 if index == 0 else 0.5, 0.5 if index == 0 else 1.0, *COLUMN_BAND),
            f"{label} inside the {'doc' if index == 0 else 'auto'} bar",
            min_size=VALUE_MIN,
            color=INK if index == 0 else "#ffffff",
            weight="bold",
        )
    place(
        "bracket_label",
        (0.5, text_y),
        BRACKET_TEXT,
        BRACKET_SIZE,
        ax_band(ax, 0.0, 1.0, *BRACKET_BAND),
        "the +4.26 gain, measured at the fixed 0.5 threshold",
        min_size=BRACKET_MIN,
        weight="bold",
    )

    # --- categories + the two notes under them ----------------------------
    # axes-fraction x: the centre of each category cell, not the bar's data x
    cell_centres = (0.25, 0.75)
    cells = [ax_band(ax, 0.0, 0.5, *CELL_BAND), ax_band(ax, 0.5, 1.0, *CELL_BAND)]
    for index, centre in enumerate(cell_centres):
        cell = cells[index]
        place(
            "category_label",
            (centre, -0.085),
            CATEGORY_LABELS[index],
            CAT_SIZE,
            cell,
            "category name",
            min_size=CAT_MIN,
            color=INK,
            transform=ax.transAxes,
        )
        place(
            "category_note",
            (centre, -0.175),
            (DOC_SUB_NOTE, AUTO_SUB_NOTE)[index],
            SUB_SIZE,
            cell,
            "how the two weightings differ",
            min_size=SUB_MIN,
            color=MUTED,
            transform=ax.transAxes,
        )

    # --- axes cosmetics ---------------------------------------------------
    # the x caption is drawn by hand too, so it sits below the two notes and is
    # itself measurable (set_xlabel would float it up against the category labels)
    place(
        "x_axis_label",
        (0.5, -0.270),
        X_LABEL,
        AXIS_LABEL_SIZE,
        ax_band(ax, -0.05, 1.05, XLABEL_BAND[0], XLABEL_BAND[1]),
        "caption for the two categories",
        min_size=AXIS_LABEL_MIN,
        color=BODY,
        transform=ax.transAxes,
    )
    ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_SIZE, color=BODY, labelpad=6)
    ax.tick_params(axis="both", length=3.2, color=SPINE, labelsize=TICK_SIZE, colors=BODY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.8)

    probes.append(Probe(kind="y_axis_label", text=ax.yaxis.label, box=None, note=Y_LABEL))
    probes.extend(footnote(fig, evidence))

    fig.canvas.draw()
    boxes: list[tuple[str, tuple[float, ...]]] = []
    for kind, x0, y0, x1, y1 in strokes:
        low = ax.transData.transform((x0, y0))
        high = ax.transData.transform((x1, y1))
        boxes.append(
            (
                kind,
                (
                    min(float(low[0]), float(high[0])) - 0.6,
                    min(float(low[1]), float(high[1])) - 0.6,
                    max(float(low[0]), float(high[0])) + 0.6,
                    max(float(low[1]), float(high[1])) + 0.6,
                ),
            )
        )
    return fig, probes, boxes


def main() -> None:
    fig, _, _ = build()
    fig.savefig(OUT, dpi=DPI, transparent=TRANSPARENT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
