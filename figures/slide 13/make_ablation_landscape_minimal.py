"""Slide 13 figure: ablation landscape on the official Score @ CMA-ES axis.

Six bars in three matched pairs, every number read from
``figures/slide 13/ablation_landscape.json`` (which
``compute_ablation_landscape.py`` builds from
``experiments/ablations/ablation_summary.csv``).  Three deliberate departures from
the slide spec, all of them accuracy decisions:

* **Pairs are shown at matched settings.**  The spec put 65.84 on both the
  "1-Layer (Auto-Weights)" and "Full (with Edge Decay)" bars.  65.84 is
  ``pos-weights-auto``, the submission recipe (1 layer, auto weights); the depth
  study's own 1-layer arm is ``depth-1layer`` at 65.68 and the edge-decay pair is
  ``full`` 65.71 vs ``no-temporal-decay`` 65.77, all three at the driver's default
  ``--pos-weights doc`` (``experiments/ablations/run_all_ablations.py:116-129``).
  Putting 65.84 next to 65.77 would read as "edge decay is worth +0.07" when the
  two arms differ in loss weights as well.  So ``pos-weights-auto`` is drawn as a
  dashed reference line at 65.84 instead of contaminating a matched pair.
* **LightGBM auto is 63.22, not the ~63.70 the spec guessed**, and it is *below*
  the doc arm at tuned thresholds (-0.12) while 1.92 above it at the default 0.5
  threshold.  The parity claim is therefore "+2.62 @CMA for the graph", measured as
  65.84 - 63.22, and the group header states the flip the grid records: at 0.5 the
  tabular baseline leads 61.78 vs 61.54 (``experiments/ablations/RESULTS.md:114-115``).
* **"67m" and "112m" are minutes per cell, not parameter counts.**  Measured:
  67.6 vs 112.4 min/cell (the grid's own "+65% wall time" is against ``full`` at
  68.3), while parameters go 2,304,630 -> 3,048,166 = +32.3%.

Whiskers are +-1 sample sd over the three seeds.  Note the grid quotes population
sd (``full`` sd 0.07) where this chart uses sample sd (0.09); the per-seed range
65.63-65.80 is identical either way, and the chart's rule of thumb - any delta
under ~0.2 is seed noise - is the grid's own (``RESULTS.md:81-82``).

    conda run -n bda_asg2 python "figures/slide 13/make_ablation_landscape_minimal.py"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent  # figures/slide 13
EVIDENCE = HERE / "ablation_landscape.json"
OUT = HERE / "unfeatured_ablations_landscape.png"

FIG_W, FIG_H, DPI = 7.5, 4.5, 300
TRANSPARENT = False  # spec allows transparent or pure white; white for slides

# one canvas unit = 0.1 inch on both axes, so strokes and gaps stay isotropic
XMAX, YMAX = 75.0, 45.0

# the official score axis; truncated at 62 by the spec, which the footnote says
X_LIM = (62.0, 66.6)
X_TICKS = [62.0, 63.0, 64.0, 65.0, 66.0]
X_LABEL = "Official Score @ CMA-ES (%)"

# left margin carries the arm names, right margin is where the value labels live;
# the bottom band has to hold tick labels + the axis label + two footnote lines,
# measured: ticks take ~36px, the label ~40px, each footnote ~25px
AX_RECT = (0.250, 0.215, 0.715, 0.675)

# the vertical stack, in canvas units (see `stack()` below)
REF_STRIP = 2.5  # the dashed reference line's label lives here
HEADER_H = 2.1  # group section header
ROW_PITCH = 2.35  # one bar plus its gap
BAR_H = 1.70
DIVIDER = 0.45  # gap straddled by the divider between two groups
PAD_TOP, PAD_BOTTOM = 0.8, 0.8

VALUE_GAP = 0.30  # pt on the x axis between bar end and its value label
TAG_INSET = 0.22  # pt from the left edge of the bar to the inside tag
WHISKER_GAP = 0.08  # pt an inside tag must stop short of the error whisker
DIVIDER_GAP = 0.10  # pt the reference line stops below a group divider

# type sizes, canvas points
HEADER_SIZE, HEADER_MIN = 7.8, 6.2
TAG_SIZE, TAG_MIN = 7.0, 5.4
VALUE_SIZE = 9.6
TICK_SIZE, AXIS_SIZE = 8.6, 9.4
REF_SIZE, REF_MIN = 7.2, 5.6
FOOTNOTE_SIZE, FOOTNOTE_MIN = 6.6, 5.0

# spec palette
INK = "#111827"
BODY = "#374151"
MUTED = "#6b7280"
GRID = "#e5e7eb"
DIVIDER_COLOR = "#d8dee9"
SPINE = "#9ca3af"
NAVY = "#1e3a8a"  # 1 HGT layer, edge decay ON (spec's deep navy)
SLATE = "#64748b"  # 2 HGT layers
TEAL = "#0d9488"  # edge decay off
AMBER = "#d97706"  # LightGBM with auto weights
GRAY = "#94a3b8"  # LightGBM with doc weights
ON_BAR = "#ffffff"  # text drawn inside a bar

# arm -> (bar colour, inside tag).  Baselines carry a tag naming the pairing so a
# reader never compares the wrong two bars; variants carry the measured result.
ARMS: dict[str, tuple[str, str]] = {
    "depth-1layer": (NAVY, "pair baseline · 1 layer, doc weights"),
    "depth-2layer": (SLATE, "+32% params · +66% wall · Δ+0.14 is noise (2/3 seeds)"),
    "full": (NAVY, "pair baseline · edge decay ON, doc weights"),
    "no-temporal-decay": (TEAL, "Δ+0.06 · edge decay is free to drop (2/3 seeds)"),
    "lightgbm": (GRAY, "pair baseline · doc weights"),
    "lightgbm-pos-weights-auto": (AMBER, "Δ−0.12 @CMA but +1.92 @0.5 · graph keeps +2.62 @CMA"),
}

# one header per group, with the qualifier the grid itself records
GROUP_TITLES: dict[str, str] = {
    "Depth & Efficiency": "Depth & Efficiency — same recipe, one extra HGT layer",
    "Temporal Mechanism": "Temporal Mechanism — edge decay on/off, node Δt kept",
    "Tabular Baseline Parity": (
        "Tabular Baseline Parity — same 21 features · at 0.5 the GBM leads 61.78 vs 61.54"
    ),
}


class Box(TypedDict):
    x0: float
    y0: float
    x1: float
    y1: float


class Probe(TypedDict):
    kind: str
    text: Text
    box: Box | None  # None = only "inside the canvas" applies (footnotes)
    note: str


class Stroke(TypedDict):
    kind: str
    box: Box


@dataclass(frozen=True)
class Arm:
    variant: str
    score_cmaes: float
    score_cmaes_sd: float
    score_default: float
    spread: tuple[float, float]
    minutes_per_cell: float
    n_parameters: int

    @staticmethod
    def from_dict(raw: dict[str, object]) -> Arm:
        spread = raw["spread"]
        assert isinstance(spread, list) and len(spread) == 2
        return Arm(
            variant=str(raw["variant"]),
            score_cmaes=float(raw["score_cmaes"]),  # type: ignore[arg-type]
            score_cmaes_sd=float(raw["score_cmaes_sd"]),  # type: ignore[arg-type]
            score_default=float(raw["score_default"]),  # type: ignore[arg-type]
            spread=(float(spread[0]), float(spread[1])),
            minutes_per_cell=float(raw["minutes_per_cell"]),  # type: ignore[arg-type]
            n_parameters=int(raw["n_parameters"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Evidence:
    groups: list[tuple[str, list[Arm]]]
    reference: Arm
    reference_meaning: str
    seeds: tuple[str, ...]
    val_cutoff: str
    anchors: str
    noise_range: tuple[float, float]
    pair_deltas: list[dict[str, object]]


def read_evidence(path: Path) -> Evidence:
    """Load the measured grid, checking every field the chart draws.

    Each check raises instead of asserting: a silently missing key would otherwise
    draw a plausible bar, which is the failure mode that matters for a figure whose
    whole point is that its numbers are the repo's.
    """
    if not path.exists():
        raise SystemExit(
            f'{path} is missing - run `python "figures/slide 13/compute_ablation_landscape.py"`'
        )
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path} could not be read as JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} is not a JSON object")
    raw: dict[str, object] = {str(key): value for key, value in loaded.items()}

    def object_field(name: str) -> dict[str, object]:
        value = raw.get(name)
        if not isinstance(value, dict):
            raise SystemExit(f"{path}: {name!r} is missing or is not an object")
        return {str(key): item for key, item in value.items()}

    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list):
        raise SystemExit(f"{path}: 'groups' is missing or is not an array")
    groups: list[tuple[str, list[Arm]]] = []
    for entry in groups_raw:
        if not isinstance(entry, dict):
            raise SystemExit(f"{path}: a group entry is not an object")
        group = {str(key): value for key, value in entry.items()}
        arms = group.get("arms")
        if not isinstance(arms, list):
            raise SystemExit(f"{path}: group {group.get('group')!r} has no arms array")
        rows = [{str(k): v for k, v in arm.items()} for arm in arms if isinstance(arm, dict)]
        if not rows:
            raise SystemExit(f"{path}: group {group.get('group')!r} has no arm objects")
        groups.append((str(group.get("group")), [Arm.from_dict(row) for row in rows]))

    reference = object_field("reference")
    floor = object_field("noise_floor")
    full_seed_raw = floor.get("full_per_seed")
    if not isinstance(full_seed_raw, dict):
        raise SystemExit(f"{path}: noise_floor.full_per_seed is missing")
    full_seed = {str(key): value for key, value in full_seed_raw.items()}
    numbers = [float(value) for value in full_seed.values() if isinstance(value, (int, float))]
    if not numbers:
        raise SystemExit(f"{path}: noise_floor.full_per_seed holds no numbers")
    protocol = object_field("protocol")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list):
        raise SystemExit(f"{path}: 'seeds' is missing or is not an array")
    deltas = raw.get("pair_deltas")
    if not isinstance(deltas, list):
        raise SystemExit(f"{path}: 'pair_deltas' is missing or is not an array")

    return Evidence(
        groups=groups,
        reference=Arm.from_dict(reference),
        reference_meaning=str(reference.get("meaning", "")),
        seeds=tuple(str(seed) for seed in seeds),
        val_cutoff=str(cast_first(protocol.get("val_cutoff"))),
        anchors=str(cast_first(protocol.get("n_train_anchors"))),
        noise_range=(min(numbers), max(numbers)),
        pair_deltas=[delta for delta in deltas if isinstance(delta, dict)],
    )


def cast_first(value: object) -> object:
    """The protocol fields are stored as the set of values seen; take the only one."""
    if isinstance(value, list):
        if len(value) != 1:
            raise SystemExit(f"expected one protocol value, got {value}")
        return value[0]
    return value


def stack(evidence: Evidence) -> tuple[float, list[float], list[float], list[float]]:
    """Vertical layout: total height, bar centres, header baselines, divider ys."""
    group_h = HEADER_H + 2 * ROW_PITCH
    total = PAD_TOP + REF_STRIP + 3 * group_h + 2 * DIVIDER + PAD_BOTTOM
    y = total - PAD_TOP - REF_STRIP
    bars: list[float] = []
    headers: list[float] = []
    dividers: list[float] = []
    for index, _ in enumerate(evidence.groups):
        y -= HEADER_H
        headers.append(y + HEADER_H * 0.62)
        y -= 0.35  # header to first bar
        for _ in range(2):
            y -= ROW_PITCH * 0.5
            bars.append(y)
            y -= ROW_PITCH * 0.5
        if index < len(evidence.groups) - 1:
            y -= DIVIDER
            dividers.append(y)
    return total, bars, headers, dividers


def draw_width(fig: Figure, text: Text) -> float:
    """Rendered width of a Text artist in pixels."""
    canvas = FigureCanvasAgg(fig)
    return float(text.get_window_extent(renderer=canvas.get_renderer()).width)


def fit_width(fig: Figure, text: Text, limit_px: float, min_size: float) -> bool:
    """Shrink `text` until it fits `limit_px`, or fail and report the size needed."""
    size = float(text.get_fontsize())
    while draw_width(fig, text) > limit_px and size > min_size:
        size -= 0.1
        text.set_fontsize(size)
    return draw_width(fig, text) <= limit_px


def data_box(ax: Axes, x0: float, y0: float, x1: float, y1: float) -> Box:
    low = ax.transData.transform((x0, y0))
    high = ax.transData.transform((x1, y1))
    return Box(
        x0=min(low[0], high[0]),
        y0=min(low[1], high[1]),
        x1=max(low[0], high[0]),
        y1=max(low[1], high[1]),
    )


def px_per_x_unit() -> float:
    """Canvas pixels per unit of the score axis, for width budgeting."""
    return AX_RECT[2] * FIG_W * DPI / (X_LIM[1] - X_LIM[0])


def text_width_px(fig: Figure, ax: Axes, label: str, size: float) -> float:
    """Width a label would occupy at `size`, measured on an invisible probe."""
    probe = ax.text(0, 0, label, fontsize=size, alpha=0.0)
    width = draw_width(fig, probe)
    probe.remove()
    return width


def footnote_variants(evidence: Evidence) -> list[tuple[str, ...]]:
    """Two provenance lines, each with progressively shorter fallbacks."""
    seeds = "/".join(evidence.seeds)
    lo, hi = evidence.noise_range
    return [
        (
            f"bar = 3-seed mean ({seeds}) · whisker = ±1 sd · "
            f"score = ½(macro-F1 + micro-F1) · {evidence.anchors} anchors, "
            f"val cutoff {evidence.val_cutoff} · x truncated at 62",
            f"bar = 3-seed mean ({seeds}) · whisker = ±1 sd · "
            f"score = ½(macro-F1 + micro-F1) · val cutoff {evidence.val_cutoff}",
            "bar = 3-seed mean · whisker = ±1 sd · score = ½(macro-F1 + micro-F1)",
        ),
        (
            f"source: experiments/ablations/ablation_summary.csv · Δ under ~0.2 is "
            f"seed noise (`full` alone spans {lo:.2f}–{hi:.2f})",
            "source: experiments/ablations/ablation_summary.csv · Δ under ~0.2 is seed noise",
        ),
    ]


def footnote(fig: Figure, evidence: Evidence) -> list[Probe]:
    """Two provenance lines; each keeps the longest variant that stays readable."""
    limit = AX_RECT[2] * FIG_W * DPI * 0.98
    probes: list[Probe] = []
    for y, variants in zip((0.072, 0.028), footnote_variants(evidence), strict=True):
        for line in variants:
            text = fig.text(
                0.5, y, line, ha="center", va="center", fontsize=FOOTNOTE_SIZE, color=MUTED
            )
            if fit_width(fig, text, limit, FOOTNOTE_MIN):
                probes.append(Probe(kind="footnote", text=text, box=None, note=f"y={y:g}"))
                break
            text.remove()
        else:
            raise SystemExit(f"no footnote variant fits at {FOOTNOTE_MIN} pt")
    return probes


def build() -> tuple[Figure, list[Probe], list[Stroke]]:
    evidence = read_evidence(EVIDENCE)
    total, bar_ys, header_ys, divider_ys = stack(evidence)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_alpha(0.0 if TRANSPARENT else 1.0)
    if not TRANSPARENT:
        fig.patch.set_facecolor("#ffffff")
    ax.set_position(AX_RECT)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0.0, total)

    ax.xaxis.grid(True, linestyle=(0, (3, 3)), linewidth=0.7, color=GRID)
    ax.set_axisbelow(True)
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0, labelsize=TICK_SIZE, colors=BODY, pad=6)
    ax.tick_params(axis="x", labelsize=TICK_SIZE, colors=BODY)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel(X_LABEL, fontsize=AXIS_SIZE, color=INK, labelpad=8)

    probes: list[Probe] = []
    stroke_rects: list[tuple[str, float, float, float, float]] = []
    rows: list[tuple[Arm, float]] = []

    def place(
        kind: str,
        xy: tuple[float, float],
        label: str,
        size: float,
        box: Box | None,
        *,
        color: str = INK,
        ha: str = "left",
        va: str = "center",
        min_size: float | None = None,
        note: str = "",
    ) -> Text:
        text = ax.text(*xy, label, fontsize=size, color=color, ha=ha, va=va)
        if min_size is not None:
            limit = box["x1"] - box["x0"] if box is not None else AX_RECT[2] * FIG_W * DPI
            if not fit_width(fig, text, limit, min_size):
                raise SystemExit(f"{kind} does not fit at {min_size} pt: {label}")
        probes.append(Probe(kind=kind, text=text, box=box, note=note))
        return text

    def stroke(kind: str, x0: float, y0: float, x1: float, y1: float) -> None:
        """Record a drawn mark in data coords; converted to pixels once, at the end.

        Recording geometry instead of artists keeps the checker's rules independent
        of draw order: a stroke's pixel box comes from the same transform the text
        boxes use.
        """
        stroke_rects.append((kind, x0, y0, x1, y1))

    # --- group sections and their bars -----------------------------------------
    index = 0
    for group_index, (group, arms) in enumerate(evidence.groups):
        title = GROUP_TITLES.get(group, group)
        top_of_group = header_ys[group_index] + HEADER_H * 0.4
        bottom_of_group = bar_ys[index + 1] - ROW_PITCH * 0.5
        place(
            "group_header",
            (X_LIM[0] + 0.12, header_ys[group_index]),
            title,
            HEADER_SIZE,
            data_box(ax, X_LIM[0] + 0.12, bottom_of_group, X_LIM[1], top_of_group),
            color=INK,
            min_size=HEADER_MIN,
            note=f"group={group}",
        )
        for arm in arms:
            y = bar_ys[index]
            color, tag = ARMS[arm.variant]
            half = BAR_H / 2
            ax.barh(
                y=y,
                width=arm.score_cmaes - X_LIM[0],
                left=X_LIM[0],
                height=BAR_H,
                align="center",
                color=color,
                edgecolor="none",
                zorder=2,
            )
            bar_box = data_box(ax, X_LIM[0], y - half, arm.score_cmaes, y + half)

            # +-1 sample sd whisker through the bar end, caps at both ends
            lo = arm.score_cmaes - arm.score_cmaes_sd
            hi = arm.score_cmaes + arm.score_cmaes_sd
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.2, zorder=4)
            stroke("error_whisker", lo, y - 0.03, hi, y + 0.03)
            cap = BAR_H * 0.30
            for value in (lo, hi):
                ax.plot([value, value], [y - cap, y + cap], color=color, linewidth=1.2, zorder=4)
                stroke("error_cap", value - 0.01, y - cap, value + 0.01, y + cap)
            # the value sits clear of the whisker; a probe measures how wide it is,
            # which fixes where a tag written after it would have to start
            value_label = f"{arm.score_cmaes:.2f}"
            value_start = arm.score_cmaes + VALUE_GAP
            value_px = text_width_px(fig, ax, value_label, VALUE_SIZE)
            tag_start = value_start + (value_px + 12.0) / px_per_x_unit()

            # a long bar carries its tag inside, in white, but only up to where the
            # whisker starts; a short bar (the two LightGBM arms) pushes the tag
            # after the value label instead of over the error bar
            room_px = (lo - X_LIM[0] - TAG_INSET - WHISKER_GAP) * px_per_x_unit()
            inside = text_width_px(fig, ax, tag, TAG_SIZE) <= room_px
            place(
                "value_label",
                (value_start, y),
                value_label,
                VALUE_SIZE,
                data_box(
                    ax,
                    value_start - 0.05,
                    y - half,
                    X_LIM[1] if inside else tag_start,
                    y + half,
                ),
                color=INK,
                note=f"arm={arm.variant}",
            )
            if inside:
                place(
                    "bar_tag",
                    (X_LIM[0] + TAG_INSET, y),
                    tag,
                    TAG_SIZE,
                    bar_box,
                    color=ON_BAR,
                    min_size=TAG_MIN,
                    note=f"arm={arm.variant} inside",
                )
            else:
                place(
                    "bar_tag",
                    (tag_start, y),
                    tag,
                    TAG_SIZE,
                    data_box(ax, tag_start - 0.02, y - half, X_LIM[1], y + half),
                    color=BODY,
                    min_size=TAG_MIN,
                    note=f"arm={arm.variant} after value",
                )
            rows.append((arm, y))
            index += 1
        if group_index < len(evidence.groups) - 1:
            y_div = divider_ys[group_index]
            ax.axhline(y_div, color=DIVIDER_COLOR, linewidth=0.9, zorder=1)
            stroke("group_divider", X_LIM[0], y_div - 0.02, X_LIM[1], y_div + 0.02)

    # --- the submission recipe as a reference line, not a bar ------------------
    ref = evidence.reference
    # the line stops at the divider above the tabular group: the +2.62 parity claim
    # is carried by that group's tag, and a full-height line would cut through the
    # group header and the tag that follows the short bar
    line_bottom = divider_ys[len(evidence.groups) - 2] + DIVIDER_GAP
    ax.plot(
        [ref.score_cmaes, ref.score_cmaes],
        [line_bottom, total],
        color=NAVY,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.75,
        zorder=3,
    )
    stroke("reference_line", ref.score_cmaes - 0.012, line_bottom, ref.score_cmaes + 0.012, total)
    place(
        "reference_label",
        (X_LIM[0] + 0.12, total - PAD_TOP - REF_STRIP * 0.45),
        f"reference · {ref.variant} = {ref.score_cmaes:.2f} @CMA "
        f"({ref.score_default:.2f} @0.5) · best mean in the grid",
        REF_SIZE,
        data_box(ax, X_LIM[0] + 0.12, total - PAD_TOP - REF_STRIP, X_LIM[1], total - PAD_TOP),
        color=NAVY,
        min_size=REF_MIN,
        note=evidence.reference_meaning,
    )

    # y tick labels: the arm names, in the same order as the bars
    names = [
        {
            "depth-1layer": "1 HGT layer",
            "depth-2layer": "2 HGT layers",
            "full": "HGT full (edge decay)",
            "no-temporal-decay": "HGT w/o edge decay",
            "lightgbm": "LightGBM (doc weights)",
            "lightgbm-pos-weights-auto": "LightGBM (auto weights)",
        }[arm.variant]
        for arm, _ in rows
    ]
    ax.set_yticks([y for _, y in rows])
    ax.set_yticklabels(names)
    for label in ax.get_yticklabels():
        label.set_fontsize(TICK_SIZE)

    probes.extend(footnote(fig, evidence))
    strokes = [
        Stroke(kind=kind, box=data_box(ax, x0, y0, x1, y1))
        for kind, x0, y0, x1, y1 in stroke_rects
    ]
    return fig, probes, strokes


def main() -> None:
    fig, _, _ = build()
    fig.savefig(OUT, transparent=TRANSPARENT, facecolor=fig.get_facecolor())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
