#!/usr/bin/env python3
"""Slide 8 figure: retention probability against idle time on two idle clocks.

Two curves, one quantity:

  Curve A  General Activity (indigo, steep) - P(cohort user opens an issue in the
           next 7 days | dt days since their last event of ANY activity type);
  Curve B  Role Habit (teal, gentle)       - the same probability, but dt counts
           days since their last *issue*, i.e. the role-specific habit clock.

Both are divided by their own dt = 0 value, so the y axis is retention relative
to an active user: 1.0 = as likely as a fully current user. That normalization
is what makes the two curves comparable on one axis and is stated on the canvas.
The solid lines are the fitted decays; the neutral dots are the observed
retention in each whole-day bin of Delta t, labelled in the legend the same way
the slide 7 figure labels its observed rate.

Measured result (see `compute_retention_curves.py` in this folder): the general
clock collapses fastest - tau 3.7 d, half-life 2.5 d - while the habit clock
holds up, tau 6.0 d, half-life 4.1 d, still ~0.49 of its day-0 level a week after
the user's last issue. A user idle on the platform overall is a much stronger
dropout signal than a user idle in one role.

Everything drawn here is read from `retention_curve_data.csv` and
`retention_fits.json` written by `compute_retention_curves.py` next to this file;
the script refuses to run without them rather than plotting invented numbers.

Run from the repo root:
    /train-data-5-hdd/charles/miniconda3/envs/bda_asg2/bin/python \
        "figures/slide 8/make_dual_temporal_decay_minimal.py"
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent  # figures/slide 8
CURVE_CSV = HERE / "retention_curve_data.csv"
FITS_JSON = HERE / "retention_fits.json"
OUT_PNG = HERE / "dual_temporal_decay_minimal.png"

# --- the two plotted series ---------------------------------------------------
# Same role target on both curves, so the y axis means one thing; the curves
# differ only in which idle clock feeds dt ("any" activity vs the same activity).
SERIES_TARGET = "open_issue"
AXIS_GENERAL = "any"  # Curve A
AXIS_HABIT = "same"  # Curve B

# --- canvas -------------------------------------------------------------------
FIG_W, FIG_H, DPI = 6.0, 4.5, 300  # compact 4:3
TRANSPARENT = False  # True -> transparent background instead of pure white
# generous margins: the chart has to sit beside slide text without wasting space
AX_RECT = (0.145, 0.165, 0.690, 0.660)

X_MAX = 28.0
Y_MAX = 1.0
X_TICKS = np.arange(0.0, 28.001, 7.0)  # weekly ticks
Y_TICKS = np.arange(0.0, 1.001, 0.2)

# --- style --------------------------------------------------------------------
INK = "#111827"
BODY = "#374151"
MUTED = "#6b7280"
GRID = "#e5e7eb"
INDIGO = "#4f46e5"  # solid: general activity clock, steep
TEAL = "#0d9488"  # solid: role habit clock, gentle
LINE_A_LW = 2.6
LINE_B_LW = 2.2
DOT_ALPHA = 0.85  # measured bins behind the fitted curves (slide 7 convention)
DOT_SIZE = 13
DOT_COLOR = "#94a3b8"  # one neutral marker colour, so a single legend entry labels both clocks

X_LABEL = "Idle Time (Days)"
Y_LABEL = "Retention Probability"


def read_json(path: Path) -> dict[str, Any]:
    """Parse a JSON artifact, with a readable failure instead of a traceback."""
    try:
        parsed = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        msg = f"could not read {path.name}: {exc}"
        raise SystemExit(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"{path.name} is not a JSON object"
        raise SystemExit(msg)
    return parsed


def read_curve_rows(path: Path) -> list[dict[str, str]]:
    """Read the measured bins with the stdlib csv module (no dataframe dependency)."""
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        msg = f"could not read {path.name}: {exc}"
        raise SystemExit(msg) from exc


def find_series(provenance: dict[str, Any], axis: str) -> dict[str, Any]:
    """The recorded fit block for one axis of the plotted target."""
    for entry in provenance.get("series", []):
        if isinstance(entry, dict) and entry.get("axis") == axis and entry.get("target") == SERIES_TARGET:
            return entry
    msg = f"no fits recorded for axis={axis} target={SERIES_TARGET} in {FITS_JSON.name}"
    raise SystemExit(msg)


def retention_at(x: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    """floor + amp * exp(-x / tau) on the dt = 1.0-normalized scale."""
    return float(fit["floor"]) + float(fit["amp"]) * np.exp(-x / float(fit["tau_days"]))


def load_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float,
                        dict[str, float], dict[str, float], dict[str, Any]]:
    """Measured bins and fits for both clocks, normalized to their own dt = 0 value."""
    if not CURVE_CSV.exists() or not FITS_JSON.exists():
        msg = (
            f"missing {CURVE_CSV.name} / {FITS_JSON.name}; run "
            f'"figures/slide 8/compute_retention_curves.py" first'
        )
        raise SystemExit(msg)
    provenance = read_json(FITS_JSON)

    bins: dict[str, tuple[list[float], list[float], float]] = {
        AXIS_GENERAL: ([], [], 0.0),
        AXIS_HABIT: ([], [], 0.0),
    }
    for row in read_curve_rows(CURVE_CSV):
        axis = row.get("axis", "")
        if axis not in bins or row.get("target") != SERIES_TARGET or row.get("supported") != "True":
            continue
        dt_list, p_list, pairs = bins[axis]
        try:
            dt_list.append(float(row["dt_days"]))
            p_list.append(float(row["p_smooth"]))
            bins[axis] = (dt_list, p_list, pairs + float(row["n"]))
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"malformed bin row in {CURVE_CSV.name}: {exc}"
            raise SystemExit(msg) from exc

    fits = {axis: find_series(provenance, axis) for axis in bins}
    scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for axis, (dt_list, p_list, _pairs) in bins.items():
        if not dt_list:
            msg = f"no supported bins for axis={axis} target={SERIES_TARGET} in {CURVE_CSV.name}"
            raise SystemExit(msg)
        retention = fits[axis].get("retention", {})
        base = float(retention.get("raw_at_zero", float("nan")))
        if not np.isfinite(base) or base <= 0.0:
            msg = f"no usable dt = 0 value for axis={axis}; rerun the compute script"
            raise SystemExit(msg)
        scaled[axis] = (np.asarray(dt_list), np.asarray(p_list) / base)

    return (
        scaled[AXIS_GENERAL][0],
        scaled[AXIS_GENERAL][1],
        scaled[AXIS_HABIT][0],
        scaled[AXIS_HABIT][1],
        bins[AXIS_GENERAL][2],
        bins[AXIS_HABIT][2],
        fits[AXIS_GENERAL]["retention"],
        fits[AXIS_HABIT]["retention"],
        provenance,
    )


def fit_text(fig: Figure, text, limit_points: float, min_size: float = 4.6) -> bool:
    """Shrink a footnote until its measured width fits limit_points; True if it fits."""
    # 1 figure unit on x = FIG_W inches = FIG_W * 72 points, so a measured width
    # in points compares directly against the limit.
    size = text.get_fontsize()
    while True:
        fig.canvas.draw()
        if text.get_window_extent().width <= limit_points:
            return True
        if size <= min_size:
            return False
        size -= 0.2
        text.set_fontsize(size)


def footnote(fig: Figure, fit_a: dict[str, float], fit_b: dict[str, float]) -> None:
    """One small provenance line, no box, no arrow, shortened until it fits.

    Pair counts and the anchor span stay out of the figure (they live in
    `retention_fits.json`); the canvas only states what the y axis is normalized
    to and the two half-lives, which is what a reader needs beside the slide text.
    """
    halves = f"half-life {fit_a['half_life_days']:.1f} d vs {fit_b['half_life_days']:.1f} d"
    # longest first; the first variant that fits the axes width is the one drawn
    variants = [
        f"{SERIES_TARGET} role \u00b7 both curves normalised to their own day-0 value \u00b7 {halves}",
        f"{SERIES_TARGET} role \u00b7 normalised to day-0 value \u00b7 {halves}",
        f"normalised to day-0 value \u00b7 {halves}",
    ]
    text = fig.text(AX_RECT[0], 0.036, variants[-1], fontsize=6.2, color=MUTED, ha="left", va="bottom")
    limit = AX_RECT[2] * FIG_W * 72.0
    for line in variants:
        text.set_text(line)
        if fit_text(fig, text, limit):
            break


def build() -> Figure:
    dt_a, p_a, dt_b, p_b, pairs_a, pairs_b, fit_a, fit_b, provenance = load_pair()

    fig = Figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_alpha(0.0 if TRANSPARENT else 1.0)
    fig.patch.set_facecolor("none" if TRANSPARENT else "white")
    ax = fig.add_axes(AX_RECT)
    ax.set_facecolor("none" if TRANSPARENT else "white")

    x_fine = np.linspace(0.0, X_MAX, 561)

    # measured bins first, in one neutral colour so a single legend entry names
    # them for both clocks, the way the slide 7 figure names its observed rate
    for dt_bins, p_bins in ((dt_a, p_a), (dt_b, p_b)):
        ax.scatter(dt_bins, p_bins, s=DOT_SIZE, color=DOT_COLOR, alpha=DOT_ALPHA,
                   linewidths=0.0, zorder=2)
    ax.plot(x_fine, retention_at(x_fine, fit_a), color=INDIGO, lw=LINE_A_LW,
            linestyle="-", zorder=4, solid_capstyle="round")
    ax.plot(x_fine, retention_at(x_fine, fit_b), color=TEAL, lw=LINE_B_LW,
            linestyle="-", zorder=3, solid_capstyle="round")

    # minimalist frame: left + bottom only, subtle dashed grid
    ax.grid(True, which="major", color=GRID, lw=0.7, linestyle=(0, (3, 3)), zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=8.5, colors=BODY,
                   length=3.5, width=0.8, pad=4)
    for tick in ax.get_yticklabels():
        tick.set_color(BODY)

    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(0.0, Y_MAX)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(Y_TICKS)
    ax.set_xticklabels([f"{int(v)}" for v in X_TICKS])
    ax.set_yticklabels([f"{v:.1f}" for v in Y_TICKS])

    ax.set_xlabel(X_LABEL, fontsize=9.5, color=INK, labelpad=7)
    ax.set_ylabel(Y_LABEL, fontsize=9.5, color=INK, labelpad=7)

    legend = ax.legend(
        handles=[
            Line2D([], [], color=DOT_COLOR, marker="o", markersize=4.6, alpha=DOT_ALPHA,
                   linestyle="None", markeredgewidth=0.0),
            Line2D([], [], color=INDIGO, lw=LINE_A_LW, linestyle="-"),
            Line2D([], [], color=TEAL, lw=LINE_B_LW, linestyle="-"),
        ],
        labels=[
            "Observed Retention per \u0394t (Measured)",
            f"General Activity (\u0394t_any: \u03c4 \u2248 {fit_a['tau_days']:.1f} d)",
            f"Role Habit (\u0394t_same: \u03c4 \u2248 {fit_b['tau_days']:.1f} d)",
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.2,
        handlelength=2.2,
        borderaxespad=0.6,
        labelspacing=0.7,
    )
    for text in legend.get_texts():
        text.set_color(BODY)

    footnote(fig, fit_a, fit_b)
    return fig


def main() -> int:
    fig = build()
    fig.savefig(OUT_PNG, dpi=DPI, transparent=TRANSPARENT)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
