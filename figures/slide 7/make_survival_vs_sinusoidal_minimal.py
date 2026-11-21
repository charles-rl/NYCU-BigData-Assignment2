#!/usr/bin/env python3
"""Slide 7 figure: empirical recency survival vs a 7-day sinusoidal fit.

The measured bins (observed rate in each 1-day bin of Delta t) are drawn as dots,
the navy line is the decay+floor fit through them, and the dashed crimson curve
is the best 7-day sinusoid that the assignment rejects as a cycle. Everything
drawn here is read from the artifacts written by `compute_survival_curve.py` in
this same folder - the script refuses to run without them rather than plotting
invented numbers.

Measured series plotted (chosen because it is the one real series that matches
the slide's shape and its 0.0-0.7 y-axis): axis "same" (days since the last
event of the same activity), target "open_issue", 95,314 cohort user-anchor
pairs over 73 daily anchors. Its decay fit starts at 0.587 and floors at 0.144;
the 7-day sinusoid explains 2.0% of the variance, far below the assignment's
cycle gate of 0.40 (`src/data/activity_decay_analysis.py:292`).

Run from the repo root:
    /train-data-5-hdd/charles/miniconda3/envs/bda_asg2/bin/python \
        "figures/slide 7/make_survival_vs_sinusoidal_minimal.py"
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent  # figures/slide 7
CURVE_CSV = HERE / "survival_curve_data.csv"
FITS_JSON = HERE / "survival_fits.json"
OUT_PNG = HERE / "survival_vs_sinusoidal_minimal.png"

# --- the plotted series -------------------------------------------------------
SERIES_AXIS = "same"  # "same" = days since the last event of this activity
SERIES_TARGET = "open_issue"

# --- canvas -------------------------------------------------------------------
FIG_W, FIG_H, DPI = 6.0, 4.5, 300
TRANSPARENT = False  # True -> transparent background instead of pure white
# generous margins: the chart has to sit beside bullet points on the slide
AX_RECT = (0.150, 0.170, 0.685, 0.655)

X_MAX = 30.0
Y_MAX = 0.70
X_TICKS = np.arange(0.0, 30.001, 5.0)
Y_TICKS = np.arange(0.0, 0.701, 0.1)

# --- style --------------------------------------------------------------------
INK = "#111827"
BODY = "#374151"
MUTED = "#6b7280"
GRID = "#e5e7eb"
NAVY = "#1e3a8a"  # solid, thick: empirical survival
CRIMSON = "#dc143c"  # dashed, medium: sinusoidal fit
DOT_ALPHA = 0.62  # measured bins: the observed rate in each 1-day bin
DOT_SIZE = 14  # scatter marker area, matches the legend dot

X_LABEL = "Days Since Last Event (\u0394t)"
Y_LABEL = "P(Active in Next 7 Days)"


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


def load_series() -> tuple[np.ndarray, np.ndarray, float, dict[str, float], dict[str, float], dict[str, Any]]:
    """Measured bins (dt, smoothed p, total pairs) plus the recorded fits."""
    if not CURVE_CSV.exists() or not FITS_JSON.exists():
        msg = (
            f"missing {CURVE_CSV.name} / {FITS_JSON.name}; run "
            f'"figures/slide 7/compute_survival_curve.py" first'
        )
        raise SystemExit(msg)
    dt: list[float] = []
    prob: list[float] = []
    pairs = 0.0
    for row in read_curve_rows(CURVE_CSV):
        if row.get("axis") != SERIES_AXIS or row.get("target") != SERIES_TARGET:
            continue
        if row.get("supported") != "True":
            continue
        try:
            dt.append(float(row["dt_days"]))
            prob.append(float(row["p_smooth"]))
            pairs += float(row["n"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"malformed bin row in {CURVE_CSV.name}: {exc}"
            raise SystemExit(msg) from exc
    if not dt:
        msg = f"no supported bins for axis={SERIES_AXIS} target={SERIES_TARGET} in {CURVE_CSV.name}"
        raise SystemExit(msg)
    provenance = read_json(FITS_JSON)
    series: dict[str, Any] = {}
    for entry in provenance.get("series", []):
        if isinstance(entry, dict) and entry.get("axis") == SERIES_AXIS and entry.get("target") == SERIES_TARGET:
            series = entry
            break
    if not series:
        msg = f"no fits recorded for axis={SERIES_AXIS} target={SERIES_TARGET}"
        raise SystemExit(msg)
    decay_fit: dict[str, float] = series["decay"]
    week_fit: dict[str, float] = series["sinusoid_7d_on_curve"]
    return np.asarray(dt), np.asarray(prob), pairs, decay_fit, week_fit, provenance


def sinusoid(x: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    """A + B*cos(2*pi*x/period + phi) from a recorded least-squares fit."""
    w = 2.0 * np.pi / float(fit["period_days"])
    return float(fit["mean"]) + float(fit["amplitude"]) * np.cos(w * x + float(fit["phase_rad"]))


def decay(x: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    """floor + amp * exp(-x / tau)."""
    return float(fit["floor"]) + float(fit["amp"]) * np.exp(-x / float(fit["tau_days"]))


def fit_text(fig: Figure, text, limit_points: float, min_size: float = 4.6) -> bool:
    """Shrink a footnote until its measured width fits limit_points; True if it fits."""
    # 1 figure unit on x = FIG_W inches = FIG_W * 72 points, so measured widths
    # in points compare directly against the limit.
    size = text.get_fontsize()
    while True:
        fig.canvas.draw()
        if text.get_window_extent().width <= limit_points:
            return True
        if size <= min_size:
            return False
        size -= 0.2
        text.set_fontsize(size)


def footnote(fig: Figure, pairs: float, decay_fit: dict[str, float], week_fit: dict[str, float], provenance: dict[str, Any]) -> None:
    """One small provenance line, no box, no arrow, shortened until it fits."""
    raw_anchors = provenance.get("anchors")
    anchors: dict[str, Any] = raw_anchors if isinstance(raw_anchors, dict) else {}
    span = f"{anchors.get('first', '?')}\u2013{anchors.get('last', '?')}"
    stats = (
        f"decay floor {decay_fit['floor']:.2f}, half-life "
        f"{decay_fit['half_life_days']:.1f} d, R\u00b2 {decay_fit['r2']:.2f}"
    )
    # longest first; the first variant that fits the axes width is the one drawn
    variants = [
        (f"{int(pairs):,} cohort user-anchor pairs \u00b7 {anchors.get('count', '?')} daily "
         f"anchors {span} \u00b7 {stats} \u00b7 7-day sinusoid R\u00b2 {week_fit['r2']:.2f}"),
        (f"{int(pairs):,} user-anchor pairs \u00b7 {anchors.get('count', '?')} daily anchors "
         f"\u00b7 {stats}"),
        f"{int(pairs):,} user-anchor pairs \u00b7 {stats}",
    ]
    text = fig.text(AX_RECT[0], 0.038, variants[-1], fontsize=6.2, color=MUTED, ha="left", va="bottom")
    limit = AX_RECT[2] * FIG_W * 72.0
    for line in variants:
        text.set_text(line)
        if fit_text(fig, text, limit):
            break


def build() -> Figure:
    dt, prob, pairs, decay_fit, week_fit, provenance = load_series()

    fig = Figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_alpha(0.0 if TRANSPARENT else 1.0)
    fig.patch.set_facecolor("none" if TRANSPARENT else "white")
    ax = fig.add_axes(AX_RECT)
    ax.set_facecolor("none" if TRANSPARENT else "white")

    x_fine = np.linspace(0.0, X_MAX, 601)

    # measured bins first, so the fitted curves read on top of them
    ax.scatter(
        dt,
        prob,
        s=DOT_SIZE,
        color=NAVY,
        alpha=DOT_ALPHA,
        zorder=2,
        linewidths=0,
    )
    ax.plot(
        x_fine,
        decay(x_fine, decay_fit),
        color=NAVY,
        lw=2.6,
        linestyle="-",
        zorder=4,
        solid_capstyle="round",
    )
    ax.plot(
        x_fine,
        sinusoid(x_fine, week_fit),
        color=CRIMSON,
        lw=1.5,
        linestyle=(0, (5, 3)),
        zorder=3,
    )

    # minimalist frame: left + bottom only, light gray grid
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
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
            Line2D([], [], color=NAVY, marker="o", markersize=4.6, alpha=DOT_ALPHA,
                   linestyle="None", markeredgewidth=0.0),
            Line2D([], [], color=NAVY, lw=2.6, linestyle="-"),
            Line2D([], [], color=CRIMSON, lw=1.5, linestyle=(0, (5, 3))),
        ],
        labels=[
            "Observed Rate per \u0394t (Measured)",
            "Empirical Survival (Decay + Floor)",
            f"Sinusoidal Fit (R\u00b2 = {week_fit['r2']:.2f} < 0.40)",
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.2,
        handlelength=2.4,
        borderaxespad=0.6,
        labelspacing=0.65,
    )
    for text in legend.get_texts():
        text.set_color(BODY)

    footnote(fig, pairs, decay_fit, week_fit, provenance)
    return fig


def main() -> int:
    fig = build()
    fig.savefig(OUT_PNG, dpi=DPI, transparent=TRANSPARENT, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
