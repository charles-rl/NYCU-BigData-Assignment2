#!/usr/bin/env python3
"""Slide 3 figure: corpus data-quality audit timeline (publication figure).

Produces ``figures/slide 3/corpus_data_quality_audit.png`` — a horizontal three-track
timeline of the GH Archive corpus (2025-12-01 .. 2026-11-20):

  track 1  hourly sanity rate as a colour-coded status bar,
  track 2  the model windows actually selected,
  track 3  daily event density (PR+Issue volume, PushEvent volume).

Window choices follow ``docs/VALIDATION_DATA.md`` (§2 the November outage, §3 audit
definitions, §4 validation-window selection, §10 multi-anchor training). Track 3 is drawn
from the audit output ``raw_data/coverage.csv`` (written by ``scripts/verify_raw_data.py``);
a per-day extract is cached beside this script as ``corpus_daily_audit.csv`` so the figure
can be rebuilt without the 216 GB raw corpus on hand.

Nothing sits below the tracks except the shared date axis: the metric callout boxes that
used to fill the bottom band are gone, so the figure height stops at the month labels.

Run:            python "figures/slide 3/make_corpus_audit_figure.py"
Layout audit:   python "figures/slide 3/check_layout.py"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]  # pyrightconfig.json venvPath is stale on this box; pandas resolves and runs in env bda_asg2
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent  # figures/slide 3
REPO = HERE.parent.parent
CACHE = HERE / "corpus_daily_audit.csv"
OUT = HERE / "corpus_data_quality_audit.png"

ACTIVITIES = [
    "ok_open_pr",
    "ok_open_issue",
    "ok_issue_comment",
    "ok_pr_review",
    "ok_pr_review_comment",
    "ok_push",
]

# --------------------------------------------------------------------------- palette --

INK = "#1c2430"
MUTED = "#5b6b7c"
GRID = "#dde2e8"

GREEN = "#2f8f4e"
GREEN_EDGE = "#1d6a37"
ORANGE = "#e8912a"
ORANGE_EDGE = "#8f5207"
RED = "#d81f26"
RED_EDGE = "#8d1418"
YELLOW = "#eac53f"
YELLOW_EDGE = "#7d6209"

BLUE = "#1f6feb"  # golden validation week
SLATE = "#3d5a80"  # PR/Issue density + training anchors
PURPLE = "#7b4fa6"  # official test week

# --------------------------------------------------------------------------- geometry --

FIG_W, FIG_H, DPI = 14.0, 5.1, 300
L, R = 0.070, 0.958  # left / right edge of every track

X0, X1 = "2025-11-27", "2026-12-06"  # x-axis padding around the corpus

AX_T = (L, 0.556, R - L, 0.330)  # tracks 1 + 2
AX_D = (L, 0.168, R - L, 0.340)  # track 3
AX_B = (L, 0.112, R - L, 0.056)  # shared date axis: the only element below the tracks

BAR_Y, BAR_H = 0.795, 0.19  # track-1 status bar, in AX_T y-units (0..1)
BR_Y, BR_TICK = 0.36, 0.10  # track-2 bracket baseline and arm height

# (start, end, fill, edge, hatch, short caption, caption date, ha, caption y in AX_T units)
PHASES = [
    ("2025-12-01", "2026-03-14", GREEN, GREEN_EDGE, None, "CONTIGUOUS  ·  98.4%", None, None, 0.795),
    ("2026-03-15", "2026-10-26", ORANGE, ORANGE_EDGE, "///", "INTERMITTENT DROPOUTS", None, None, 0.795),
    ("2026-10-27", "2026-11-08", RED, RED_EDGE, None, "OUTAGE", "2026-10-21", "right", 0.945),
    ("2026-11-09", "2026-11-20", YELLOW, YELLOW_EDGE, None, "INCOMPLETE", "2026-12-06", "right", 0.600),
]

# (start, end, colour, short caption, caption anchor date, ha)
WINDOWS = [
    ("2025-12-22", "2026-02-08", SLATE, "8 training anchors", "2025-12-22", "left"),
    ("2026-02-15", "2026-02-22", BLUE, "validation week", "2026-02-26", "left"),
    ("2026-11-22", "2026-11-29", PURPLE, "test week", "2026-11-19", "right"),
]

DENS_MAX = 1_000_000  # left axis, PR+Issue events/day
PUSH_MAX = 4_300_000  # right axis, PushEvent/day


def ts(s: str) -> float:
    """Date -> matplotlib date number."""
    return float(mdates.date2num(pd.Timestamp(s)))


def xfrac(s: str) -> float:
    """Date -> figure-x fraction (every track shares this span)."""
    return L + (ts(s) - ts(X0)) / (ts(X1) - ts(X0)) * (R - L)


# ------------------------------------------------------------------------------- data --


def load_daily() -> pd.DataFrame:
    """Per-day audit aggregates, indexed by day."""
    src = REPO / "raw_data" / "coverage.csv"
    if src.exists():
        cov = pd.read_csv(src, parse_dates=["day"])
        cov["all_ok"] = cov[ACTIVITIES].all(axis=1)
        agg = cov.groupby("day").agg(
            hours=("hour", "count"),
            events=("events", "sum"),
            push=("PushEvent", "sum"),
            pr=("PullRequestEvent", "sum"),
            issues=("IssuesEvent", "sum"),
            hours_ok=("all_ok", "sum"),
        )
        agg["pr_issue"] = agg["pr"] + agg["issues"]
        agg["sanity"] = agg["hours_ok"] / agg["hours"]
        agg.to_csv(CACHE)
        return agg
    if CACHE.exists():  # rebuild from the cached extract
        return pd.read_csv(CACHE, parse_dates=["day"]).set_index("day")
    raise SystemExit(f"no audit data: neither {src} nor {CACHE} exists")


# ----------------------------------------------------------------------------- figure --


def build() -> Figure:
    daily = load_daily()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.edgecolor": "#9aa4b1",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    axT = fig.add_axes(AX_T)
    axD = fig.add_axes(AX_D, sharex=axT)
    axB = fig.add_axes(AX_B, frameon=False)

    x0, x1 = ts(X0), ts(X1)
    for ax in (axT, axD, axB):
        ax.set_xlim(x0, x1)
    axT.set_ylim(0, 1)
    axT.set_xticks([])
    axT.set_yticks([])
    for spine in axT.spines.values():
        spine.set_visible(False)

    # month grid, regime tint, and the two highlighted weeks — shared by every track
    for a, b, colour, *_ in PHASES:
        for ax in (axT, axD):
            ax.axvspan(ts(a), ts(b) + 1, color=colour, alpha=0.07, lw=0, zorder=0)
    for m in pd.date_range("2025-12-01", "2026-12-01", freq="MS"):
        for ax in (axT, axD):
            ax.axvline(ts(m), color=GRID, lw=0.5, zorder=-1)
    for ax in (axT, axD):
        ax.axvspan(ts("2026-11-22"), ts("2026-11-29"), color=PURPLE, alpha=0.15, lw=0, zorder=1)
        ax.axvspan(ts("2026-02-15"), ts("2026-02-22"), color=BLUE, alpha=0.17, lw=0, zorder=2)

    # -------------------------------------------------------------- track 1: sanity --
    for a, b, colour, edge, hatch, caption, cap_x, cap_ha, cap_y in PHASES:
        axT.barh(
            BAR_Y,
            width=ts(b) + 1 - ts(a),
            left=ts(a),
            height=BAR_H,
            color=colour,
            edgecolor=edge,
            hatch=hatch,
            linewidth=0.9,
            zorder=3,
        )
        if cap_x is None:  # wide segment: caption inside the bar
            x, ha = ts(a) + (ts(b) + 1 - ts(a)) / 2, "center"
        else:  # narrow segment: caption just outside the bar
            x, ha = ts(cap_x), cap_ha
        axT.text(
            x,
            cap_y,
            caption,
            ha=ha,
            va="center",
            fontsize=8.0 if cap_x is None else 7.6,
            fontweight="bold",
            color="white" if cap_x is None else (RED if colour == RED else YELLOW_EDGE),
            zorder=4,
        )

    # ------------------------------------------------------ track 2: model windows --
    for a, b, colour, caption, anchor, ha in WINDOWS:
        axT.plot(
            [ts(a), ts(a), ts(b) + 1, ts(b) + 1],
            [BR_Y + BR_TICK, BR_Y, BR_Y, BR_Y + BR_TICK],
            color=colour,
            lw=2.0 if colour == BLUE else 1.4,
            solid_capstyle="round",
            zorder=4,
        )
        axT.text(
            ts(anchor),
            BR_Y - 0.09,
            caption,
            ha=ha,
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color=colour,
            zorder=4,
        )

    # -------------------------------------------------- track 3: daily event density --
    axD.set_ylim(-0.06 * DENS_MAX, DENS_MAX)
    axD.set_yticks(np.arange(0, DENS_MAX + 1, 200_000))
    axD.set_yticklabels(["0", "200k", "400k", "600k", "800k", "1M"])
    axD.tick_params(axis="y", labelsize=7.5)
    axD.grid(axis="y", color=GRID, lw=0.5)
    axD.set_axisbelow(True)
    for side in ("top", "right"):
        axD.spines[side].set_visible(False)

    days = daily.index
    axD.fill_between(days, 0, daily["pr_issue"], color=SLATE, alpha=0.18, lw=0, zorder=2)
    axD.plot(days, daily["pr_issue"], color=SLATE, lw=1.2, zorder=3)

    axP = axD.twinx()
    axP.set_ylim(-0.06 * PUSH_MAX, PUSH_MAX)
    axP.set_yticks(np.arange(0, PUSH_MAX + 1, 1_000_000))
    axP.set_yticklabels(["0", "1M", "2M", "3M", "4M"])
    axP.tick_params(axis="y", labelsize=7.5, colors=RED)
    for side in ("top", "left"):
        axP.spines[side].set_visible(False)
    axP.spines["right"].set_color("#e0a0a3")
    axP.plot(days, daily["push"], color=RED, lw=1.2, ls=(0, (4, 2)), alpha=0.95, zorder=4)

    # series are named by their axes, colour-matched to the curves — no inline legend
    axD.set_ylabel("PR + Issue events / day", fontsize=8, color=SLATE, labelpad=4)
    axP.set_ylabel("PushEvent / day", fontsize=8, color=RED, labelpad=4)

    # ---------------------------------------------------------- shared date axis ----
    axB.set_ylim(0, 1)
    axB.set_yticks([])
    for side in ("top", "left", "right"):
        axB.spines[side].set_visible(False)
    ticks = pd.date_range("2025-12-01", "2026-12-01", freq="MS")
    axB.set_xticks([ts(t) for t in ticks])
    axB.set_xticklabels([t.strftime("%b") for t in ticks], fontsize=7.6, color=MUTED)
    axB.tick_params(length=3.5, pad=2)
    for date, year in (("2025-12-15", "2025"), ("2026-06-01", "2026")):
        axB.text(
            ts(date),
            0.02,
            year,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
            fontweight="bold",
        )

    # ----------------------------------------------------------- track name gutter --
    for ax, frac, name in (
        (axT, 0.795, "1  SANITY"),
        (axT, 0.36, "2  WINDOWS"),
        (axD, 0.5, "3  DENSITY"),
    ):
        pos = ax.get_position()
        fig.text(
            0.021,
            pos.y0 + frac * pos.height,
            name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color=MUTED,
        )

    # ------------------------------------------------------------------ title block --
    fig.text(
        L,
        0.975,
        "GH Archive 343-Day Corpus Audit: Data Sanity & Validation Window Selection",
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        L,
        0.930,
        "December 1, 2025 – November 20, 2026  ·  8,232 hourly files analyzed",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    # ------------------------------------------------------------- provenance line --
    # the date axis above is the last element in the figure: no callout boxes below it
    fig.text(
        L,
        0.030,
        "audit: scripts/verify_raw_data.py  ·  window rules: docs/VALIDATION_DATA.md",
        fontsize=6.8,
        color=MUTED,
        va="bottom",
    )
    return fig


def main() -> None:
    build().savefig(OUT, dpi=DPI, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
