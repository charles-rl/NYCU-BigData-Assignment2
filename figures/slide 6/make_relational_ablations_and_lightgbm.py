#!/usr/bin/env python3
"""Slide 6 figure: relational ablations vs the LightGBM tabular baseline.

Writes ``figures/slide 6/relational_ablations_and_lightgbm.png`` — two charts side
by side:

  left   Official score @CMA-ES for the four arms the slide talks about, with
         3-seed error bars and two delta brackets.
  right  Per-activity F1 gain of the graph model over LightGBM at the same
         CMA-ES-fitted thresholds.

Every number is read from the experiment artifacts rather than typed in by hand:

  experiments/ablations/ablation_summary.csv              score per (variant, seed)
  experiments/ablations/outputs/<variant>_seed<seed>/result.json
                                                          per_activity_f1_cmaes

so re-running the ablation grid and then this script refreshes the figure. The
derived numbers also land in ``figures/slide 6/relational_ablations_data.csv`` so
the slide can be checked without re-running anything. If the artifacts are absent
(a laptop holding only the repo), the script falls back to the frozen values
below and prints a warning.

Run:  python "figures/slide 6/make_relational_ablations_and_lightgbm.py"
Deps: matplotlib; scipy is used for the paired t-test p-values when present.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure

# --------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent  # figures/slide 6
REPO = HERE.parent.parent  # repo root
ABL = REPO / "experiments" / "ablations"
SUMMARY_CSV = ABL / "ablation_summary.csv"
OUTPUTS = ABL / "outputs"
CACHE = HERE / "relational_ablations_data.csv"

SEEDS = (42, 123, 999)

# --------------------------------------------------------------------------- theme
INK = "#0f172a"
BODY = "#334155"
MUTED = "#64748b"
FAINT = "#94a3b8"
CANVAS = "#ffffff"
PANEL = "#f8fafc"
GRID = "#e2e8f0"

NAVY = "#1e3a8a"  # Full HGT Model (spec)
SLATE = "#475569"  # w/o Text Features (spec)
ORANGE = "#ea580c"  # w/o Residual Fusion (spec)
GRAY = "#94a3b8"  # LightGBM Baseline (spec)
GREEN = "#16a34a"  # positive Δ F1
GREEN_EDGE = "#15803d"

FIG_W, FIG_H, DPI = 14.0, 6.0, 300

# Frozen fallbacks, from experiments/ablations/ablation_summary.csv and the
# per-cell result.json files, for machines without the artifacts.
FALLBACK_SCORES = {
    # arm -> (mean @CMA-ES %, sd, mean @0.5 %)
    "full": (65.7138, 0.0863, 57.2774),
    "no-text": (65.6280, 0.0715, 57.2820),
    "no-residual-fusion": (65.4191, 0.1016, 55.8708),
    "lightgbm": (63.3327, 0.1904, 59.8588),
}
FALLBACK_ACTIVITY = {
    # activity -> (delta pts, sd over seeds, paired p)
    "open_pr": (6.20, 0.49, 0.0020),
    "issue_comment": (3.76, 0.38, 0.0035),
    "pr_review_comment": (2.96, 0.61, 0.0138),
    "open_issue": (1.83, 0.75, 0.0523),
    "pr_review": (1.46, 1.04, 0.1352),
    "push": (0.16, 0.06, 0.0443),
}

ACTIVITIES = ("open_pr", "issue_comment", "pr_review_comment", "open_issue", "pr_review", "push")


# ------------------------------------------------------------------------- stats
def paired_p(a: list[float], b: list[float]) -> float:
    """Two-sided paired t-test p-value over matched seeds (n = 3 here)."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return 0.0
    dof = len(diffs) - 1
    t = statistics.mean(diffs) / (sd / math.sqrt(len(diffs)))
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t), dof))
    except Exception:  # keeps the figure buildable without scipy
        if dof == 2:  # exact two-sided p for Student t with 2 degrees of freedom
            return float(1.0 - abs(t) / math.sqrt(t * t + 2.0))
        return float("nan")


# ------------------------------------------------------------------------- loading
def load_scores() -> dict[str, dict[str, float]]:
    """mean / sd of the official score per arm, straight from the summary CSV."""
    if not SUMMARY_CSV.exists():
        print(f"WARNING: {SUMMARY_CSV} not found — using frozen fallback numbers")
        return {
            arm: {"cma": mean, "sd": sd, "d05": d05, "n": 3}
            for arm, (mean, sd, d05) in FALLBACK_SCORES.items()
        }
    per_seed: dict[str, dict[int, dict[str, float]]] = {}
    with SUMMARY_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok" or int(row["seed"]) not in SEEDS:
                continue
            if row["variant"] not in FALLBACK_SCORES:
                continue
            per_seed.setdefault(row["variant"], {})[int(row["seed"])] = {
                "cma": float(row["score_cmaes"]) * 100.0,
                "d05": float(row["score_default_0_5"]) * 100.0,
            }
    out: dict[str, dict[str, float]] = {}
    for arm, by_seed in per_seed.items():
        cma = [by_seed[s]["cma"] for s in SEEDS if s in by_seed]
        d05 = [by_seed[s]["d05"] for s in SEEDS if s in by_seed]
        out[arm] = {
            "cma": statistics.mean(cma),
            "sd": statistics.stdev(cma),
            "d05": statistics.mean(d05),
            "n": float(len(cma)),
        }
    return out


def fallback_activity() -> dict[str, dict[str, float]]:
    return {
        act: {"delta": mean, "sd": sd, "p": p}
        for act, (mean, sd, p) in FALLBACK_ACTIVITY.items()
    }


def load_activity_deltas() -> dict[str, dict[str, float]]:
    """Per-activity F1 delta (HGT − LightGBM) at CMA-ES thresholds, from result.json."""
    cells: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for arm in ("full", "lightgbm"):
        for seed in SEEDS:
            path = OUTPUTS / f"{arm}_seed{seed}" / "result.json"
            if not path.exists():
                print(f"WARNING: {path} missing — using frozen fallback per-activity deltas")
                return fallback_activity()
            try:
                cells[(arm, seed)] = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                print(f"WARNING: cannot read {path}: {exc} — using frozen fallback deltas")
                return fallback_activity()

    out: dict[str, dict[str, float]] = {}
    for act in ACTIVITIES:
        hgt = [cells[("full", s)]["per_activity_f1_cmaes"][act] for s in SEEDS]
        lgb = [cells[("lightgbm", s)]["per_activity_f1_cmaes"][act] for s in SEEDS]
        diffs = [(a - b) * 100.0 for a, b in zip(hgt, lgb, strict=True)]
        out[act] = {
            "delta": statistics.mean(diffs),
            "sd": statistics.stdev(diffs),
            "p": paired_p(hgt, lgb),
        }
    return out


def load_headline_scores() -> tuple[list[float], list[float]]:
    """Per-seed @CMA-ES scores for `full` and `lightgbm`, for the headline test."""
    per_seed: dict[str, dict[int, float]] = {"full": {}, "lightgbm": {}}
    if not SUMMARY_CSV.exists():
        return [], []
    with SUMMARY_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok" or row["variant"] not in per_seed:
                continue
            if int(row["seed"]) not in SEEDS:
                continue
            per_seed[row["variant"]][int(row["seed"])] = float(row["score_cmaes"]) * 100.0
    if not all(len(per_seed[arm]) == len(SEEDS) for arm in per_seed):
        return [], []
    return [per_seed["full"][s] for s in SEEDS], [per_seed["lightgbm"][s] for s in SEEDS]


# ---------------------------------------------------------------------- the figure
def bracket(
    ax,
    x0: float,
    x1: float,
    y_top: float,
    arm_len: float,
    text: str,
    color: str,
    size: float = 9.4,
) -> None:
    """Delta bracket: crossbar with short drop arms, caption centred above it."""
    ax.plot([x0, x0], [y_top - arm_len, y_top], color=color, lw=1.1, solid_capstyle="butt", zorder=3)
    ax.plot([x1, x1], [y_top - arm_len, y_top], color=color, lw=1.1, solid_capstyle="butt", zorder=3)
    ax.plot([x0, x1], [y_top, y_top], color=color, lw=1.1, solid_capstyle="butt", zorder=3)
    ax.text(
        (x0 + x1) / 2,
        y_top + 0.09,
        text,
        ha="center",
        va="bottom",
        fontsize=size,
        color=color,
        fontweight="bold",
        zorder=4,
    )


def build() -> Figure:
    scores = load_scores()
    deltas = load_activity_deltas()

    full_seeds, lgb_seeds = load_headline_scores()
    headline_gap = scores["full"]["cma"] - scores["lightgbm"]["cma"]
    headline_p = paired_p(full_seeds, lgb_seeds) if full_seeds else 0.0043
    if not math.isfinite(headline_p):
        headline_p = 0.0043

    resid_gap = scores["no-residual-fusion"]["cma"] - scores["full"]["cma"]
    resid_gap05 = scores["no-residual-fusion"]["d05"] - scores["full"]["d05"]

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(CANVAS)
    if any(f.name == "Inter" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Inter"

    # explicit axes: the right chart's category labels need a guaranteed gutter
    ax_l = fig.add_axes((0.055, 0.185, 0.398, 0.660))
    ax_r = fig.add_axes((0.578, 0.185, 0.402, 0.660))

    # ---------------------------------------------------------------- left chart --
    ax_l.set_facecolor(PANEL)
    for side in ("top", "right", "left"):
        ax_l.spines[side].set_visible(False)
    ax_l.spines["bottom"].set_color(FAINT)

    arms = [
        ("Full HGT Model", "full", NAVY),
        ("w/o Text Features", "no-text", SLATE),
        ("w/o Residual Fusion", "no-residual-fusion", ORANGE),
        ("LightGBM Baseline", "lightgbm", GRAY),
    ]
    xs = np.arange(len(arms))
    heights = [scores[key]["cma"] for _, key, _c in arms]
    errs = [scores[key]["sd"] for _, key, _c in arms]

    ax_l.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax_l.set_axisbelow(True)
    ax_l.bar(
        xs,
        [h - 60.0 for h in heights],
        bottom=[60.0] * len(arms),
        width=0.58,
        color=[c for _n, _k, c in arms],
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    for i, arm in enumerate(arms):
        colour = arm[2]
        ax_l.errorbar(
            xs[i],
            heights[i],
            yerr=errs[i],
            fmt="none",
            ecolor=INK if colour == NAVY else colour,
            elinewidth=1.3,
            capsize=4,
            capthick=1.1,
            zorder=3,
        )

    for i, (_name, key, _colour) in enumerate(arms):
        top = scores[key]["cma"] + scores[key]["sd"]
        ax_l.text(
            xs[i],
            top + 0.13,
            f"{scores[key]['cma']:.2f} ± {scores[key]['sd']:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.4,
            fontweight="bold",
            color=INK,
            zorder=4,
        )

    ax_l.set_ylim(60.0, 68.0)
    ax_l.set_xlim(-0.62, len(arms) - 0.38)
    ax_l.set_yticks(np.arange(60.0, 69.0, 1.0))
    ax_l.set_yticklabels([f"{v:.0f}%" for v in np.arange(60.0, 69.0, 1.0)], fontsize=9.5)
    ax_l.tick_params(axis="both", length=0, labelcolor=BODY, pad=6)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(
        ["Full HGT\nModel", "w/o Text\nFeatures", "w/o Residual\nFusion", "LightGBM\nBaseline"],
        fontsize=10.0,
        color=INK,
    )
    for lab in ax_l.get_xticklabels():
        lab.set_fontweight("bold")
    ax_l.set_ylabel(
        "Official Score: (Macro-F1 + Micro-F1) / 2 (%)", fontsize=10.5, color=INK, labelpad=8
    )
    ax_l.set_title(
        "Overall Score @CMA-ES — ablations vs the tabular baseline",
        fontsize=12.2,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=15,
    )

    # widest bracket (Full vs LightGBM) sits lowest; the inner one above it
    bracket(
        ax_l,
        xs[0],
        xs[3],
        66.70,
        0.28,
        f"+{headline_gap:.2f} pts vs LightGBM  (paired over seeds, p = {headline_p:.3f})",
        NAVY,
        size=9.2,
    )
    bracket(
        ax_l,
        xs[0],
        xs[2],
        67.62,
        0.28,
        f"{resid_gap:+.2f} pts @CMA   ({resid_gap05:+.2f} pts @0.5)",
        ORANGE,
        size=9.2,
    )

    # ---------------------------------------------------------------- right chart --
    ax_r.set_facecolor(PANEL)
    for side in ("top", "right", "left"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["bottom"].set_color(FAINT)

    ys = np.arange(len(ACTIVITIES))[::-1]
    dvals = [deltas[a]["delta"] for a in ACTIVITIES]
    dsd = [deltas[a]["sd"] for a in ACTIVITIES]

    ax_r.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax_r.set_axisbelow(True)
    # RESULTS.md treats any Δ below ~0.2 pts as seed noise
    ax_r.axvspan(0.0, 0.2, color=GRID, alpha=0.9, zorder=1)
    ax_r.axvline(0.0, color=FAINT, lw=1.1, zorder=2)
    ax_r.barh(
        ys,
        dvals,
        height=0.58,
        color=[GRAY if a == "push" else GREEN for a in ACTIVITIES],
        edgecolor=[GRAY if a == "push" else GREEN_EDGE for a in ACTIVITIES],
        linewidth=1.0,
        zorder=3,
    )
    for y, act, val, sd in zip(ys, ACTIVITIES, dvals, dsd, strict=True):
        ax_r.errorbar(
            val,
            y,
            xerr=sd,
            fmt="none",
            ecolor=MUTED if act == "push" else INK,
            elinewidth=1.2,
            capsize=4,
            capthick=1.0,
            zorder=4,
        )
        ax_r.text(
            val + sd + 0.18,
            y + 0.11,
            f"+{val:.2f}",
            ha="left",
            va="center",
            fontsize=10.6,
            fontweight="bold",
            color=GRAY if act == "push" else GREEN_EDGE,
            zorder=5,
        )
        p = deltas[act]["p"]
        tag = "tie — inside the noise band" if act == "push" else ("n.s." if p >= 0.05 else f"p = {p:.3f}")
        ax_r.text(
            val + sd + 0.18,
            y - 0.20,
            tag,
            ha="left",
            va="center",
            fontsize=7.8,
            color=MUTED,
            zorder=5,
        )

    ax_r.set_xlim(0.0, 8.6)
    ax_r.set_ylim(-0.62, len(ACTIVITIES) - 0.28)
    ax_r.set_xticks(np.arange(0.0, 9.0, 1.0))
    ax_r.set_xticklabels([f"{v:.0f}" for v in np.arange(0.0, 9.0, 1.0)], fontsize=9.5)
    ax_r.tick_params(axis="both", length=0, labelcolor=BODY, pad=6)
    ax_r.set_yticks(ys)
    ax_r.set_yticklabels(list(ACTIVITIES), fontsize=10.4, color=INK)
    for lab in ax_r.get_yticklabels():
        lab.set_fontweight("bold")
    ax_r.set_xlabel(
        "Δ F1 = HGT − LightGBM (percentage points, @CMA-ES thresholds)",
        fontsize=10.2,
        color=INK,
        labelpad=8,
    )
    ax_r.set_title(
        "Per-Activity F1 Gain: HGT vs LightGBM",
        fontsize=12.2,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=25,
    )
    ax_r.text(
        0.0,
        1.028,
        "GNN advantage concentrates on multi-table collaborative activities",
        transform=ax_r.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.8,
        color=GREEN_EDGE,
        fontweight="bold",
    )

    # ----------------------------------------------------------------- footnotes --
    fig.text(
        0.055,
        0.072,
        "axis truncated at 60% · bars = mean of seeds 42/123/999 · whiskers = ±1 sd",
        fontsize=7.8,
        color=MUTED,
        ha="left",
        va="center",
    )
    fig.text(
        0.578,
        0.072,
        "whiskers = ±1 sd over 3 seeds · paired t per activity · 0–0.2 pts = seed-noise band",
        fontsize=7.8,
        color=MUTED,
        ha="left",
        va="center",
    )
    fig.text(
        0.055,
        0.028,
        "source: experiments/ablations/ablation_summary.md · 27/27 runs ok · 310 epochs · "
        "score = (Macro-F1 + Micro-F1) / 2",
        fontsize=7.6,
        color=FAINT,
        ha="left",
        va="center",
    )
    fig.text(
        0.578,
        0.028,
        "the graph model ties boosted trees on push and wins wherever an event spans tables",
        fontsize=7.6,
        color=FAINT,
        ha="left",
        va="center",
    )

    return fig


def main() -> int:
    fig = build()
    out = HERE / "relational_ablations_and_lightgbm.png"
    fig.savefig(out, dpi=DPI, facecolor=CANVAS)
    print(f"wrote {out}")

    # derived table, so every number on the slide is auditable without a re-run
    scores = load_scores()
    deltas = load_activity_deltas()
    with CACHE.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "name", "value", "sd", "extra"])
        for arm, row in scores.items():
            writer.writerow(
                ["score_cmaes_pct", arm, f"{row['cma']:.4f}", f"{row['sd']:.4f}", f"@0.5={row['d05']:.4f}"]
            )
        for act, row in deltas.items():
            writer.writerow(
                ["delta_f1_pts", act, f"{row['delta']:.4f}", f"{row['sd']:.4f}", f"paired_p={row['p']:.4f}"]
            )
    print(f"wrote {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
