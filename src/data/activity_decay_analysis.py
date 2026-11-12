"""Empirical P(activity in the next 7 days | elapsed days since the last event).

Purpose: calibrate the two numbers a temporal model needs before training.

  1. **Temporal edge weights.**  For every target activity we estimate the
     empirical hazard curve
         p_act(dt) = P(user does `act` in (t, t+7] | days since their last event = dt)
     for dt in [0, 30], pooled over many cutoffs ("anchors") spread across the
     data window.  The curve is then summarised as an exponential decay
     constant (tau, half-life) plus a weekly periodicity score, so an edge
     weight can be set to exp(-dt / tau) per activity type instead of one
     shared time encoding.

  2. **Loss positive weights.**  Per-activity prevalence and positive counts
     (both pooled over anchors and in the single official validation window),
     from which pos_weight = n_negative / n_positive for BCEWithLogitsLoss.

Two axes are reported, because "their last event" is ambiguous and both matter:

  * ``any``  - dt since the user's last event of **any** of the 6 types
               (the literal reading: how long has the user been quiet).
  * ``same`` - dt since the user's last event of **that same** type
               (how overdue is this particular activity; this is the axis that
               shows weekly periodicity for pushes and reviews).

Definitions follow the pipeline in `src/data/temporal_split.py`:
  * feature side: a row counts only when it is *visible* at the anchor, i.e.
    row time < anchor (`first_seen_at` for PullRequests/Issues/PRReviews);
  * label side: an event counts when its *event* time is in [anchor, anchor+7)
    (`created_at` / `submitted_at`), with `Issues.is_pull_request` rows
    excluded from `open_issue`;
  * users: cohort users only (`Users.is_cohort`), and only those first seen
    before the anchor (a user that does not exist yet is not a negative).

Coverage limit worth knowing: the tables span 2026-10-01 .. 2026-10-31, so a
7-day-ahead label needs anchor <= 10-24, which leaves at most 23 days of
lookback.  dt = 24..30 therefore has **no support** in this dataset; the plots
mark those bins as empty instead of inventing data.  Per-bin support `n` is
written to the CSV so the model config can trust only well-populated bins.

Run from the repo root:
    python -m src.data.activity_decay_analysis
    python -m src.data.activity_decay_analysis --max-delta 30 --horizon-days 7
    python -m src.data.activity_decay_analysis --no-plots

Writes (gitignored, default `--out temporal_output`):
    decay_curves.csv            tidy per (axis, activity, dt) curve + support
    activity_summary.csv        one row per activity: rarity, weights, fit
    activity_summary.json       same, structured, ready to paste into config
    summary.md                  human-readable tables for the report
    01_decay_any.png            6 panels: p(dt) with CI, support, fit
    02_decay_same.png           6 panels, same-type recency axis
    03_overlay.png              all activities overlaid (log y), both axes
    04_dt_distribution.png      dt histogram + P(activity | dt) heatmap
    05_class_rarity.png         prevalence, pos_weight, positives per window
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.data.split_io import DEFAULT_DB, load_tables, parse_cutoff
from src.data.temporal_split import (
    ACTIVITY_LABELS,
    LABEL_COLS,
    ROW_TIME,
    extract_labels,
)

SECONDS_PER_DAY = 86_400.0
AXES = ("any", "same")
AXIS_DESC = {
    "any": "days since the user's last event (any of the 6 types)",
    "same": "days since the user's last event of the same type",
}
NEVER = "never"
BEYOND = ">max"

# One colour per activity, kept stable across every figure.
COLORS = {
    "open_pr": "#1f77b4",
    "open_issue": "#d62728",
    "issue_comment": "#2ca02c",
    "pr_review": "#9467bd",
    "pr_review_comment": "#8c564b",
    "push": "#ff7f0e",
}


# ------------------------------------------------------------------ inputs --


def cohort_users(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cohort users only: non-cohort users have incomplete activity."""
    users = tables["Users"]
    if "is_cohort" not in users.columns:
        return users
    keep = users["is_cohort"].astype("boolean").fillna(False).astype(bool)
    return cast(pd.DataFrame, users[keep])


def activity_streams(
    tables: dict[str, pd.DataFrame], user_ids: np.ndarray
) -> dict[str, pd.DataFrame]:
    """Per activity, the (u, rt) rows a model could see: u = user_id, rt = row time.

    `any` is the union of all six streams, i.e. the user's event history.
    """
    streams: dict[str, pd.DataFrame] = {}
    for label, table, user_col, _event_col, row_filter in ACTIVITY_LABELS:
        df = tables[table]
        row_col = ROW_TIME[table]
        keep = df[user_col].notna().to_numpy()
        if row_filter == "not_pull_request" and "is_pull_request" in df.columns:
            is_pr = (
                df["is_pull_request"]
                .astype("boolean")
                .fillna(False)
                .astype(bool)
                .to_numpy()
            )
            keep &= ~is_pr
        u = df.loc[keep, user_col].astype("int64").to_numpy()
        rt = pd.to_datetime(df.loc[keep, row_col]).to_numpy()
        inside = np.isin(u, user_ids)
        streams[label] = pd.DataFrame(
            {"u": u[inside], "rt": pd.to_datetime(rt[inside])}
        )
    streams["any"] = pd.concat(
        [streams[label] for label, *_ in ACTIVITY_LABELS], ignore_index=True
    )
    return streams


def days_since_last(
    stream: pd.DataFrame, anchor: pd.Timestamp, user_ids: np.ndarray
) -> np.ndarray:
    """dt in days (float) since the user's last visible event before `anchor`.

    NaN where the user has no visible event: censored, not "dt = 0".
    """
    visible = stream.loc[stream["rt"] < anchor]
    if visible.empty:
        return np.full(user_ids.shape, np.nan)
    last = visible.groupby("u", sort=False)["rt"].max().reindex(user_ids)
    delta = (anchor - last).dt.total_seconds() / SECONDS_PER_DAY
    return delta.to_numpy(dtype=float)


def labels_at(
    tables: dict[str, pd.DataFrame],
    anchor: pd.Timestamp,
    horizon_days: int,
    users: pd.DataFrame,
) -> np.ndarray:
    """(n_users, 6) int labels for events with event time in [anchor, anchor+h)."""
    end = cast(pd.Timestamp, anchor + pd.Timedelta(days=horizon_days))
    lab = extract_labels(tables, anchor, end, users)
    return lab[LABEL_COLS].to_numpy(dtype=np.int64)


# ------------------------------------------------------------- statistics --


def bin_index(dt: np.ndarray, max_delta: int) -> np.ndarray:
    """Map dt to a bin: 0..max_delta, then the `never` and `>max` buckets."""
    idx = np.full(dt.shape, max_delta + 1, dtype=np.int64)  # censored: never seen
    seen = np.isfinite(dt)
    inside = seen & (dt <= max_delta)
    beyond = seen & (dt > max_delta)
    idx[inside] = np.clip(dt[inside], 0, None).astype(np.int64)
    idx[beyond] = max_delta + 2
    return idx


def bin_labels(max_delta: int) -> list[str]:
    """Bucket names, in bin-index order."""
    return [str(d) for d in range(max_delta + 1)] + [NEVER, BEYOND]


def wilson(
    pos: np.ndarray, n: np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion; NaN where n == 0."""
    p = np.full(pos.shape, np.nan, dtype=float)
    ok = n > 0
    p[ok] = pos[ok] / n[ok]
    pbar = p.copy()
    denom = 1.0 + z**2 / np.where(ok, n, 1)
    centre = (pbar + z**2 / (2 * np.where(ok, n, 1))) / denom
    half = (
        z
        * np.sqrt(
            pbar * (1 - pbar) / np.where(ok, n, 1)
            + z**2 / (4 * np.where(ok, n, 1) ** 2)
        )
        / denom
    )
    lo = np.clip(centre - half, 0, 1)
    hi = np.clip(centre + half, 0, 1)
    lo[~ok] = np.nan
    hi[~ok] = np.nan
    return lo, hi


def fit_decay(
    dt: np.ndarray, p: np.ndarray, n: np.ndarray, min_support: int
) -> dict[str, float]:
    """Fit p(dt) = floor + amp * exp(-dt / tau) on supported bins.

    Returns tau / half-life / floor / amp / r2, plus the raw decay ratio
    p(1)/p(7) and a log-linear slope as a robust fallback description.
    """
    out: dict[str, float] = {
        "tau_days": float("nan"),
        "half_life_days": float("nan"),
        "floor": float("nan"),
        "amp": float("nan"),
        "r2": float("nan"),
        "p_dt1": float("nan"),
        "p_dt7": float("nan"),
        "decay_ratio_1_7": float("nan"),
        "loglinear_slope_per_day": float("nan"),
    }
    ok = np.isfinite(p) & (n >= min_support)
    if ok.sum() < 3:
        return out

    x, y, w = dt[ok], p[ok], n[ok]
    out["p_dt1"] = float(y[x == 1][0]) if (x == 1).any() else float("nan")
    out["p_dt7"] = float(y[x == 7][0]) if (x == 7).any() else float("nan")
    if out["p_dt7"] and out["p_dt7"] > 0:
        out["decay_ratio_1_7"] = out["p_dt1"] / out["p_dt7"]

    positive = y > 0
    if positive.sum() >= 3:
        slope = np.polyfit(x[positive], np.log(y[positive]), 1, w=np.sqrt(w[positive]))[
            0
        ]
        out["loglinear_slope_per_day"] = float(slope)
        if slope < 0:
            out["tau_days"] = float(-1.0 / slope)
            out["half_life_days"] = float(np.log(2) * -1.0 / slope)

    try:
        from scipy.optimize import curve_fit

        def model(t: np.ndarray, floor: float, amp: float, tau: float) -> np.ndarray:
            return floor + amp * np.exp(-t / tau)

        p0 = [max(float(y.min()), 1e-4), max(float(y.max() - y.min()), 1e-3), 5.0]
        popt, _ = curve_fit(
            model,
            x,
            y,
            p0=p0,
            sigma=1.0 / np.sqrt(w),
            absolute_sigma=False,
            bounds=([0.0, 0.0, 0.4], [1.0, 1.0, 500.0]),
            maxfev=20_000,
        )
        floor, amp, tau = (float(v) for v in popt)
        pred = model(x, floor, amp, tau)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out.update(
            tau_days=tau,
            half_life_days=float(np.log(2) * tau),
            floor=floor,
            amp=amp,
            r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        )
    except (ValueError, RuntimeError):  # optional polish; the log fit gave tau
        return out
    return out


PERIOD_POWER_GATE = 0.30
# A least-squares sinusoid scan replaces the periodogram as the cycle detector.
# Rationale: on a 25-point curve the FFT spreads a single sine over several
# bins, so a peak's share of in-band power fails to separate signal from null --
# a real 50%-amplitude weekly cycle scores 0.25 while a noiseless pure
# exponential scores 0.26 through detrend leakage. Scanning candidate periods
# and scoring the variance a sinusoid explains does separate them.
PERIOD_R2_GATE = 0.40
PERIOD_AMP_GATE = 0.10
PERIOD_MIN_SUPPORT = 500


def periodicity(
    dt: np.ndarray,
    p: np.ndarray,
    n: np.ndarray,
    min_support: int,
    max_delta: int,
    model: dict[str, float] | None = None,
) -> dict[str, float]:
    """Weekly-bump ratios plus the dominant period of the detrended curve.

    bump(k) = p(k) / mean(p(k-3..k-2, k+2..k+3)) - 1 is the excess over the
    local trend at a 7-day multiple.

    The residual is taken against the *fitted decay* when `model` is supplied,
    not against a quadratic. A quadratic cannot absorb a discrete exponential:
    detrending `floor + amp * exp(-dt/tau)` with one leaves a sawtooth whose
    leakage lands at the band edge and reports an 11.3-day "cycle". A pure
    synthetic exponential with no noise scores 11.27 d on that detrend, which is
    the tell -- the number is a property of the detrend, not of the data.
    Subtracting the model makes an exponential score zero by construction.

    The cycle itself is found by scanning candidate periods and scoring the
    variance a sinusoid explains at each, rather than by a periodogram. A
    25-point FFT splits one sine across several bins, so peak *share* does not
    separate signal from null (measured: a true 50%-amplitude weekly cycle
    scores 0.25, a noiseless pure exponential 0.26).
    """
    out = {
        "bump_7": float("nan"),
        "bump_14": float("nan"),
        "bump_21": float("nan"),
        "plateau_ratio": float("nan"),
        "dominant_period_days": float("nan"),
        "cycle_r2": float("nan"),
        "cycle_amplitude_share": float("nan"),
    }
    # A bump read off a 100-row bin is noise, so bump/plateau need real support.
    bump_support = max(min_support, 150)
    ok = np.isfinite(p) & (n >= bump_support)
    if ok.sum() < 5:
        return out
    x, y = dt[ok], p[ok]

    for k in (7, 14, 21):
        at = y[x == k]
        near = y[(np.abs(x - k) >= 2) & (np.abs(x - k) <= 3)]
        if at.size and near.size and near.mean() > 0:
            out[f"bump_{k}"] = float(at[0] / near.mean() - 1.0)

    # A user on a 7-day cadence has dt uniform in [0, 7): the signature is a
    # plateau then a drop, not a spike at dt = 7.  So also report how much of
    # the near-term rate survives into week 2.
    head = y[(x >= 1) & (x <= 6)]
    tail = y[(x >= 8) & (x <= 14)]
    if head.size and tail.size and head.mean() > 0:
        out["plateau_ratio"] = float(tail.mean() / head.mean())

    grid = np.arange(0, max_delta + 1)
    vals = pd.Series(np.nan, index=grid, dtype=float)
    vals.loc[x.astype(int)] = y
    vals = vals.interpolate(limit_direction="both")
    if vals.isna().any() or len(grid) < 8:
        return out
    fit_vals = vals.to_numpy()
    base: np.ndarray | None = None
    if model is not None:
        tau = float(model.get("tau_days", float("nan")))
        floor = float(model.get("floor", float("nan")))
        amp = float(model.get("amp", float("nan")))
        if np.isfinite(tau) and np.isfinite(floor) and np.isfinite(amp) and tau > 0:
            base = floor + amp * np.exp(-grid / tau)
    if base is None:
        logv = np.log(np.clip(fit_vals, 1e-6, None))
        base = np.exp(np.polyval(np.polyfit(grid, logv, 2), grid))
    resid = fit_vals - base
    tss = float(np.sum((resid - resid.mean()) ** 2))
    # Below ~500 rows per bin the scan's false-positive rate climbs to 0.7%; a
    # cycle claim off support that thin is not worth making.
    if tss <= 0 or float(np.median(n[ok])) < PERIOD_MIN_SUPPORT:
        return out
    best_r2, best_period, best_amp = -1.0, float("nan"), float("nan")
    level = float(np.mean(np.clip(base, 1e-6, None)))
    for period in np.arange(2.0, 12.001, 0.05):
        design = np.column_stack(
            [
                np.ones_like(grid, dtype=float),
                np.sin(2 * np.pi * grid / period),
                np.cos(2 * np.pi * grid / period),
            ]
        )
        coef, *_ = np.linalg.lstsq(design, resid, rcond=None)
        r2 = 1.0 - float(np.sum((resid - design @ coef) ** 2)) / tss
        if r2 > best_r2:
            best_r2 = r2
            best_period = float(period)
            best_amp = float(np.hypot(coef[1], coef[2]) / level)
    out["cycle_r2"] = best_r2
    out["cycle_amplitude_share"] = best_amp
    # Calibrated against the null (Poisson-sampled `floor + amp*exp(-dt/tau)`
    # curves, tau 1.5-12 d, floor 0.01-0.30, amp 0.05-1.0, n 300-5000): at
    # n >= 1500 the scan stays under r2 0.47 and amplitude 0.26, while a 30%
    # weekly sine on a tau=4 d decay scores r2 0.44-0.48 and is detected 20/20
    # times at n = 2000. Both gates are required -- r2 alone lets a low-n
    # wiggle through, amplitude alone lets a strong trend residual through.
    if best_r2 >= PERIOD_R2_GATE and best_amp >= PERIOD_AMP_GATE:
        out["dominant_period_days"] = best_period
    return out


def shape_label(stats: dict[str, float]) -> str:
    """Heuristic shape label used for the report table.

    Separation is decay speed (tau) plus how much of the curve survives as an
    asymptote (floor): a low floor is "the user goes quiet", a high floor is
    "some users keep going", and a genuine 7-day resonance would show up as a
    positive bump@7 instead.
    """
    tau = stats.get("tau_days", float("nan"))
    floor = stats.get("floor", float("nan"))
    bump = stats.get("bump_7", float("nan"))
    if np.isfinite(bump) and bump >= 0.15:
        return "weekly periodicity"
    if not np.isfinite(tau):
        return "no fit"
    if tau <= 3.5 and floor < 0.06:
        return "sharp exponential decay"
    if tau <= 7:
        return "decay to a floor"
    if tau <= 15:
        return "slow decay"
    return "flat / steady"


# ------------------------------------------------------------------ panel --


@dataclass
class Panel:
    """Accumulated (n, positives) counts per (axis, activity, dt bucket)."""

    max_delta: int
    n_act: int
    n: np.ndarray = field(init=False)
    pos: np.ndarray = field(init=False)
    windows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.nbins = self.max_delta + 3
        self.n = np.zeros((len(AXES), self.n_act, self.nbins), dtype=np.int64)
        self.pos = np.zeros((len(AXES), self.n_act, self.nbins), dtype=np.int64)

    def add(
        self,
        dt_any: np.ndarray,
        dt_same: np.ndarray,
        y: np.ndarray,
        anchor: pd.Timestamp,
        usable: np.ndarray | None = None,
    ) -> None:
        """Fold one anchor's user panel into the counts.

        `usable` is a length-`n_act` boolean mask: an activity whose stream was
        damaged around this anchor contributes no rows, so it cannot drag its own
        curve with a stretched or collapsed denominator.
        """
        keep = np.ones(self.n_act, dtype=bool) if usable is None else np.asarray(usable)
        for axis, dt_matrix in enumerate(
            (dt_any[:, None].repeat(self.n_act, 1), dt_same)
        ):
            for j in range(self.n_act):
                if not keep[j]:
                    continue
                idx = bin_index(dt_matrix[:, j], self.max_delta)
                self.n[axis, j] += np.bincount(idx, minlength=self.nbins)
                self.pos[axis, j] += np.bincount(
                    idx, weights=y[:, j].astype(float), minlength=self.nbins
                ).astype(np.int64)
        self.windows.append(
            {
                "anchor": anchor.isoformat(),
                "users": int(y.shape[0]),
                **{
                    act: (int(y[:, j].sum()) if keep[j] else np.nan)
                    for j, act in enumerate(LABEL_COLS)
                },
            }
        )


# ----------------------------------------------------------------- report --


def curve_table(panel: Panel) -> pd.DataFrame:
    """Tidy curve: one row per (axis, activity, bucket)."""
    labels = bin_labels(panel.max_delta)
    rows: list[dict[str, Any]] = []
    for a, axis in enumerate(AXES):
        for j, act in enumerate(LABEL_COLS):
            n = panel.n[a, j]
            pos = panel.pos[a, j]
            p = np.divide(pos, n, out=np.full(n.shape, np.nan), where=n > 0)
            lo, hi = wilson(pos, n)
            for b, bucket in enumerate(labels):
                dt = float(b) if bucket.isdigit() else np.nan
                rows.append(
                    {
                        "axis": axis,
                        "activity": act,
                        "bucket": bucket,
                        "dt_days": dt,
                        "n": int(n[b]),
                        "positives": int(pos[b]),
                        "p": float(p[b]),
                        "ci_lo": float(lo[b]),
                        "ci_hi": float(hi[b]),
                    }
                )
    return pd.DataFrame(rows)


def smoothing(curve: pd.DataFrame, alpha: float) -> pd.Series:
    """Laplace-smoothed p, so a 1-of-3 bin does not read as p = 0.33."""
    return (curve["positives"] + alpha) / (curve["n"] + 2 * alpha)


def curve_slice(
    curve: pd.DataFrame, axis: str, act: str, numeric: bool = False
) -> pd.DataFrame:
    """Rows of `curve` for one (axis, activity), optionally the dt bins only."""
    mask = (curve["axis"] == axis) & (curve["activity"] == act)
    if numeric:
        mask = mask & curve["dt_days"].notna()
    out = cast(pd.DataFrame, curve.loc[mask])
    if numeric:
        out = cast(pd.DataFrame, out.sort_values("dt_days"))
    return out


def summarise(
    panel: Panel, curve: pd.DataFrame, min_support: int, alpha: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-activity summary: rarity, pos_weight, decay fit, periodicity."""
    windows = pd.DataFrame(panel.windows)
    rows: list[dict[str, Any]] = []
    for act in LABEL_COLS:
        pooled = curve_slice(curve, "any", act)
        pos_total = int(pooled["positives"].sum())
        n_total = int(pooled["n"].sum())
        prevalence = pos_total / n_total if n_total else float("nan")

        dt_rows = curve_slice(curve, "any", act, numeric=True)
        fit = fit_decay(
            dt_rows["dt_days"].to_numpy(),
            dt_rows["p"].to_numpy(),
            dt_rows["n"].to_numpy(),
            min_support,
        )
        fit.update(
            periodicity(
                dt_rows["dt_days"].to_numpy(),
                dt_rows["p"].to_numpy(),
                dt_rows["n"].to_numpy(),
                min_support,
                panel.max_delta,
                fit,
            )
        )

        same_dt = curve_slice(curve, "same", act, numeric=True)
        fit_same = fit_decay(
            same_dt["dt_days"].to_numpy(),
            same_dt["p"].to_numpy(),
            same_dt["n"].to_numpy(),
            min_support,
        )
        fit_same.update(
            periodicity(
                same_dt["dt_days"].to_numpy(),
                same_dt["p"].to_numpy(),
                same_dt["n"].to_numpy(),
                min_support,
                panel.max_delta,
                fit_same,
            )
        )

        pos_by_window = (
            windows[act].astype(float).to_numpy() if act in windows else np.array([0.0])
        )
        # NaN = the activity's stream was damaged at that anchor, so it never
        # entered the curve; counting it as a zero-positive window would read as
        # "nobody did this" instead of "not observed".
        pos_by_window = pos_by_window[np.isfinite(pos_by_window)]
        if pos_by_window.size == 0:
            pos_by_window = np.array([0.0])
        rows.append(
            {
                "activity": act,
                "positives_pooled": pos_total,
                "user_anchors_pooled": n_total,
                "prevalence_pooled": prevalence,
                "pos_weight": (1.0 - prevalence) / prevalence
                if prevalence > 0
                else float("nan"),
                "neg_per_pos": (1.0 - prevalence) / prevalence
                if prevalence > 0
                else float("nan"),
                "positives_per_window_mean": float(pos_by_window.mean()),
                "positives_per_window_min": int(pos_by_window.min()),
                "positives_per_window_max": int(pos_by_window.max()),
                "share_windows_all_zero": float((pos_by_window == 0).mean()),
                "shape_any": shape_label(fit),
                "tau_days_any": fit["tau_days"],
                "half_life_days_any": fit["half_life_days"],
                "decay_ratio_1_7_any": fit["decay_ratio_1_7"],
                "floor_any": fit["floor"],
                "r2_any": fit["r2"],
                "bump_7_any": fit["bump_7"],
                "plateau_ratio_any": fit["plateau_ratio"],
                "period_any": fit["dominant_period_days"],
                "shape_same": shape_label(fit_same),
                "tau_days_same": fit_same["tau_days"],
                "half_life_days_same": fit_same["half_life_days"],
                "decay_ratio_1_7_same": fit_same["decay_ratio_1_7"],
                "r2_same": fit_same["r2"],
                "bump_7_same": fit_same["bump_7"],
                "plateau_ratio_same": fit_same["plateau_ratio"],
                "period_same": fit_same["dominant_period_days"],
                "p_given_never_any": curve_row_p(curve, "any", act, NEVER),
                "p_given_never_same": curve_row_p(curve, "same", act, NEVER),
                "_fit_any": fit,
                "_fit_same": fit_same,
            }
        )
    summary = pd.DataFrame(rows)
    fits = {
        act: {
            "any": summary.loc[summary["activity"] == act, "_fit_any"].iloc[0],
            "same": summary.loc[summary["activity"] == act, "_fit_same"].iloc[0],
        }
        for act in LABEL_COLS
    }
    summary = summary.drop(columns=["_fit_any", "_fit_same"])
    return summary, fits


def curve_row_p(curve: pd.DataFrame, axis: str, act: str, bucket: str) -> float:
    row = curve_slice(curve, axis, act)
    hit = row.loc[row["bucket"] == bucket, "p"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def event_stats(tables: dict[str, pd.DataFrame], user_ids: np.ndarray) -> pd.DataFrame:
    """Raw event counts per activity for the cohort (context for rarity)."""
    rows = []
    for label, table, user_col, event_col, row_filter in ACTIVITY_LABELS:
        df = tables[table]
        keep = df[user_col].notna().to_numpy()
        if row_filter == "not_pull_request" and "is_pull_request" in df.columns:
            is_pr = (
                df["is_pull_request"]
                .astype("boolean")
                .fillna(False)
                .astype(bool)
                .to_numpy()
            )
            keep &= ~is_pr
        u = df.loc[keep, user_col].astype("int64").to_numpy()
        inside = u[np.isin(u, user_ids)]
        per_user = pd.Series(inside).value_counts()
        rows.append(
            {
                "activity": label,
                "events": len(inside),
                "users_with_event": int(per_user.size),
                "users_with_event_share": per_user.size / max(len(user_ids), 1),
                "events_per_active_user_median": float(per_user.median())
                if per_user.size
                else 0.0,
                "events_per_active_user_p95": float(per_user.quantile(0.95))
                if per_user.size
                else 0.0,
                "event_time_col": event_col,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots --


def load_coverage(path: Path) -> pd.DataFrame:
    """Hour-level audit frame written by `scripts/verify_raw_data.py`."""
    frame = pd.read_csv(path, dtype={"day": str})
    frame = cast(pd.DataFrame, frame[frame["events"].notna()].copy())
    frame["day"] = frame["day"].astype(str)
    frame["events"] = frame["events"].astype(float)
    frame["PushEvent"] = frame["PushEvent"].astype(float)
    return frame


def load_activity_coverage(path: Path) -> pd.DataFrame:
    """Day x activity audit frame written by `scripts/verify_raw_data.py`.

    The hour frame cannot express a day-level density band: it is aggregated to
    `hours_ok == hours`, which is unreachable in a sparse regime (a typical day
    has 19/24 intact, so P(24/24) is under 1%). The day verdict -- "this day's
    total is within the local band" -- only exists in this frame, so it has to be
    read separately and merged in.
    """
    frame = pd.read_csv(path, dtype={"day": str})
    frame = cast(pd.DataFrame, frame[frame["activity"].notna()])
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(frame["day"].unique())))
    for activity in LABEL_COLS:
        block = cast(pd.DataFrame, frame[frame["activity"] == activity])
        if block.empty:
            continue
        index = pd.DatetimeIndex(pd.to_datetime(block["day"]))
        out[f"clean_{activity}"] = pd.Series(
            block["clean"].astype(bool).to_numpy(), index=index
        ).reindex(out.index, fill_value=False)
        if "usable" in block.columns:
            out[f"usable_{activity}"] = pd.Series(
                block["usable"].astype(bool).to_numpy(), index=index
            ).reindex(out.index, fill_value=False)
    out.index.name = "day"
    return out.sort_index()


def activity_columns(coverage: pd.DataFrame) -> list[str]:
    """Activities carrying their own `ok_<activity>` hour flags, if any."""
    return [act for act in LABEL_COLS if f"ok_{act}" in coverage.columns]


def clean_columns(per_day: pd.DataFrame) -> list[str]:
    """Activities with their own day-level flag (`usable_<act>` or `clean_<act>`).

    Separate from `activity_columns` because the two frames are at different
    granularities: the hour frame carries `ok_<act>`, the per-day frame carries
    the day verdicts. Reading the wrong one silently degrades to the
    all-activity verdict, which is the bug that made every cell look unusable.

    `usable` is preferred over `clean` when both exist. `clean` demands a
    flawless day, which a sparse regime cannot supply -- with 19/24 hours
    typically intact, P(24/24) is under 1% and a whole month reads as unusable.
    `usable` is the day-level band the curve actually needs.
    """
    return [
        act
        for act in LABEL_COLS
        if f"usable_{act}" in per_day.columns or f"clean_{act}" in per_day.columns
    ]


def day_flag_column(act: str, per_day: pd.DataFrame) -> str:
    """The column holding `act`'s day verdict, preferring the looser band."""
    return f"usable_{act}" if f"usable_{act}" in per_day.columns else f"clean_{act}"


def day_quality(
    coverage: pd.DataFrame,
    min_volume_share: float,
    min_push_share: float,
    activity_day: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-day completeness: how many hours look sane, and whether all do.

    GH Archive drops whole event types from individual hours (most often
    `PushEvent`) while the rest of the hour stays full -- or, on the worst days,
    inflates to several times its normal volume. Both modes pass a "file exists
    and is > 1 KB" check, so days are judged from the event mix instead, and a
    day is only *clean* when **every** hour of it clears both gates.

    A day that is missing pushes is poison for the curve that conditions on
    recency: users whose last event fell in the gap look stale, so their Δt is
    inflated and the decay looks faster than it is.
    """
    frame = coverage.copy()
    median = float(frame["events"].median())
    frame["push_share"] = frame["PushEvent"] / frame["events"].where(
        frame["events"] > 0, np.nan
    )
    frame["ok"] = (frame["events"] >= min_volume_share * median) & (
        frame["push_share"].fillna(0) >= min_push_share
    )
    # The audit now flags each activity separately, because the damage is
    # activity-specific: an hour can keep every push and lose every review. When
    # those flags are present they define cleanliness, and the push-share proxy
    # above is only the legacy fallback.
    per_activity = activity_columns(frame)
    for act in per_activity:
        frame[f"ok_{act}"] = frame[f"ok_{act}"].astype(bool)
    grouped = frame.groupby("day", sort=True)
    per_day = pd.DataFrame(
        {
            "hours": grouped["ok"].size(),
            "hours_ok": grouped["ok"].sum(),
            "events": grouped["events"].sum(),
            "pushes": grouped["PushEvent"].sum(),
        }
    )
    events = per_day["events"].to_numpy(dtype=float)
    per_day = per_day.assign(
        ok_share=per_day["hours_ok"].astype(float) / per_day["hours"].astype(float),
        push_share=np.divide(
            per_day["pushes"].to_numpy(dtype=float),
            events,
            out=np.full(events.shape, np.nan),
            where=events > 0,
        ),
        clean=per_day["hours_ok"] == per_day["hours"],
    )
    if per_activity:
        for act in per_activity:
            block = frame.groupby("day", sort=True)[f"ok_{act}"]
            per_day[f"hours_ok_{act}"] = block.sum()
            per_day[f"clean_{act}"] = per_day[f"hours_ok_{act}"] == per_day["hours"]
        if activity_day is not None and not activity_day.empty:
            # Prefer the day-level density band from coverage_activity.csv: the
            # hour flags can only say "every hour intact", which no sparse day
            # satisfies, so falling back to them rejects whole months.
            index = pd.DatetimeIndex(pd.to_datetime(per_day.index))
            for act in per_activity:
                for prefix in ("clean", "usable"):
                    column = f"{prefix}_{act}"
                    if column in activity_day.columns:
                        per_day[column] = pd.Series(
                            activity_day[column]
                            .reindex(index)
                            .fillna(False)
                            .to_numpy(),
                            index=per_day.index,
                        ).astype(bool)
        for act in per_activity:
            source = f"usable_{act}"
            if source in per_day.columns:
                per_day[f"usable_{act}"] = per_day[source].astype(bool)
            else:
                per_day[f"usable_{act}"] = per_day[f"clean_{act}"]
        per_day["clean"] = per_day[[f"clean_{act}" for act in per_activity]].all(axis=1)
        per_day["usable"] = per_day[[f"usable_{act}" for act in per_activity]].all(
            axis=1
        )
        # An hour is only *really* sane when all six streams survived it. The
        # legacy `ok` column tests volume plus push share, which answers a
        # narrower question and, against a corpus-wide median, rejects every
        # sparse hour of the recent months -- reporting that number as "sane
        # hours" understates coverage by a third.
        sane = frame.assign(
            _sane=frame[[f"ok_{act}" for act in per_activity]].all(axis=1)
        )
        per_day["hours_ok_all"] = sane.groupby("day", sort=True)["_sane"].sum()
    return cast(pd.DataFrame, per_day)


def select_anchors(
    anchors: list[pd.Timestamp],
    per_day: pd.DataFrame,
    mode: str,
    horizon_days: int,
    max_delta: int,
) -> list[pd.Timestamp]:
    """Keep the anchors whose label window (and lookback) has full coverage."""
    if mode == "off":
        return anchors
    usable = anchor_usability(anchors, per_day, mode, horizon_days, max_delta)
    return [anchor for anchor in anchors if bool(usable.loc[anchor].all())]


def anchor_usability(
    anchors: list[pd.Timestamp],
    per_day: pd.DataFrame,
    mode: str,
    horizon_days: int,
    max_delta: int,
) -> pd.DataFrame:
    """Per-(anchor, activity) coverage: may this cell enter the curve?

    The audit finds that loss is activity-specific -- an hour can keep every
    push and lose every review, and the months differ (`push` is usable on 45 of
    89 days, `pr_review_comment` on 20). Dropping a whole *day* therefore throws
    away good cells: a day whose `push` stream is intact but whose review stream
    collapsed is a valid push anchor and an invalid review anchor.

    Judging the cell instead of the day is what turns the strict filter from
    "keeps 0 anchors" into a usable panel, and it is also the honest encoding:
    a probability is only estimated from user-anchors whose own stream survived.
    """
    acts = clean_columns(per_day)
    day_index = pd.DatetimeIndex(pd.to_datetime(per_day.index))
    anchor_index = pd.DatetimeIndex(anchors)
    cols = pd.Index(LABEL_COLS)
    if mode == "off":
        return pd.DataFrame(True, index=anchor_index, columns=cols)
    if not acts:
        # Legacy audit without per-activity flags: every activity shares the
        # day-level verdict.
        clean = pd.Series(
            per_day["clean"].astype(bool).to_numpy(), index=day_index, dtype=bool
        )
        rows_legacy: list[dict[str, bool]] = []
        for anchor in anchors:
            window = pd.date_range(anchor, anchor + pd.Timedelta(days=horizon_days - 1))
            days = list(window)
            if mode == "strict":
                days += list(
                    pd.date_range(
                        anchor - pd.Timedelta(days=max_delta),
                        anchor - pd.Timedelta(days=1),
                    )
                )
            seen = clean.reindex(pd.DatetimeIndex(days)).astype("boolean")
            ok = bool(seen.fillna(False).to_numpy(dtype=bool).all())
            rows_legacy.append(dict.fromkeys(LABEL_COLS, ok))
        legacy = pd.DataFrame(rows_legacy, index=anchor_index)
        return cast(pd.DataFrame, legacy[cols])
    columns = [day_flag_column(act, per_day) for act in acts]
    clean_all = per_day[columns].astype(bool)
    # `per_day` is keyed by "YYYY-MM-DD" strings, so reindexing it with a
    # DatetimeIndex would return all-NaN and mark everything unusable.
    clean_all.index = day_index
    rows: list[dict[str, bool]] = []
    for anchor in anchors:
        window = pd.date_range(anchor, anchor + pd.Timedelta(days=horizon_days - 1))
        days = list(window)
        if mode == "strict":
            days += list(
                pd.date_range(
                    anchor - pd.Timedelta(days=max_delta),
                    anchor - pd.Timedelta(days=1),
                )
            )
        known = clean_all.reindex(pd.DatetimeIndex(days))
        # A day the audit never saw is unknown, not clean.
        row: dict[str, bool] = {}
        for act in acts:
            series = cast(pd.Series, known.loc[:, day_flag_column(act, per_day)])
            row[act] = bool(series.astype("boolean").fillna(False).to_numpy().all())
        rows.append(row)
    frame = pd.DataFrame(rows, index=anchor_index)
    for act in LABEL_COLS:
        if act not in frame.columns:
            frame[act] = False
    return cast(pd.DataFrame, frame[cols])


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) index runs where `flags` is true."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def build_panel(
    tables: dict[str, pd.DataFrame],
    users: pd.DataFrame,
    streams: dict[str, pd.DataFrame],
    anchors: list[pd.Timestamp],
    horizon_days: int,
    max_delta: int,
    verbose: bool = True,
    usability: pd.DataFrame | None = None,
) -> Panel:
    """Fold every anchor's user panel into bound (n, positives) counts."""
    panel = Panel(max_delta=max_delta, n_act=len(LABEL_COLS))
    first_seen = users.set_index("user_id")["first_seen_at"]
    seen = users.set_index("user_id")
    for anchor in anchors:
        known = cast(pd.Series, first_seen[first_seen < anchor])
        if known.empty:
            continue
        ids = known.index.to_numpy(dtype="int64")
        frame = seen.loc[ids].reset_index()
        y = labels_at(tables, anchor, horizon_days, frame)
        dt_any = days_since_last(streams["any"], anchor, ids)
        dt_same = np.column_stack(
            [days_since_last(streams[act], anchor, ids) for act in LABEL_COLS]
        )
        usable = (
            None if usability is None else usability.loc[anchor, :].to_numpy(dtype=bool)
        )
        panel.add(dt_any, dt_same, y, anchor, usable)
        if verbose:
            print(
                f"  {anchor:%Y-%m-%d %H:%M}  users={len(ids):5d}  "
                f"dt_any p50={np.nanmedian(dt_any):5.2f}d  "
                f"censored={int(np.isnan(dt_any).sum()):4d}  "
                + " ".join(
                    f"{act}={int(y[:, j].sum())}" for j, act in enumerate(LABEL_COLS)
                )
            )
    return panel


def smoothed_curve(panel: Panel, alpha: float) -> pd.DataFrame:
    """`curve_table` plus the Laplace-smoothed p per (axis, activity)."""
    curve = curve_table(panel)
    curve["p_smooth"] = np.nan
    for axis in AXES:
        for act in LABEL_COLS:
            mask = (curve["axis"] == axis) & (curve["activity"] == act)
            curve.loc[mask, "p_smooth"] = smoothing(
                cast(pd.DataFrame, curve.loc[mask]), alpha
            ).to_numpy()
    return curve


def _mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
        }
    )
    return plt


def plot_panels(
    curve: pd.DataFrame,
    fits: dict[str, dict[str, dict[str, float]]],
    axis: str,
    panel: Panel,
    path: Path,
    min_support: int,
) -> None:
    """6 panels: empirical curve with Wilson CI, support, exponential fit."""
    plt = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for ax, act in zip(axes.ravel(), LABEL_COLS, strict=True):
        rows = curve_slice(curve, axis, act)
        dt_rows = curve_slice(curve, axis, act, numeric=True)
        x = dt_rows["dt_days"].to_numpy()
        p = dt_rows["p"].to_numpy()
        n = dt_rows["n"].to_numpy()
        col = COLORS[act]
        # Bins with no rows carry no information: they must not be drawn, else
        # Laplace smoothing and the CI manufacture a plausible-looking value.
        valid = n > 0
        supported = n >= min_support
        # Draw only bins that clear `min_support`: at the tail a handful of users
        # and one positive reads as p = 0.3, which is noise that would dominate
        # the eye at the right-hand edge. The bar row still shows real support.
        keep = valid & supported
        last_supported = float(x[supported].max()) if supported.any() else 0.0

        ax.axvspan(
            last_supported + 0.5,
            panel.max_delta + 0.5,
            color="0.9",
            zorder=0,
            lw=0,
        )
        # An activity with no usable anchor at all has an empty `valid` mask;
        # `n[valid].max()` would then raise on a zero-size reduction.
        if valid.any():
            ax.bar(
                x[valid],
                n[valid] / max(float(n[valid].max()), 1.0) * 0.02,
                color="0.75",
                width=0.85,
                zorder=1,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "no usable anchors",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="0.45",
            )
        ax.fill_between(
            x[keep],
            dt_rows["ci_lo"].to_numpy()[keep],
            dt_rows["ci_hi"].to_numpy()[keep],
            color=col,
            alpha=0.18,
            lw=0,
            zorder=2,
        )
        ax.plot(
            x[keep],
            p[keep],
            "o-",
            color=col,
            ms=3.5,
            lw=1.4,
            zorder=4,
            label="empirical",
        )
        smoothed = smoothing(dt_rows, 0.5).to_numpy()
        if keep.any():
            ax.plot(
                x[keep],
                smoothed[keep],
                color=col,
                lw=0.8,
                ls=":",
                alpha=0.85,
                zorder=3,
                label="Laplace smoothed",
            )

        fit = fits[act][axis]
        if np.isfinite(fit["tau_days"]) and np.isfinite(fit["floor"]):
            grid = np.arange(0, last_supported + 0.001, 0.25)
            ax.plot(
                grid,
                fit["floor"] + fit["amp"] * np.exp(-grid / fit["tau_days"]),
                color="k",
                ls="--",
                lw=1.0,
                alpha=0.75,
                zorder=5,
                label=f"fit tau={fit['tau_days']:.1f}d",
            )
        for k in (7, 14, 21):
            if k <= last_supported:
                ax.axvline(k, color="0.5", lw=0.6, ls="-.", alpha=0.5, zorder=0)

        thin = int((~supported & valid).sum())
        ax.set_title(f"{act}  [{shape_label(fit)}]")
        top = float(np.nanmax(p[valid])) if valid.any() else 0.02
        ax.set_ylim(0, max(top, 0.02) * 1.18)
        ax.text(
            0.02,
            0.94,
            f"n={int(n.sum())}  thin bins={thin}\n"
            f"P(never)={float(rows.loc[rows['bucket'] == NEVER, 'p'].iloc[0]):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="0.35",
        )
        if last_supported < panel.max_delta:
            ax.text(
                (last_supported + panel.max_delta + 1) / 2,
                0.5,
                f"no support\n(dt > {int(last_supported)}d unobservable)",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=6.5,
                color="0.45",
                rotation=90,
            )
        ax.set_ylabel(
            "P(active in next 7d)" if act in ("open_pr", "pr_review_comment") else ""
        )
        ax.set_xlabel("dt (days)" if act in ("pr_review_comment", "push") else "")
        ax.legend(
            loc="upper right", fontsize=6.5, frameon=False, bbox_to_anchor=(1.0, 0.80)
        )
    fig.suptitle(
        f"P(activity in the next 7 days) vs {AXIS_DESC[axis]}\n"
        "bars = per-bin support (relative), line shading = 95% Wilson CI, "
        "dash-dot = 7/14/21 days, grey band = no data",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def plot_overlay(
    curve: pd.DataFrame, panel: Panel, path: Path, min_support: int
) -> None:
    """All six activities on one axis, log scale, one panel per axis."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, axis in zip(axes, AXES, strict=True):
        for act in LABEL_COLS:
            dt_rows = curve_slice(curve, axis, act, numeric=True)
            keep = dt_rows["n"] >= max(min_support, 5)
            ax.plot(
                dt_rows.loc[keep, "dt_days"],
                dt_rows.loc[keep, "p"],
                "o-",
                ms=3,
                lw=1.3,
                color=COLORS[act],
                label=act,
            )
        ax.set_yscale("log")
        ax.set_xlabel("dt (days)")
        ax.set_title(AXIS_DESC[axis])
        ax.axhline(0, color="0.6", lw=0.6)
        for k in (7, 14, 21):
            ax.axvline(k, color="0.5", lw=0.6, ls="-.", alpha=0.45)
    axes[0].set_ylabel("P(active in next 7d)  [log]")
    axes[1].legend(fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle("Decay curves: sharp exponential vs steady / periodic", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_distribution(
    panel: Panel, curve: pd.DataFrame, path: Path, min_support: int
) -> None:
    """How much support each dt bin has, and the P(activity | dt) surface."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    n_any = panel.n[0, 0]
    thin = panel.n[0, 0] < min_support
    cols = ["#4c78a8" if not t else "#c9c9c9" for t in thin]
    ax.bar(np.arange(panel.max_delta + 1), n_any[: panel.max_delta + 1], color=cols)
    ax.set_xlabel("dt = days since last event (any type)")
    ax.set_ylabel("user-anchor rows")
    ax.set_title(
        f"Support per dt bin (grey = n < {min_support}; "
        f"never={int(n_any[panel.max_delta + 1])}, >max={int(n_any[panel.max_delta + 2])})"
    )

    ax = axes[1]
    grid = np.full((len(LABEL_COLS), panel.max_delta + 1), np.nan)
    for j, act in enumerate(LABEL_COLS):
        rows = curve_slice(curve, "any", act, numeric=True)
        vals = rows["p"].to_numpy()
        nn = rows["n"].to_numpy()
        vals = np.where(nn >= min_support, vals, np.nan)
        grid[j, : len(vals)] = vals
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")
    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=(-0.5, panel.max_delta + 0.5, -0.5, len(LABEL_COLS) - 0.5),
    )
    ax.set_yticks(range(len(LABEL_COLS)))
    ax.set_yticklabels(LABEL_COLS, fontsize=8)
    ax.set_xlabel("dt (days since last event, any type)")
    ax.set_title(f"P(activity in next 7d | dt), bins with n < {min_support} greyed")
    fig.colorbar(im, ax=ax, label="probability")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_clean_vs_all(
    curve: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path,
    min_support: int,
) -> None:
    """Same-activity curves with and without the completeness filter.

    Dashed = every anchor (contaminated by the missing hours); solid = clean
    anchors only. Where the two agree the activity was never affected, so the
    cheap full-anchor number can be used; where they separate, the gap *is* the
    data loss and only the solid curve is trustworthy.
    """
    plt = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    gap = (
        comparison.set_index("activity")["prevalence_clean"]
        - (comparison.set_index("activity")["prevalence_all"])
    )
    for ax, act in zip(axes.ravel(), LABEL_COLS, strict=True):
        for quality, style, colour, label in (
            ("all", "--", "#9a9a9a", "all anchors"),
            ("clean", "-", "#4c78a8", "clean anchors"),
        ):
            rows = curve_slice(
                cast(pd.DataFrame, curve[curve["quality"] == quality]),
                "same",
                act,
                numeric=True,
            )
            if rows.empty:
                continue
            ok = rows["n"].to_numpy() >= min_support
            x = rows["dt_days"].to_numpy()[ok]
            y = rows["p_smooth"].to_numpy()[ok]
            ax.plot(x, y, style, color=colour, lw=1.8, label=label)
        delta = float(gap.get(act, 0.0))
        ax.set_title(
            f"{act}   prevalence {delta * 100:+.2f} pp when filtered",
            fontsize=9,
        )
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[1, :]:
        ax.set_xlabel("Δt = days since the user's last event of that type")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(activity within 7 days)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Effect of the GH Archive completeness filter (dashed vs solid)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_rarity(summary: pd.DataFrame, events: pd.DataFrame, path: Path) -> None:
    """Prevalence, pos_weight, positives per window, raw event counts."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    acts = summary["activity"].tolist()
    cols = [COLORS[a] for a in acts]

    ax = axes[0]
    ax.bar(acts, summary["prevalence_pooled"] * 100, color=cols)
    for i, v in enumerate(summary["prevalence_pooled"] * 100):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("P(active in a 7-day window) [%]")
    ax.set_title("Class prevalence (pooled over anchors)")
    ax.tick_params(axis="x", rotation=30, labelsize=7.5)

    ax = axes[1]
    ax.bar(acts, summary["pos_weight"], color=cols)
    ax.set_yscale("log")
    for i, v in enumerate(summary["pos_weight"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("pos_weight = neg / pos  [log]")
    ax.set_title("BCE positive weight per activity")
    ax.tick_params(axis="x", rotation=30, labelsize=7.5)

    ax = axes[2]
    x = np.arange(len(acts))
    ax.bar(
        x - 0.2,
        summary["positives_per_window_mean"],
        width=0.4,
        color=cols,
        label="mean per window",
    )
    ax.errorbar(
        x,
        summary["positives_per_window_mean"],
        yerr=[
            summary["positives_per_window_mean"] - summary["positives_per_window_min"],
            summary["positives_per_window_max"] - summary["positives_per_window_mean"],
        ],
        fmt="none",
        ecolor="0.3",
        lw=1,
        capsize=2,
        label="min..max",
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        events.set_index("activity").loc[acts, "events"],
        "k^",
        ms=4,
        label="raw events (right)",
    )
    ax2.set_ylabel("raw events in the tables")
    ax.set_xticks(x)
    ax.set_xticklabels(acts, rotation=30, fontsize=7.5)
    ax.set_ylabel("positive users per 7-day window")
    ax.set_title("Positives per window vs raw events")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax2.legend(fontsize=7, frameon=False, loc="upper right")

    fig.suptitle("Class rarity and positive counts", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ write --


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _num(value: float, spec: str) -> str:
    """Format a fit statistic, showing `-` instead of a NaN that would read as 0."""
    return format(value, spec) if np.isfinite(value) else "-"


def _period_cell(fit: dict[str, float]) -> str:
    """Cycle length for the table, or an explicit "none" when unproven."""
    period = fit.get("dominant_period_days", float("nan"))
    return f"{period:.1f}" if np.isfinite(period) else "none"


def write_markdown(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    fits: dict[str, dict[str, dict[str, float]]],
    panel: Panel,
    config: dict[str, Any],
    official: pd.DataFrame,
    path: Path,
    comparison: pd.DataFrame | None = None,
    coverage: dict[str, Any] | None = None,
    clean_official: pd.DataFrame | None = None,
    fits_unfiltered: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    anchors = panel.windows
    lines: list[str] = []
    lines.append("# Temporal decay and class rarity\n")
    lines.append(
        f"Anchors: {len(anchors)} daily cutoffs "
        f"({anchors[0]['anchor'][:10]} .. {anchors[-1]['anchor'][:10]}), "
        f"label horizon {config['horizon_days']}d, "
        f"dt grid 0..{config['max_delta']}d.\n"
    )
    lines.append("## Class rarity and loss weights\n")
    lines.append(
        "Denominator here is *known users at each anchor* (pooled over anchors), so "
        "these prevalences are higher than the all-users numbers further down.\n\n"
    )
    lines.append(
        "| activity | prevalence | pos_weight | pos / window (mean [min..max]) "
        "| windows all-zero | raw events | users with event |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    flat = [f for act in LABEL_COLS for f in fits[act].values()]
    n_bump = sum(1 for f in flat if np.isfinite(f["bump_7"]) and f["bump_7"] >= 0.15)
    n_cycle = sum(1 for f in flat if np.isfinite(f["dominant_period_days"]))

    ev = events.set_index("activity")
    for _, r in summary.iterrows():
        act = str(r["activity"])
        lines.append(
            f"| {act} | {r['prevalence_pooled'] * 100:.2f}% | "
            f"{r['pos_weight']:.2f} | {r['positives_per_window_mean']:.1f} "
            f"[{r['positives_per_window_min']}..{r['positives_per_window_max']}] | "
            f"{r['share_windows_all_zero'] * 100:.0f}% | "
            f"{int(ev.loc[act, 'events'])} | "
            f"{ev.loc[act, 'users_with_event_share'] * 100:.0f}% |\n"
        )
    lines.append("\n## Decay shape\n")
    lines.append(
        "| activity | axis | shape | tau (d) | half-life (d) | p(1)/p(7) | "
        "bump@7 | week2/week1 | cycle (d) | p(never) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    for _, r in summary.iterrows():
        for axis, suffix, pcol in (
            ("any", "any", "p_given_never_any"),
            ("same", "same", "p_given_never_same"),
        ):
            f = fits[str(r["activity"])][axis]
            lines.append(
                f"| {r['activity']} | {axis} | {r[f'shape_{suffix}']} | "
                f"{_num(f['tau_days'], '.2f')} | "
                f"{_num(f['half_life_days'], '.2f')} | "
                f"{_num(f['decay_ratio_1_7'], '.2f')} | "
                f"{_num(f['bump_7'], '+.2f')} | "
                f"{_num(f['plateau_ratio'], '.2f')} | {_period_cell(f)} | "
                f"{r[pcol]:.3f} |\n"
            )
    lines.append(
        f"\n## Weights on the official validation window\n"
        f"All {int(official['users'].iloc[0])} cohort users scored at anchor "
        f"{config['official_anchor']} (this is the denominator "
        "`splits/answer.csv` uses: a user who had not appeared yet still counts as "
        "a negative, which is why these prevalences are lower than the pooled "
        "known-user numbers above).\n"
    )
    lines.append(
        "| activity | positives | prevalence | pos_weight | sqrt pos_weight |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    for _, r in official.iterrows():
        lines.append(
            f"| {r['activity']} | {int(r['positives'])} | "
            f"{r['prevalence'] * 100:.2f}% | {r['pos_weight']:.2f} | "
            f"{r['pos_weight_sqrt']:.2f} |\n"
        )

    if clean_official is not None:
        lines.append(
            "\n### The same weights measured on clean anchors\n"
            f"Mean over {int(clean_official['anchors'].iloc[0])} fully-covered anchors, "
            "same all-users denominator. Use these instead when the official anchor "
            "week itself is incomplete: a thin hour loses whichever event types its "
            "shard carried, so a broken week deflates *every* activity's positives, "
            "not only push.\n"
        )
        lines.append(
            "| activity | prevalence (mean) | range | pos_weight | sqrt pos_weight |\n"
            "| --- | --- | --- | --- | --- |\n"
        )
        for _, r in clean_official.iterrows():
            lines.append(
                f"| {r['activity']} | {r['prevalence_mean'] * 100:.2f}% | "
                f"{r['prevalence_min'] * 100:.2f}-{r['prevalence_max'] * 100:.2f}% | "
                f"{r['pos_weight']:.2f} | {r['pos_weight_sqrt']:.2f} |\n"
            )

    any_taus = [fits[act]["any"]["tau_days"] for act in LABEL_COLS]
    any_taus = [t for t in any_taus if np.isfinite(t)]
    lo = min(any_taus) if any_taus else float("nan")
    hi = max(any_taus) if any_taus else float("nan")
    lines.append("\n## How to use these numbers\n")
    lines.append(
        "- `TEMPORAL_EDGE_TAU_DAYS[a]` comes from the **same-type** axis: it is the "
        "decay of P(a) as a function of days since the user's own last `a` event, "
        "so `w = exp(-dt_a / tau_a)` is well scaled per activity. Using the "
        "any-type tau would over-fade every activity "
        f"(any-type tau is only {lo:.1f}-{hi:.1f} d).\n"
        "- `POS_WEIGHTS[a] = (1-p)/p` is the standard `BCEWithLogitsLoss` weight "
        "if you train on the pooled anchor panel; `sqrt` is the gentler variant "
        "that does not over-optimise macro-F1 on push. The snippet uses the "
        "all-users clean-anchor table, which matches the `splits/answer.csv` "
        "denominator.\n"
        f"- Weekly resonance: {n_bump}/{2 * len(LABEL_COLS)} curves have "
        f"bump@7 >= +0.15 and {n_cycle}/{2 * len(LABEL_COLS)} pass the sinusoid "
        f"scan (r2 >= {PERIOD_R2_GATE:.2f} and amplitude >= {PERIOD_AMP_GATE:.0%} "
        "of the curve level), so "
        f"{'periodicity is real for ' + str(n_bump) + ' activities' if n_bump else 'the shape is decay to a floor, not periodicity'}"
        ".\n"
    )
    # Recommended values. Two different biases pull in opposite directions:
    #   * the full 61-anchor panel is the better estimate for an activity whose
    #     rows survived, since it has 8x the anchors and the bad days are a
    #     minority it averages over;
    #   * the 8-anchor clean panel is the better estimate for an activity whose
    #     rows were lost, because there a zero-push day is not noise, it is the
    #     data loss itself stretching every affected user's dt by weeks.
    # "Rows were lost" is read off the prevalence shift, not hard-coded, so the
    # rule still holds if a later download shifts the damage to another activity.
    shifted: set[str] = set()
    if comparison is not None and not comparison.empty:
        shifted = {
            str(r["activity"])
            for _, r in comparison.iterrows()
            if abs(float(r["prevalence_clean"]) - float(r["prevalence_all"])) > 0.05
        }
    weights = (
        {
            str(k): float(v)
            for k, v in clean_official.set_index("activity")["pos_weight"].items()
        }
        if clean_official is not None
        else {str(r["activity"]): float(r["pos_weight"]) for _, r in summary.iterrows()}
    )

    # The unfiltered pass ran over every requested anchor, not just `panel`'s.
    n_full = len(panel.windows)
    sizes = f"{n_full}-anchor" if fits_unfiltered is None else "all-anchor"
    lines.append("\n## Config snippet\n")
    lines.append("```python\n")
    lines.append(
        "# edge weight = exp(-dt_days / tau), dt = days since the user's own last\n"
        "# event of that type\n"
        f"# contaminated = {sorted(shifted) or 'none'} (those take the clean-anchor "
        f"tau; the rest take the {sizes} fit)\n"
        "TEMPORAL_EDGE_TAU_DAYS = {\n"
    )
    for _, r in summary.iterrows():
        act = str(r["activity"])
        src = fits
        if act not in shifted and fits_unfiltered is not None:
            src = fits_unfiltered
        tau = src[act]["same"]["tau_days"]
        tau = tau if np.isfinite(tau) else float(config["fallback_tau"])
        note = "  # clean anchors only" if act in shifted else ""
        lines.append(f"    '{act}': {tau:.2f},{note}\n")
    lines.append("}\n\nPOS_WEIGHTS = {  # BCEWithLogitsLoss(pos_weight=...)\n")
    for act in LABEL_COLS:
        lines.append(f"    '{act}': {weights[act]:.2f},\n")
    lines.append("}\n```\n")

    if coverage:
        lines.append("\n## Data completeness\n")
        lines.append(
            "GH Archive drops whole event types from individual hours. Each of the "
            "six activities is audited on its own (\n"
            "`scripts/verify_raw_data.py`): an hour passes `ok_<activity>` when it "
            "carries at least "
            f"{config.get('min_activity_share', 0.30):.0%} of that activity's local "
            "±7-day median density and no more than "
            f"{config.get('max_activity_ratio', 4.0):.0f}× it. A day is *clean* for "
            "an activity when every hour passes; *usable* when the day's total sits "
            "inside the band and at least "
            f"{config.get('min_day_hours_share', 0.75):.0%} of its hours pass.\n"
        )
        lines.append(
            f"- hours scanned: {coverage['hours']}, sane for all six: "
            f"{coverage['hours_ok']} "
            f"({coverage['hours_ok'] / max(coverage['hours'], 1) * 100:.1f}%), "
            f"unreadable files: {coverage.get('unreadable', 0)}\n"
            f"- days: {coverage['days']}, clean for all six: {coverage['days_clean']}, "
            f"usable for all six: {coverage.get('days_usable', coverage['days_clean'])}\n"
            f"- longest clean run: {coverage['longest_run']}\n"
            f"- anchors requested: {coverage['anchors_asked']}, kept by "
            f"`--clean-days {config.get('clean_days', 'label')}`: "
            f"{coverage['anchors_kept']}\n"
        )
        per_act_clean = coverage.get("days_clean_per_activity")
        per_act_usable = coverage.get("days_usable_per_activity")
        if per_act_clean:
            lines.append(
                "\n| activity | clean days | usable days | anchors usable |\n"
                "| --- | --- | --- | --- |\n"
            )
            for act in LABEL_COLS:
                lines.append(
                    f"| {act} | {per_act_clean.get(act, 0)} | "
                    f"{(per_act_usable or {}).get(act, 0)} | "
                    f"{coverage['anchors_per_activity'].get(act, 0)} |\n"
                )
        if coverage["bad_days"]:
            shown = ", ".join(coverage["bad_days"][:18])
            more = len(coverage["bad_days"]) - 18
            lines.append(f"- days with an unusable hour: {shown}")
            lines.append(f"{f' (+{more} more)' if more > 0 else ''}\n")
        if coverage["anchors_kept"] < 10:
            lines.append(
                f"\n> Only {coverage['anchors_kept']} anchor(s) survive the filter, and "
                "they sit inside one contiguous clean stretch. Treat the filtered "
                "curves as low-power: the Δt lookback still reaches back across the "
                "bad days, so a user whose last event fell in a gap looks staler "
                "than they were and the fitted tau is biased **up**.\n"
            )
        if comparison is not None and not comparison.empty:
            lines.append("\n### Filtered vs unfiltered\n")
            lines.append(
                "| activity | prev (all) | prev (clean) | tau (all) | tau (clean) | "
                "p(1)/p(7) all | p(1)/p(7) clean |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
            )
            for _, r in comparison.iterrows():
                lines.append(
                    f"| {r['activity']} | {r['prevalence_all'] * 100:.2f}% | "
                    f"{r['prevalence_clean'] * 100:.2f}% | "
                    f"{_num(float(r['tau_all']), '.2f')} | "
                    f"{_num(float(r['tau_clean']), '.2f')} | "
                    f"{_num(float(r['ratio_all']), '.2f')} | "
                    f"{_num(float(r['ratio_clean']), '.2f')} |\n"
                )
            shifted_names = (
                [
                    str(r["activity"])
                    for _, r in comparison.iterrows()
                    if abs(float(r["prevalence_clean"]) - float(r["prevalence_all"]))
                    > 0.05
                ]
                if not comparison.empty
                else []
            )
            example = ""
            other_max = 0.0
            for _, r in comparison.iterrows():
                name = str(r["activity"])
                shift = abs(float(r["prevalence_clean"]) - float(r["prevalence_all"]))
                if name in shifted_names:
                    example = (
                        f"so `{name}` going "
                        f"{float(r['prevalence_all']) * 100:.1f}% -> "
                        f"{float(r['prevalence_clean']) * 100:.1f}% is the data loss"
                    )
                else:
                    other_max = max(other_max, shift)
            lines.append(
                "\nTwo things move here, and only one of them is contamination. "
                "*Prevalence* is the direct read-out: it shifts sharply for the "
                "activity whose rows are actually missing, "
                + (
                    f"{example}, while every other activity stays inside "
                    f"{other_max * 100:.1f} points. "
                    if example
                    else "and no activity moved by more than 5 points, so nothing "
                    "here looks contaminated. "
                )
                + "*Tau* is not: the clean panel holds fewer anchors and its Δt "
                "lookback still crosses the bad days, which stretches the tail and "
                "inflates tau. So for an unaffected activity prefer the "
                f"{n_full}-anchor unfiltered tau, and for "
                f"{', '.join(f'`{n}`' for n in shifted_names) or 'a damaged activity'} "
                "prefer the clean prevalence/pos_weight.\n"
            )
    path.write_text("".join(lines))


# ------------------------------------------------------------------- main --


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="empirical P(activity in next N days | days since last event)"
    )
    ap.add_argument(
        "--db", default=DEFAULT_DB, help=f"parquet folder (default {DEFAULT_DB})"
    )
    ap.add_argument("--out", default="temporal_output", help="output folder")
    ap.add_argument("--max-delta", type=int, default=30, help="largest dt bin, in days")
    ap.add_argument("--horizon-days", type=int, default=7, help="label horizon in days")
    ap.add_argument(
        "--anchor-step", default="1D", help="spacing between anchors (pandas freq)"
    )
    ap.add_argument(
        "--first-anchor",
        default=None,
        help="first anchor, naive UTC (default: data start + 1 day)",
    )
    ap.add_argument(
        "--last-anchor",
        default=None,
        help="last anchor, naive UTC (default: last row - horizon)",
    )
    ap.add_argument(
        "--min-support",
        type=int,
        default=20,
        help="minimum rows in a bin to trust/fit it",
    )
    ap.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="Laplace alpha for the smoothed curve",
    )
    ap.add_argument(
        "--fallback-tau",
        type=float,
        default=7.0,
        help="tau used in the config snippet when a fit fails",
    )
    ap.add_argument("--no-plots", action="store_true", help="skip the PNG figures")
    ap.add_argument(
        "--coverage",
        default="raw_data/coverage.csv",
        help="hour-level audit from scripts/verify_raw_data.py",
    )
    ap.add_argument(
        "--coverage-activity",
        default="raw_data/coverage_activity.csv",
        help="day x activity audit (carries the day-level density band)",
    )
    ap.add_argument(
        "--clean-days",
        default="label",
        choices=("off", "label", "strict"),
        help=(
            "which anchors to trust: 'off' keeps all, 'label' needs every day of "
            "the label window to be fully covered, 'strict' also needs the whole "
            "dt lookback covered"
        ),
    )
    ap.add_argument(
        "--min-hour-volume-share",
        type=float,
        default=0.25,
        help="an hour is thin below this share of the corpus median event count",
    )
    ap.add_argument(
        "--min-hour-push-share",
        type=float,
        default=0.30,
        help="an hour is mis-composed below this PushEvent share",
    )
    ap.add_argument(
        "--no-unfiltered",
        action="store_true",
        help="skip the unfiltered comparison pass (only meaningful with --clean-days)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tables = load_tables(args.db)
    if not tables:
        print(f"no parquet files in {args.db}")
        return 1

    row_stamps = [
        pd.to_datetime(tables[name][col])
        for name, col in ROW_TIME.items()
        if name in tables and col in tables[name].columns
    ]
    data_start = cast(pd.Timestamp, min(s.min() for s in row_stamps))
    data_end = cast(pd.Timestamp, max(s.max() for s in row_stamps))

    users = cohort_users(tables)
    user_ids = users["user_id"].to_numpy(dtype="int64")

    first = (
        parse_cutoff(args.first_anchor)
        if args.first_anchor
        else cast(pd.Timestamp, data_start.normalize() + pd.Timedelta(days=1))
    )
    last = (
        parse_cutoff(args.last_anchor)
        if args.last_anchor
        else cast(
            pd.Timestamp, data_end.normalize() - pd.Timedelta(days=args.horizon_days)
        )
    )
    if last < first:
        print(f"no anchor fits: first={first} last={last}")
        return 1
    anchors = list(pd.date_range(first, last, freq=args.anchor_step))

    print(f"data window      : {data_start} .. {data_end}")
    print(f"cohort users     : {len(user_ids)}")
    print(
        f"anchors          : {len(anchors)} ({anchors[0]} .. {anchors[-1]}, "
        f"step {args.anchor_step}, horizon {args.horizon_days}d)"
    )
    print(f"dt grid          : 0..{args.max_delta} days")

    streams = activity_streams(tables, user_ids)

    coverage: dict[str, Any] | None = None
    per_day: pd.DataFrame | None = None
    coverage_path = Path(args.coverage)
    raw_cov: pd.DataFrame | None = None
    if coverage_path.exists():
        raw_cov = pd.read_csv(coverage_path, dtype={"day": str})
        activity_path = Path(args.coverage_activity)
        activity_day = (
            load_activity_coverage(activity_path) if activity_path.exists() else None
        )
        per_day = day_quality(
            load_coverage(coverage_path),
            args.min_hour_volume_share,
            args.min_hour_push_share,
            activity_day,
        )
    elif args.clean_days != "off":
        print(
            f"warning: {coverage_path} not found, running unfiltered "
            "(build it with `python -m scripts.verify_raw_data`)"
        )

    # An anchor outside the downloaded data contributes no positives and a huge
    # dt for every user, which reads as "nobody is active" rather than "not
    # observed". Restrict to days the corpus actually covers before any
    # quality filtering, so a download gap cannot look like decay.
    if per_day is not None and not per_day.empty:
        covered = pd.DatetimeIndex(pd.to_datetime(per_day.index))
        in_data = [a for a in anchors if a.normalize() in covered]
        if len(in_data) < len(anchors):
            print(
                f"anchors in data  : {len(in_data)}/{len(anchors)} "
                f"(dropped {len(anchors) - len(in_data)} with no audited hours)"
            )
        anchors = in_data
        if not anchors:
            print("no anchor falls inside the audited data")
            return 1

    kept = anchors
    usability: pd.DataFrame | None = None
    if per_day is not None and args.clean_days != "off":
        # Usability is per (anchor, activity), not per anchor: an anchor stays
        # in as long as *some* activity has an intact stream around it, and each
        # activity then contributes only its own usable cells.
        usability = anchor_usability(
            anchors, per_day, args.clean_days, args.horizon_days, args.max_delta
        )
        kept = [anchor for anchor in anchors if bool(usability.loc[anchor].any())]
        clean_flag = per_day["clean"].astype(bool).to_numpy()
        runs = _runs(clean_flag)
        longest = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 0)
        days = pd.to_datetime(per_day.index)
        unreadable = 0 if raw_cov is None else int((~raw_cov["events"].notna()).sum())
        coverage = {
            "unreadable": unreadable,
            "hours": int(per_day["hours"].sum()),
            "hours_ok": int(
                per_day["hours_ok_all"].sum()
                if "hours_ok_all" in per_day.columns
                else per_day["hours_ok"].sum()
            ),
            "days": len(per_day),
            "days_clean": int(clean_flag.sum()),
            "days_usable": int(per_day["usable"].astype(bool).sum())
            if "usable" in per_day.columns
            else int(clean_flag.sum()),
            "days_clean_per_activity": {
                act: (
                    int(per_day[f"clean_{act}"].astype(bool).sum())
                    if f"clean_{act}" in per_day.columns
                    else int(clean_flag.sum())
                )
                for act in LABEL_COLS
            },
            "days_usable_per_activity": {
                act: (
                    int(per_day[f"usable_{act}"].astype(bool).sum())
                    if f"usable_{act}" in per_day.columns
                    else int(clean_flag.sum())
                )
                for act in LABEL_COLS
            },
            "longest_run": (
                f"{days[longest[0]]:%Y-%m-%d} .. {days[longest[1] - 1]:%Y-%m-%d} "
                f"({longest[1] - longest[0]}d)"
                if runs
                else "none"
            ),
            "anchors_asked": len(anchors),
            "anchors_kept": len(kept),
            "anchors_per_activity": {
                act: int(usability[act].sum()) for act in LABEL_COLS
            },
            "bad_days": [f"{d:%Y-%m-%d}" for d in days[~clean_flag]],
            "kept": [f"{a:%Y-%m-%d}" for a in kept],
        }
        print(
            f"completeness     : {coverage['days_clean']}/{coverage['days']} clean "
            f"days ({coverage['hours_ok']}/{coverage['hours']} sane hours), "
            f"longest run {coverage['longest_run']}"
        )
        print(
            f"anchors kept     : {len(kept)}/{len(anchors)} anchors, "
            f"per activity "
            + " ".join(
                f"{act}={coverage['anchors_per_activity'][act]}" for act in LABEL_COLS
            )
            + f" (--clean-days {args.clean_days})"
        )
    if not kept:
        print("no anchor survives the completeness filter; try --clean-days off")
        return 1

    panel = build_panel(
        tables,
        users,
        streams,
        kept,
        args.horizon_days,
        args.max_delta,
        usability=None
        if usability is None
        else cast(pd.DataFrame, usability.loc[kept]),
    )

    official = cast(
        pd.Timestamp, data_end.normalize() - pd.Timedelta(days=args.horizon_days)
    )
    # Rarity depends on the denominator: a user who does not exist yet at the
    # anchor is censored, but the official answer file still scores them 0.
    # Both views are reported so pos_weight can match whichever is trained on.
    official_all = labels_at(tables, official, args.horizon_days, users)
    all_pos = official_all.sum(axis=0)
    all_prev = all_pos / len(official_all)
    official_rows = pd.DataFrame(
        {
            "activity": LABEL_COLS,
            "positives": all_pos,
            "users": len(official_all),
            "prevalence": all_prev,
            "pos_weight": (1.0 - all_prev) / np.where(all_prev > 0, all_prev, np.nan),
            "pos_weight_sqrt": np.sqrt(
                (1.0 - all_prev) / np.where(all_prev > 0, all_prev, np.nan)
            ),
        }
    )
    # The official anchor can land on a week GH Archive lost, which deflates the
    # positives for every activity, not just push (a thin hour loses whichever
    # event types its shard carried). The same all-users denominator measured on
    # clean anchors is the like-for-like replacement.
    clean_rows: pd.DataFrame | None = None
    if kept != anchors:
        per_anchor = np.column_stack(
            [
                labels_at(tables, anchor, args.horizon_days, users).mean(axis=0)
                for anchor in kept
            ]
        )
        mean_prev = per_anchor.mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = (1.0 - mean_prev) / mean_prev
        clean_rows = pd.DataFrame(
            {
                "activity": LABEL_COLS,
                "prevalence_mean": mean_prev,
                "prevalence_min": per_anchor.min(axis=1),
                "prevalence_max": per_anchor.max(axis=1),
                "users": len(users),
                "anchors": len(kept),
                "pos_weight": weights,
                "pos_weight_sqrt": np.sqrt(weights),
            }
        )
        print(
            "clean-anchor weights (all users): "
            + " ".join(
                f"{r['activity']!s}={r['pos_weight']:.2f}"
                for _, r in clean_rows.iterrows()
            )
        )

    answer_path = Path("splits/answer.csv")
    if answer_path.exists():
        truth = pd.read_csv(answer_path).set_index("UserID")
        mine = pd.DataFrame(
            official_all,
            index=pd.Index(users["github_user_id"].to_numpy(dtype="int64")),
            columns=pd.Index(LABEL_COLS),
        )
        shared = mine.index.intersection(truth.index)
        ref = cast(pd.DataFrame, truth.loc[shared, LABEL_COLS])
        agree = float((mine.loc[shared].to_numpy() == ref.to_numpy()).mean())
        print(
            f"\ncross-check vs splits/answer.csv at {official}: "
            f"{len(shared)}/{len(mine)} users match, cell agreement {agree:.4f}"
        )

    curve = smoothed_curve(panel, args.smooth_alpha)
    summary, fits = summarise(panel, curve, args.min_support, args.smooth_alpha)
    events = event_stats(tables, user_ids)

    # The filtered panel is authoritative, but the unfiltered one is kept so the
    # contamination is visible instead of silently baked in. Reporting both turns
    # "which activity is broken?" into a number the reader can check.
    comparison: pd.DataFrame | None = None
    summary_all: pd.DataFrame | None = None
    fits_all: dict[str, dict[str, dict[str, float]]] | None = None
    if coverage is not None and not args.no_unfiltered and len(kept) < len(anchors):
        print("running the unfiltered comparison pass ...")
        panel_all = build_panel(
            tables,
            users,
            streams,
            anchors,
            args.horizon_days,
            args.max_delta,
            verbose=False,
        )
        curve_all = smoothed_curve(panel_all, args.smooth_alpha)
        summary_all, fits_all = summarise(
            panel_all, curve_all, args.min_support, args.smooth_alpha
        )
        rows: list[dict[str, Any]] = []
        for _, r in summary.iterrows():
            act = str(r["activity"])
            other = cast(pd.DataFrame, summary_all).set_index("activity").loc[act]
            rows.append(
                {
                    "activity": act,
                    "prevalence_all": float(other["prevalence_pooled"]),
                    "prevalence_clean": float(r["prevalence_pooled"]),
                    "tau_all": float(fits_all[act]["same"]["tau_days"]),
                    "tau_clean": float(fits[act]["same"]["tau_days"]),
                    "ratio_all": float(fits_all[act]["same"]["decay_ratio_1_7"]),
                    "ratio_clean": float(fits[act]["same"]["decay_ratio_1_7"]),
                    "pos_weight_all": float(other["pos_weight"]),
                    "pos_weight_clean": float(r["pos_weight"]),
                }
            )
        comparison = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve = curve.assign(quality="clean" if coverage else "all")
    written_curve = curve
    written_summary = summary
    if summary_all is not None and coverage is not None:
        alt_curve = smoothed_curve(
            build_panel(
                tables,
                users,
                streams,
                anchors,
                args.horizon_days,
                args.max_delta,
                verbose=False,
            ),
            args.smooth_alpha,
        ).assign(quality="all")
        written_curve = pd.concat([curve, alt_curve], ignore_index=True)
        written_summary = pd.concat(
            [summary.assign(quality="clean"), summary_all.assign(quality="all")],
            ignore_index=True,
        )
    written_curve.to_csv(out / "decay_curves.csv", index=False)
    written_summary.to_csv(out / "activity_summary.csv", index=False)
    official_rows.to_csv(out / "official_window_weights.csv", index=False)
    if clean_rows is not None:
        clean_rows.to_csv(out / "official_window_weights_clean.csv", index=False)
    if comparison is not None:
        comparison.to_csv(out / "clean_vs_unfiltered.csv", index=False)

    config = {
        "db": args.db,
        "data_window": [data_start.isoformat(), data_end.isoformat()],
        "horizon_days": args.horizon_days,
        "max_delta": args.max_delta,
        "anchor_step": args.anchor_step,
        "n_anchors": len(panel.windows),
        "anchors": [w["anchor"] for w in panel.windows],
        "min_support": args.min_support,
        "smooth_alpha": args.smooth_alpha,
        "fallback_tau": args.fallback_tau,
        "n_cohort_users": len(user_ids),
        "official_anchor": official.isoformat(),
        "clean_days": args.clean_days,
        "min_hour_volume_share": args.min_hour_volume_share,
        "min_hour_push_share": args.min_hour_push_share,
        "coverage": coverage,
        "support_note": (
            "dt bins beyond (last_anchor - data_start) days have no support: a "
            f"{args.horizon_days}-day label needs anchors before the last row, which "
            "caps observable dt in this dataset"
        ),
    }
    edge_tau: dict[str, float] = {}
    pos_weights: dict[str, float] = {}
    for _, r in summary.iterrows():
        act = str(r["activity"])
        tau_same = fits[act]["same"]["tau_days"]
        edge_tau[act] = (
            float(tau_same) if np.isfinite(tau_same) else float(args.fallback_tau)
        )
        pos_weights[act] = float(r["pos_weight"])

    payload = {
        "config": config,
        "activity_summary": summary.to_dict(orient="records"),
        "activity_summary_unfiltered": (
            summary_all.to_dict(orient="records") if summary_all is not None else None
        ),
        "clean_vs_unfiltered": (
            comparison.to_dict(orient="records") if comparison is not None else None
        ),
        "events": events.to_dict(orient="records"),
        "curves": {act: fits[act] for act in LABEL_COLS},
        "temporal_edge_tau_days": edge_tau,
        "pos_weights": pos_weights,
        "official_window_clean": (
            None
            if clean_rows is None
            else {
                "anchors": [f"{a:%Y-%m-%d}" for a in kept],
                "users": len(users),
                "weights": clean_rows.to_dict(orient="records"),
            }
        ),
        "official_window": {
            "anchor": official.isoformat(),
            "horizon_days": args.horizon_days,
            "users": len(official_all),
            "weights": official_rows.to_dict(orient="records"),
        },
    }
    (out / "activity_summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n"
    )
    write_markdown(
        summary,
        events,
        fits,
        panel,
        config,
        official_rows,
        out / "summary.md",
        comparison=comparison,
        coverage=coverage,
        clean_official=clean_rows,
        fits_unfiltered=fits_all,
    )

    if not args.no_plots:
        plot_panels(
            curve, fits, "any", panel, out / "01_decay_any.png", args.min_support
        )
        plot_panels(
            curve, fits, "same", panel, out / "02_decay_same.png", args.min_support
        )
        plot_overlay(curve, panel, out / "03_overlay.png", args.min_support)
        plot_distribution(
            panel, curve, out / "04_dt_distribution.png", args.min_support
        )
        plot_rarity(summary, events, out / "05_class_rarity.png")
        if comparison is not None and written_curve["quality"].nunique() > 1:
            plot_clean_vs_all(
                written_curve,
                comparison,
                out / "06_clean_vs_all.png",
                args.min_support,
            )

    ev = events.set_index("activity")
    print("\nclass rarity and loss weights")
    print(
        f"  {'activity':>18} {'prevalence':>11} {'pos_weight':>11} "
        f"{'pos/window':>11} {'raw events':>11}"
    )
    for _, r in summary.iterrows():
        act = str(r["activity"])
        print(
            f"  {act:>18} {r['prevalence_pooled'] * 100:>10.2f}% "
            f"{r['pos_weight']:>11.2f} {r['positives_per_window_mean']:>11.1f} "
            f"{int(ev.loc[act, 'events']):>11d}"
        )

    print("\ndecay shape (axis=same: days since the last event of that type)")
    print(
        f"  {'activity':>18} {'shape':>24} {'tau(d)':>8} {'half-life':>10} "
        f"{'p(1)/p(7)':>10} {'bump@7':>8} {'period':>7}"
    )
    for _, r in summary.iterrows():
        f = fits[str(r["activity"])]["same"]
        print(
            f"  {r['activity']:>18} {r['shape_same']:>24} "
            f"{_num(f['tau_days'], '.2f'):>8} "
            f"{_num(f['half_life_days'], '.2f'):>10} "
            f"{_num(f['decay_ratio_1_7'], '.2f'):>10} "
            f"{_num(f['bump_7'], '+.2f'):>8} {_period_cell(f):>7}"
        )

    print("\ndecay shape (axis=any: days since the user's last event)")
    print(
        f"  {'activity':>18} {'shape':>24} {'tau(d)':>8} {'half-life':>10} "
        f"{'p(1)/p(7)':>10} {'bump@7':>8}"
    )
    for _, r in summary.iterrows():
        f = fits[str(r["activity"])]["any"]
        print(
            f"  {r['activity']:>18} {r['shape_any']:>24} "
            f"{_num(f['tau_days'], '.2f'):>8} "
            f"{_num(f['half_life_days'], '.2f'):>10} "
            f"{_num(f['decay_ratio_1_7'], '.2f'):>10} "
            f"{_num(f['bump_7'], '+.2f'):>8}"
        )

    print(
        f"\nwrote {out}/decay_curves.csv, activity_summary.{{csv,json}}, summary.md"
        + ("" if args.no_plots else " and 5 PNGs")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
