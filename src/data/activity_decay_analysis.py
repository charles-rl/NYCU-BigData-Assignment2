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


def periodicity(
    dt: np.ndarray, p: np.ndarray, n: np.ndarray, min_support: int, max_delta: int
) -> dict[str, float]:
    """Weekly-bump ratios plus the dominant period of the detrended curve.

    bump(k) = p(k) / mean(p(k-3..k-2, k+2..k+3)) - 1 is the excess over the
    local trend at a 7-day multiple.  The periodogram peak is a coarse
    cross-check on a 20-point curve, so both are reported.
    """
    out = {
        "bump_7": float("nan"),
        "bump_14": float("nan"),
        "bump_21": float("nan"),
        "plateau_ratio": float("nan"),
        "dominant_period_days": float("nan"),
        "peak_power_share": float("nan"),
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
    logv = np.log(np.clip(vals.to_numpy(), 1e-6, None))
    coef = np.polyfit(grid, logv, 2)  # quadratic: the tail curves in log space
    resid = logv - np.polyval(coef, grid)
    resid = resid - resid.mean()

    nfft = 4 * len(grid)  # zero-pad: finer period resolution on a short curve
    power = np.abs(np.fft.rfft(resid, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    band = (periods >= 2.0) & (periods <= 12.0)
    if band.any():
        k = int(np.argmax(power[band]))
        total = float(np.sum(power[band] ** 2))
        share = float(power[band][k] ** 2 / total) if total > 0 else float("nan")
        out["peak_power_share"] = share
        # Below 20% of the in-band power the "peak" is just leftover curvature.
        if np.isfinite(share) and share >= 0.20:
            out["dominant_period_days"] = float(periods[band][k])
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
    ) -> None:
        """Fold one anchor's user panel into the counts."""
        for axis, dt_matrix in enumerate(
            (dt_any[:, None].repeat(self.n_act, 1), dt_same)
        ):
            for j in range(self.n_act):
                idx = bin_index(dt_matrix[:, j], self.max_delta)
                self.n[axis, j] += np.bincount(idx, minlength=self.nbins)
                self.pos[axis, j] += np.bincount(
                    idx, weights=y[:, j].astype(float), minlength=self.nbins
                ).astype(np.int64)
        self.windows.append(
            {
                "anchor": anchor.isoformat(),
                "users": int(y.shape[0]),
                **{act: int(y[:, j].sum()) for j, act in enumerate(LABEL_COLS)},
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
            )
        )

        pos_by_window = windows[act].to_numpy() if act in windows else np.array([0])
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
        last_supported = float(x[supported].max()) if supported.any() else 0.0

        ax.axvspan(
            last_supported + 0.5,
            panel.max_delta + 0.5,
            color="0.9",
            zorder=0,
            lw=0,
        )
        ax.bar(
            x[valid],
            n[valid] / max(n[valid].max(), 1) * 0.02,
            color="0.75",
            width=0.85,
            zorder=1,
        )
        ax.fill_between(
            x[valid],
            dt_rows["ci_lo"].to_numpy()[valid],
            dt_rows["ci_hi"].to_numpy()[valid],
            color=col,
            alpha=0.18,
            lw=0,
            zorder=2,
        )
        ax.plot(
            x[valid],
            p[valid],
            "o-",
            color=col,
            ms=3.5,
            lw=1.4,
            zorder=4,
            label="empirical",
        )
        smoothed = smoothing(dt_rows, 0.5).to_numpy()
        keep = valid & supported
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
                "no support\n(data ends 10-31)",
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

    lines.append("\n## How to use these numbers\n")
    lines.append(
        "- `TEMPORAL_EDGE_TAU_DAYS[a]` comes from the **same-type** axis: it is the "
        "decay of P(a) as a function of days since the user's own last `a` event, "
        "so `w = exp(-dt_a / tau_a)` is well scaled per activity. Using the "
        "any-type tau would over-fade rare activities (tau 1.6-2.6 d).\n"
        "- `POS_WEIGHTS[a] = (1-p)/p` is the standard `BCEWithLogitsLoss` weight "
        "if you train on the pooled anchor panel; `sqrt` is the gentler variant "
        "that does not over-optimise macro-F1 on push.\n"
        f"- Weekly resonance: {n_bump}/{2 * len(LABEL_COLS)} curves have "
        f"bump@7 >= +0.15 and {n_cycle}/{2 * len(LABEL_COLS)} keep a spectral "
        f"peak above the 20%-of-power gate, so "
        f"{'periodicity is real for ' + str(n_bump) + ' activities' if n_bump else 'the shape is decay to a floor, not periodicity'}"
        ".\n"
    )
    lines.append("\n## Config snippet\n")
    lines.append("```python\n")
    lines.append("TEMPORAL_EDGE_TAU_DAYS = {  # edge weight = exp(-dt / tau)\n")
    for _, r in summary.iterrows():
        tau = fits[str(r["activity"])]["same"]["tau_days"]
        tau = tau if np.isfinite(tau) else float(config["fallback_tau"])
        lines.append(f"    '{r['activity']}': {tau:.2f},\n")
    lines.append("}\n\nPOS_WEIGHTS = {  # BCEWithLogitsLoss(pos_weight=...)\n")
    for _, r in summary.iterrows():
        lines.append(f"    '{r['activity']}': {r['pos_weight']:.2f},\n")
    lines.append("}\n```\n")
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
    panel = Panel(max_delta=args.max_delta, n_act=len(LABEL_COLS))
    first_seen = users.set_index("user_id")["first_seen_at"]

    for anchor in anchors:
        known = cast(pd.Series, first_seen[first_seen < anchor])
        if known.empty:
            continue
        ids = known.index.to_numpy(dtype="int64")
        frame = users.set_index("user_id").loc[ids].reset_index()
        y = labels_at(tables, anchor, args.horizon_days, frame)
        dt_any = days_since_last(streams["any"], anchor, ids)
        dt_same = np.column_stack(
            [days_since_last(streams[act], anchor, ids) for act in LABEL_COLS]
        )
        panel.add(dt_any, dt_same, y, anchor)
        print(
            f"  {anchor:%Y-%m-%d %H:%M}  users={len(ids):5d}  "
            f"dt_any p50={np.nanmedian(dt_any):5.2f}d  "
            f"censored={int(np.isnan(dt_any).sum()):4d}  "
            + " ".join(
                f"{act}={int(y[:, j].sum())}" for j, act in enumerate(LABEL_COLS)
            )
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

    curve = curve_table(panel)
    curve["p_smooth"] = np.nan
    for axis in AXES:
        for act in LABEL_COLS:
            mask = (curve["axis"] == axis) & (curve["activity"] == act)
            curve.loc[mask, "p_smooth"] = smoothing(
                cast(pd.DataFrame, curve.loc[mask]), args.smooth_alpha
            ).to_numpy()

    summary, fits = summarise(panel, curve, args.min_support, args.smooth_alpha)
    events = event_stats(tables, user_ids)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / "decay_curves.csv", index=False)
    summary.to_csv(out / "activity_summary.csv", index=False)
    official_rows.to_csv(out / "official_window_weights.csv", index=False)

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
        "events": events.to_dict(orient="records"),
        "curves": {act: fits[act] for act in LABEL_COLS},
        "temporal_edge_tau_days": edge_tau,
        "pos_weights": pos_weights,
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
        summary, events, fits, panel, config, official_rows, out / "summary.md"
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
