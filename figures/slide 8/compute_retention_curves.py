#!/usr/bin/env python3
"""Grounding probe for slide 8: retention vs idle time on two different clocks.

Slide 8 draws one quantity twice, under two definitions of "idle":

    Curve A  General Activity   - P(the user performs a target role activity in
                                  the next 7 days | dt days since their last
                                  event of ANY of the six activity types);
    Curve B  Role Habit         - the same probability, but dt is measured since
                                  their last event of that SAME role activity.

Both curves are then rescaled by their own dt = 0 value, which is what
"Retention Probability" on the slide means: 1.0 = as likely as a fully active
user, and the axis is the fraction of that activity level still retained.

The measurement is a copy of the pass used for slide 7
(`figures/slide 7/compute_survival_curve.py`), which itself re-implements
`src/data/activity_decay_analysis.py` because that driver crashes on this box
(`src/data/activity_decay_analysis.py:129`, `keep &= ~is_pr` ->
`ValueError: output array is read-only` under this pandas build) and `src/` is
out of bounds for a figure-only agent. The repo's definitions are copied
verbatim from `src/data/temporal_split.py`:

  * users: `Users.is_cohort` only, and only those with `first_seen_at < anchor`;
  * feature side (dt): a row is visible when its ROW_TIME column is < anchor -
    PullRequests / Issues / PRReviews -> `first_seen_at`,
    IssueComments / PRReviewComments / Pushes -> `created_at`;
    Issues rows with `is_pull_request` are dropped (the `open_issue` filter);
  * label side (retained): an event counts when its EVENT time is in
    [anchor, anchor + 7) - `created_at`, except PRReviews -> `submitted_at`;
  * dt bins are whole days 0..MAX_DELTA, pooled over daily anchors;
  * the fitted shape is the repo's `floor + amp * exp(-dt / tau)`.

Writes, next to this file:
  retention_curve_data.csv  axis, target, dt_days, n, positives, p, p_smooth, ci_lo, ci_hi, supported
  retention_fits.json       per-series raw + dt=0-normalized decay fits, plus provenance

Run from the repo root:
    /train-data-5-hdd/charles/miniconda3/envs/bda_asg2/bin/python \
        "figures/slide 8/compute_retention_curves.py"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]  # pyrightconfig.json venvPath is stale on this box; pandas resolves and runs in env bda_asg2
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent  # figures/slide 8
REPO = HERE.parent.parent
DB = REPO / "processed_tables" / "db"
CURVE_CSV = HERE / "retention_curve_data.csv"
FITS_JSON = HERE / "retention_fits.json"

HORIZON_DAYS = 7
MAX_DELTA = 28  # the slide's x-axis limit
SMOOTH_ALPHA = 0.5  # the repo's --smooth-alpha default
MIN_SUPPORT = 20  # the repo's --min-support default
REPORT_DT = (0, 1, 2, 3, 5, 7, 14, 21, 28)  # days quoted in the console table

# (label, table, actor column, event column, row-time column, row filter)
# copied from temporal_split.ACTIVITY_LABELS + temporal_split.ROW_TIME.
ACTIVITY_SPECS: tuple[tuple[str, str, str, str, str, str | None], ...] = (
    ("open_pr", "PullRequests", "author_id", "created_at", "first_seen_at", None),
    ("open_issue", "Issues", "author_id", "created_at", "first_seen_at", "not_pull_request"),
    ("issue_comment", "IssueComments", "author_id", "created_at", "created_at", None),
    ("pr_review", "PRReviews", "reviewer_id", "submitted_at", "first_seen_at", None),
    ("pr_review_comment", "PRReviewComments", "author_id", "created_at", "created_at", None),
    ("push", "Pushes", "user_id", "created_at", "created_at", None),
)
LABELS: tuple[str, ...] = tuple(spec[0] for spec in ACTIVITY_SPECS)
TARGETS: tuple[str, ...] = ("any", *LABELS)


def cohort_users(db: Path) -> tuple[np.ndarray, np.ndarray]:
    """Sorted cohort user ids and their visibility time, aligned to those ids."""
    users = pd.read_parquet(
        db / "Users.parquet", columns=["user_id", "is_cohort", "first_seen_at"]
    )
    keep = users["is_cohort"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    frame = users[keep]
    ids = np.sort(frame["user_id"].astype("int64").to_numpy())
    order = np.argsort(frame["user_id"].to_numpy(), kind="stable")
    seen = pd.to_datetime(frame["first_seen_at"]).to_numpy()[order]
    return ids, seen


def per_user_stream(
    db: Path, ids: np.ndarray, table: str | None, side: str
) -> list[np.ndarray]:
    """Sorted timestamps per cohort user for one table, or for all six unioned.

    `table` is a parquet stem, or None for the union of the six activity streams
    (the "any" clock). `side` picks the column: "row" = ROW_TIME (history the
    model could see), "event" = EVENT time (what the retention window asks about).
    """
    if side not in ("row", "event"):
        msg = f"side must be 'row' or 'event', got {side!r}"
        raise ValueError(msg)
    specs = ACTIVITY_SPECS if table is None else tuple(s for s in ACTIVITY_SPECS if s[1] == table)
    if not specs:
        msg = f"no activity spec for table {table!r}"
        raise ValueError(msg)
    codes: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for _label, stem, user_col, event_col, row_col, row_filter in specs:
        column = row_col if side == "row" else event_col
        need = [user_col, column]
        if row_filter == "not_pull_request" and "is_pull_request" not in need:
            need.append("is_pull_request")
        frame = pd.read_parquet(db / f"{stem}.parquet", columns=need)
        keep = frame[user_col].notna().to_numpy()
        if "is_pull_request" in frame.columns:
            # `keep &= ~is_pr` raises on this pandas build (read-only arrow
            # boolean array), so build a fresh array instead.
            is_pr = frame["is_pull_request"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            keep = keep & ~is_pr
        user_ids = frame.loc[keep, user_col].astype("int64").to_numpy()
        stamp = pd.to_datetime(frame.loc[keep, column]).to_numpy(dtype="datetime64[ns]")
        code = np.searchsorted(ids, user_ids)
        hit = (code < ids.size) & (ids[np.clip(code, 0, ids.size - 1)] == user_ids)
        codes.append(code[hit])
        times.append(stamp[hit])
    code_all = np.concatenate(codes)
    time_all = np.concatenate(times)
    order = np.lexsort((time_all, code_all))
    code_all, time_all = code_all[order], time_all[order]
    starts = np.concatenate(([0], np.cumsum(np.bincount(code_all, minlength=ids.size))))
    return [time_all[starts[i] : starts[i + 1]] for i in range(ids.size)]


def wilson(pos: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval, as used by the assignment's decay analysis."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = pos / n
        denom = 1.0 + z * z / n
        centre = (ph + z * z / (2 * n)) / denom
        half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def accumulate(
    history: list[np.ndarray],
    events: list[np.ndarray],
    seen: np.ndarray,
    stamp_anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled per-dt counts: n[dt], positives[dt] over every (user, anchor) pair."""
    n_bins = MAX_DELTA + 1
    n_by_dt = np.zeros((len(stamp_anchors), n_bins), dtype=np.int64)
    pos_by_dt = np.zeros((len(stamp_anchors), n_bins), dtype=np.int64)
    horizon = np.timedelta64(HORIZON_DAYS, "D")

    for user_idx, hist in enumerate(history):
        visible = seen[user_idx] < stamp_anchors
        if not visible.any() or hist.size == 0:
            continue
        counts = np.searchsorted(hist, stamp_anchors, side="right")
        usable = visible & (counts > 0)
        if not usable.any():
            continue
        last = hist[counts - 1]
        dt = (stamp_anchors - last) / np.timedelta64(1, "D")
        in_range = usable & (dt >= 0) & (dt <= MAX_DELTA)
        if not in_range.any():
            continue
        ev = events[user_idx]
        retained = np.zeros(len(stamp_anchors), dtype=bool)
        if ev.size:
            lo = np.searchsorted(ev, stamp_anchors, side="left")
            hi = np.searchsorted(ev, stamp_anchors + horizon, side="left")
            retained = hi > lo
        rows = np.flatnonzero(in_range)
        bins = np.floor(dt[rows]).astype(np.int64)
        np.add.at(n_by_dt, (rows, bins), 1)
        np.add.at(pos_by_dt, (rows, bins), retained[rows].astype(np.int64))

    return n_by_dt.sum(axis=0), pos_by_dt.sum(axis=0)


def fit_decay(dt: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """floor + amp * exp(-dt / tau), the shape the assignment fits to this curve."""

    def model(x: np.ndarray, floor: float, amp: float, tau: float) -> np.ndarray:
        return floor + amp * np.exp(-x / tau)

    out = {
        "floor": float("nan"),
        "amp": float("nan"),
        "tau_days": float("nan"),
        "half_life_days": float("nan"),
        "r2": float("nan"),
    }
    try:
        popt, _ = curve_fit(
            model,
            dt,
            p,
            p0=[0.10, 0.60, 3.0],
            bounds=([0.0, 0.0, 0.2], [1.0, 2.0, 120.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return out
    floor, amp, tau = (float(v) for v in popt)
    resid = p - model(dt, floor, amp, tau)
    ss_tot = float(np.sum((p - p.mean()) ** 2))
    out.update(
        floor=floor,
        amp=amp,
        tau_days=tau,
        half_life_days=float(np.log(2) * tau),
        r2=float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan"),
    )
    return out


def normalized_view(decay: dict[str, float]) -> dict[str, Any]:
    """The same fit divided by its own dt = 0 value: retention relative to active.

    Dividing by (floor + amp) - the model's value at dt = 0 - makes amp + floor
    exactly 1, so a curve starts at 1.0 and decays to floor / (floor + amp). R^2
    is unchanged by that affine rescaling.
    """
    out: dict[str, Any] = {
        "floor": float("nan"),
        "amp": float("nan"),
        "tau_days": decay.get("tau_days", float("nan")),
        "half_life_days": decay.get("half_life_days", float("nan")),
        "r2": decay.get("r2", float("nan")),
        "raw_at_zero": float("nan"),
        "retention_at": {},
    }
    floor, amp, tau = decay.get("floor"), decay.get("amp"), decay.get("tau_days")
    if floor is None or amp is None or tau is None:
        return out
    if not (np.isfinite(floor) and np.isfinite(amp) and np.isfinite(tau)):
        return out
    base = floor + amp
    if base <= 0:
        return out
    norm_floor = floor / base
    out.update(
        floor=norm_floor,
        amp=1.0 - norm_floor,
        raw_at_zero=base,
        retention_at={
            str(day): norm_floor + (1.0 - norm_floor) * float(np.exp(-day / tau))
            for day in REPORT_DT
        },
    )
    return out


def summarize_series(
    axis: str, target: str, n_by_dt: np.ndarray, pos_by_dt: np.ndarray
) -> tuple[dict[str, Any], pd.DataFrame]:
    """One row block of the tidy curve plus the fits for that series."""
    dt_all = np.arange(n_by_dt.size, dtype=float)
    supported = n_by_dt >= MIN_SUPPORT
    lo, hi = wilson(pos_by_dt, np.maximum(n_by_dt, 1))
    frame = pd.DataFrame(
        {
            "axis": axis,
            "target": target,
            "dt_days": np.arange(n_by_dt.size, dtype=int),
            "n": n_by_dt,
            "positives": pos_by_dt,
            "p": np.where(n_by_dt > 0, pos_by_dt / np.maximum(n_by_dt, 1), np.nan),
            "p_smooth": np.where(
                n_by_dt > 0,
                (pos_by_dt + SMOOTH_ALPHA) / (np.maximum(n_by_dt, 1) + 2 * SMOOTH_ALPHA),
                np.nan,
            ),
            "ci_lo": lo,
            "ci_hi": hi,
            "supported": supported,
        }
    )
    dt_ok = dt_all[supported]
    p_ok = frame.loc[supported, "p_smooth"].to_numpy(dtype=float)
    fits: dict[str, Any] = {
        "axis": axis,
        "target": target,
        "supported_bins": int(supported.sum()),
        "max_supported_dt": int(dt_ok[-1]) if dt_ok.size else -1,
        "total_pairs": int(n_by_dt.sum()),
        "p_dt0": float(p_ok[0]) if p_ok.size else float("nan"),
        "p_dt7": float(p_ok[np.searchsorted(dt_ok, 7.0)]) if dt_ok.size else float("nan"),
        "tail_mean": float(np.mean(p_ok[dt_ok >= 21])) if np.any(dt_ok >= 21) else float("nan"),
    }
    if dt_ok.size >= 8:
        decay = fit_decay(dt_ok, p_ok)
        fits["decay"] = decay
        fits["retention"] = normalized_view(decay)
    return fits, frame


def fmt(value: object, width: int, prec: int) -> str:
    """Right-aligned number, blank when the fit did not produce one."""
    if isinstance(value, float) and np.isfinite(value):
        return f"{value:{width}.{prec}f}"
    return " " * (width + 1)


def main() -> int:
    ids, seen = cohort_users(DB)
    latest = max(
        pd.to_datetime(pd.read_parquet(DB / f"{stem}.parquet", columns=[event_col])[event_col]).max()
        for _label, stem, _user_col, event_col, _row_col, _filt in ACTIVITY_SPECS
    )
    first_seen_min = pd.to_datetime(
        pd.read_parquet(DB / "Users.parquet", columns=["first_seen_at"])["first_seen_at"]
    ).min()
    # an anchor needs `horizon` days of forward labels and `MAX_DELTA` days of
    # backward history, or the dt bins at the ends have no support at all.
    first = max(first_seen_min + pd.Timedelta(days=MAX_DELTA), pd.Timestamp("2026-01-01"))
    last = latest - pd.Timedelta(days=HORIZON_DAYS)
    anchor_index = pd.date_range(first.normalize(), last.normalize(), freq="D")
    if len(anchor_index) < 8:
        print(f"anchor window too short: {first} .. {last}")
        return 1
    stamp_anchors = anchor_index.to_numpy(dtype="datetime64[ns]")

    history_any = per_user_stream(DB, ids, None, side="row")
    events: dict[str, list[np.ndarray]] = {"any": per_user_stream(DB, ids, None, side="event")}
    history_same: dict[str, list[np.ndarray]] = {}
    for label, stem, *_rest in ACTIVITY_SPECS:
        events[label] = per_user_stream(DB, ids, stem, side="event")
        history_same[label] = per_user_stream(DB, ids, stem, side="row")

    all_fits: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for target in TARGETS:
        n_dt, pos_dt = accumulate(history_any, events[target], seen, stamp_anchors)
        fits, frame = summarize_series("any", target, n_dt, pos_dt)
        all_fits.append(fits)
        frames.append(frame)
    for label, *_rest in ACTIVITY_SPECS:
        n_dt, pos_dt = accumulate(history_same[label], events[label], seen, stamp_anchors)
        fits, frame = summarize_series("same", label, n_dt, pos_dt)
        all_fits.append(fits)
        frames.append(frame)

    curve = pd.concat(frames, ignore_index=True)
    curve.to_csv(CURVE_CSV, index=False)

    provenance = {
        "anchors": {
            "first": str(anchor_index[0].date()),
            "last": str(anchor_index[-1].date()),
            "count": int(len(anchor_index)),
            "horizon_days": HORIZON_DAYS,
            "max_delta_days": MAX_DELTA,
            "min_support": MIN_SUPPORT,
            "smooth_alpha": SMOOTH_ALPHA,
            "cohort_users": int(ids.size),
            "corpus_latest_event": str(latest),
        },
        "normalization": "fit divided by its own dt = 0 value (amp + floor = 1)",
        "series": all_fits,
    }
    FITS_JSON.write_text(json.dumps(provenance, indent=2))

    print(f"cohort users {ids.size}, anchors {anchor_index[0].date()}..{anchor_index[-1].date()} "
          f"({len(anchor_index)} daily)")
    header = (f"{'axis':<5} {'target':<18} {'pairs':>8} {'maxdt':>5} {'raw p(0)':>9} "
              f"{'tau d':>7} {'half d':>7} {'R2 dec':>7} {'floor%':>7}")
    print("\n" + header)
    for fits in all_fits:
        decay = fits.get("decay", {})
        ret = fits.get("retention", {})
        print(f"{fits['axis']:<5} {fits['target']:<18} {fits['total_pairs']:>8} "
              f"{fits['max_supported_dt']:>5} {fmt(fits['p_dt0'], 9, 3)} "
              f"{fmt(decay.get('tau_days', float('nan')), 7, 2)} "
              f"{fmt(decay.get('half_life_days', float('nan')), 7, 2)} "
              f"{fmt(decay.get('r2', float('nan')), 7, 4)} "
              f"{fmt(ret.get('floor', float('nan')) * 100.0, 7, 1)}")

    print("\nnormalized retention  R(dt) / R(0)  from the fitted decay")
    print(f"{'axis':<5} {'target':<18} " + " ".join(f"{'d' + str(d):>6}" for d in REPORT_DT))
    for fits in all_fits:
        ret = fits.get("retention", {}).get("retention_at", {})
        if not ret:
            continue
        cells = " ".join(f"{fmt(ret.get(str(d), float('nan')), 6, 3)}" for d in REPORT_DT)
        print(f"{fits['axis']:<5} {fits['target']:<18} {cells}")
    print(f"\nwrote {CURVE_CSV.name} and {FITS_JSON.name} in {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
