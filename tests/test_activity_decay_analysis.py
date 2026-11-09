"""Checks for `src.data.activity_decay_analysis`.

What is verified:
  1. `bin_index` maps dt onto half-open day bins, and keeps the two censored
     buckets (`never`, `>max`) out of the numeric bins.
  2. `wilson` agrees with the pooled frequency on a large sample, returns NaN
     where a bin has no rows, and always brackets the point estimate.
  3. `Panel.add` accumulates counts per (axis, activity, bucket), so the
     `same` axis and the `any` axis really are different marginals.
  4. `fit_decay` recovers tau of a synthetic `floor + amp * exp(-dt / tau)`
     curve and refuses to fit when too few bins clear `min_support`.
  5. `shape_label` separates a sharp curve from a flat one.
  6. `periodicity` finds a 7-day resonance when one is planted, and reports
     no cycle when the curve is a clean exponential.
  7. `smoothing` is a shrunken estimate, strictly inside (0, 1) for a bin with
     zero positives.

Run from the repository root:
    python -m pytest tests/test_activity_decay_analysis.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.activity_decay_analysis import (
    AXES,
    NEVER,
    Panel,
    bin_index,
    bin_labels,
    fit_decay,
    periodicity,
    shape_label,
    smoothing,
    wilson,
)
from src.data.temporal_split import LABEL_COLS


def test_bin_index_separates_censored_buckets() -> None:
    dt = np.array([0.0, 0.4, 1.0, 6.99, 7.0, 29.9, 30.0, 30.5, np.nan])
    idx = bin_index(dt, max_delta=30)

    assert idx[0] == 0 and idx[1] == 0, "dt under one day is bin 0"
    assert idx[3] == 6, "6.99 days is still bin 6 (half-open bins)"
    assert idx[4] == 7
    assert idx[7] == 32, "dt past max_delta lands in the >max bucket"
    assert idx[8] == 31, "a censored user has no numeric dt"

    labels = bin_labels(30)
    assert labels[31] == NEVER
    assert len(labels) == 33


def test_wilson_matches_frequency_and_brackets_it() -> None:
    pos = np.array([0, 3, 50, 900])
    n = np.array([10, 10, 100, 1000])
    lo, hi = wilson(pos, n)

    p = pos / n
    assert np.all(lo <= p) and np.all(p <= hi)
    assert np.all(lo < hi), "an interval must be non-degenerate"
    assert hi[3] - lo[3] < hi[0] - lo[0], "more rows must mean a tighter interval"
    assert abs(p[3] - 0.9) < 1e-12

    lo0, hi0 = wilson(np.array([0]), np.array([0]))
    assert np.isnan(lo0[0]) and np.isnan(hi0[0]), "no rows -> no estimate"


def test_panel_accumulates_both_axes() -> None:
    panel = Panel(max_delta=5, n_act=len(LABEL_COLS))
    n_users = 4
    dt_any = np.full(n_users, 2.0)
    dt_same = np.array([[0.0, 9.0, 9.0, 9.0, 9.0, 9.0]] * n_users)
    y = np.ones((n_users, len(LABEL_COLS)), dtype=bool)

    panel.add(dt_any, dt_same, y, anchor=__import__("pandas").Timestamp("2026-10-10"))

    assert panel.n[AXES.index("any"), 0, 2] == n_users
    assert panel.n[AXES.index("any"), 0, 0] == 0
    assert panel.n[AXES.index("same"), 0, 0] == n_users
    assert panel.n[AXES.index("same"), 1, -1] == n_users, "9d > max_delta=5"
    assert panel.pos[AXES.index("any"), 0, 2] == n_users
    assert panel.windows[0]["users"] == n_users


def _synthetic(tau: float, floor: float, amp: float) -> tuple[np.ndarray, ...]:
    dt = np.arange(0, 25, dtype=float)
    p = floor + amp * np.exp(-dt / tau)
    n = np.full(dt.shape, 5_000)
    return dt, p, n


def test_fit_decay_recovers_tau() -> None:
    dt, p, n = _synthetic(tau=4.0, floor=0.05, amp=0.40)
    fit = fit_decay(dt, p, n, min_support=20)

    assert abs(fit["tau_days"] - 4.0) < 0.2
    assert abs(fit["half_life_days"] - np.log(2) * 4.0) < 0.2
    assert abs(fit["floor"] - 0.05) < 0.02
    assert fit["r2"] > 0.99


def test_fit_decay_refuses_thin_data() -> None:
    dt = np.array([0.0, 1.0, 2.0])
    p = np.array([0.4, 0.3, 0.2])
    n = np.array([1, 2, 3])  # every bin is under min_support

    fit = fit_decay(dt, p, n, min_support=100)

    assert np.isnan(fit["tau_days"])
    assert np.isnan(fit["decay_ratio_1_7"])


def test_shape_label_separates_shapes() -> None:
    sharp = fit_decay(*_synthetic(tau=1.8, floor=0.02, amp=0.30), min_support=20)
    flat = fit_decay(*_synthetic(tau=200.0, floor=0.30, amp=0.05), min_support=20)

    assert shape_label(sharp) == "sharp exponential decay"
    assert shape_label(flat) in ("slow decay", "flat / steady")
    assert shape_label({"tau_days": np.nan}) == "no fit"


def test_periodicity_finds_planted_weekly_spikes() -> None:
    dt, p, n = _synthetic(tau=4.0, floor=0.05, amp=0.40)
    spikes = np.isin(dt, [7.0, 14.0, 21.0])
    planted = p * (1.0 + 0.60 * spikes)

    per = periodicity(dt, planted, n, min_support=20, max_delta=24)
    base = periodicity(dt, p, n, min_support=20, max_delta=24)

    # A declining baseline makes bump@7 slightly negative, so the test is
    # relative: the planted spikes must lift the ratio well above the baseline.
    assert per["bump_7"] > base["bump_7"] + 0.30
    assert per["bump_14"] > base["bump_14"] + 0.30


def test_periodicity_finds_a_smooth_weekly_cycle() -> None:
    dt = np.arange(0, 25, dtype=float)
    n = np.full(dt.shape, 5_000)
    p = 0.10 + 0.02 * np.sin(2 * np.pi * dt / 7.0)

    per = periodicity(dt, p, n, min_support=20, max_delta=24)

    assert per["dominant_period_days"] == pytest.approx(7.0, abs=0.5)
    assert per["peak_power_share"] > 0.20
    # A smooth cycle has no spike, so the bump ratio must stay ~flat.
    assert abs(per["bump_7"]) < 0.05


def test_periodicity_reports_no_cycle_for_pure_exponential() -> None:
    dt, p, n = _synthetic(tau=4.0, floor=0.05, amp=0.40)
    per = periodicity(dt, p, n, min_support=20, max_delta=24)

    assert per["bump_7"] < 0.05
    assert np.isnan(per["dominant_period_days"]), "a decay must not fake a cycle"
    assert per["plateau_ratio"] < 1.0, "week 2 must be quieter than week 1"


def test_smoothing_is_shrunken() -> None:
    import pandas as pd

    frame = pd.DataFrame({"positives": [0, 1, 9], "n": [10, 10, 10]})
    out = smoothing(frame, alpha=1.0)

    assert np.all(out > 0) and np.all(out < 1)
    assert out.iloc[0] < 1 / 10, "a zero-count bin must shrink, not inflate"
    assert out.iloc[2] < 9 / 10
