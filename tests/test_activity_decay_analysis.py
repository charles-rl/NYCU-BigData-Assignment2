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
  6. `periodicity` finds a 7-day resonance when one is planted, refuses to
     score bins too thin to support a cycle claim, and reports no cycle for a
     clean exponential or for sampling noise alone.
  7. `smoothing` is a shrunken estimate, strictly inside (0, 1) for a bin with
     zero positives.

Run from the repository root:
    python -m pytest tests/test_activity_decay_analysis.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.data.activity_decay_analysis import (
    AXES,
    NEVER,
    PERIOD_R2_GATE,
    Panel,
    _runs,
    anchor_usability,
    bin_index,
    bin_labels,
    day_quality,
    fit_decay,
    load_coverage,
    periodicity,
    select_anchors,
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
    assert per["cycle_r2"] > PERIOD_R2_GATE
    assert per["cycle_amplitude_share"] == pytest.approx(0.20, abs=0.05)
    # A smooth cycle has no spike, so the bump ratio must stay ~flat.
    assert abs(per["bump_7"]) < 0.05


def test_periodicity_demands_support_for_a_cycle_claim() -> None:
    """A thin panel must not be cycle-tested: the scan overfits at low n."""
    dt = np.arange(0, 25, dtype=float)
    p = 0.10 * (1.0 + 0.50 * np.sin(2 * np.pi * dt / 7.0))

    thick = periodicity(dt, p, np.full(dt.shape, 5_000), min_support=20, max_delta=24)
    thin = periodicity(dt, p, np.full(dt.shape, 100), min_support=20, max_delta=24)

    assert np.isfinite(thick["dominant_period_days"])
    assert np.isnan(thin["dominant_period_days"])
    assert np.isnan(thin["cycle_r2"]), "unsupported bins must not be scored at all"


def test_periodicity_resists_sampling_noise() -> None:
    """Poisson noise alone must not fake a weekly cycle.

    Regression guard: the previous periodogram detector called a bogus 11.3-day
    cycle on curves whose only structure was decay.
    """
    dt, p, n = _synthetic(tau=4.0, floor=0.05, amp=0.40)
    rng = np.random.default_rng(0)
    found = 0
    for _ in range(20):
        count = rng.binomial(n.astype(int), np.clip(p, 0, 1))
        per = periodicity(dt, count / n, n, min_support=20, max_delta=24)
        found += int(np.isfinite(per["dominant_period_days"]))
    assert found <= 1, f"noise produced {found}/20 cycles"


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


def _coverage_frame() -> pd.DataFrame:
    """Sixteen audited days; only `2026-10-02` loses its PushEvents."""
    rows = []
    for day in pd.date_range("2026-09-20", "2026-10-05"):
        for hour in range(24):
            lost = day == pd.Timestamp("2026-10-02") and hour == 12
            rows.append(
                {
                    "file": f"{day:%Y-%m-%d}-{hour}.json.gz",
                    "day": f"{day:%Y-%m-%d}",
                    "hour": hour,
                    "events": 50_000,
                    "PushEvent": 0 if lost else 35_000,
                }
            )
    return pd.DataFrame(rows)


def _activity_coverage_frame() -> pd.DataFrame:
    """Same days, plus per-activity flags where the losses differ by type.

    `2026-09-30` keeps every push but loses its reviews in one hour, which is
    the case a day-level verdict cannot express.
    """
    frame = _coverage_frame()
    review_lost = (frame["day"] == "2026-09-30") & (frame["hour"] == 3)
    for act in LABEL_COLS:
        frame[f"ok_{act}"] = True
    frame.loc[frame["day"] == "2026-10-02", "ok_push"] = False
    frame.loc[frame["day"] == "2026-10-02", "ok_pr_review_comment"] = False
    frame.loc[review_lost, "ok_pr_review"] = False
    frame.loc[review_lost, "ok_issue_comment"] = False
    return frame


def _days(*spec: str) -> list[pd.Timestamp]:
    return list(pd.to_datetime(list(spec)))


def test_day_quality_needs_every_hour_clean() -> None:
    per_day = day_quality(_coverage_frame(), 0.25, 0.30)

    assert bool(per_day.loc["2026-09-28", "clean"]) is True
    assert bool(per_day.loc["2026-10-02", "clean"]) is False
    assert int(per_day.loc["2026-10-02", "hours_ok"]) == 23
    assert float(per_day.loc["2026-09-28", "push_share"]) == pytest.approx(0.7)


def test_day_quality_flags_an_hour_that_is_full_but_wrong() -> None:
    frame = _coverage_frame()
    # A full-sized hour composed only of non-push events is the sneaky case:
    # event count and file size both look healthy, only the mix is wrong.
    broken = (frame["day"] == "2026-09-30") & (frame["hour"] == 3)
    frame.loc[broken, "events"] = 120_000
    frame.loc[broken, "PushEvent"] = 0

    per_day = day_quality(frame, 0.25, 0.30)

    assert bool(per_day.loc["2026-09-30", "clean"]) is False


def test_select_anchors_drops_windows_that_touch_a_bad_day() -> None:
    per_day = day_quality(_coverage_frame(), 0.25, 0.30)
    kept = select_anchors(
        _days("2026-09-24", "2026-09-28", "2026-10-05"),
        per_day,
        "label",
        horizon_days=7,
        max_delta=30,
    )

    assert _days("2026-09-24")[0] in kept, "window 09-24..09-30 is fully clean"
    assert _days("2026-09-28")[0] not in kept, "window 09-28..10-04 hits 10-02"
    assert _days("2026-10-05")[0] not in kept, (
        "window 10-05..10-11 leaves the audit, so those days are unknown"
    )


def test_select_anchors_strict_needs_the_lookback_too() -> None:
    per_day = day_quality(_coverage_frame(), 0.25, 0.30)
    anchors = _days("2026-10-04")

    # Label window 10-04..10-05 is clean, but the 2-day dt lookback reaches
    # back into 10-02, where pushes were lost.
    assert select_anchors(anchors, per_day, "label", 2, 2) == anchors
    assert select_anchors(anchors, per_day, "strict", 2, 2) == []


def test_anchor_usability_is_per_activity_not_per_day() -> None:
    per_day = day_quality(_activity_coverage_frame(), 0.25, 0.30)
    anchors = _days("2026-09-24", "2026-09-28", "2026-10-05")
    usable = anchor_usability(anchors, per_day, "label", 7, 30)

    # 09-24..09-30 loses reviews on 09-30 but keeps pushes.
    assert bool(usable.loc[_days("2026-09-24")[0], "push"]) is True
    assert bool(usable.loc[_days("2026-09-24")[0], "pr_review"]) is False
    # 09-28..10-04 loses both push and pr_review_comment on 10-02.
    assert bool(usable.loc[_days("2026-09-28")[0], "open_pr"]) is True
    assert bool(usable.loc[_days("2026-09-28")[0], "push"]) is False
    # open_pr never lost a stream, so every audited anchor is usable for it.
    assert set(usable.index[usable["open_pr"]]) == {
        _days("2026-09-24")[0],
        _days("2026-09-28")[0],
    }


def test_select_anchors_still_demands_every_activity() -> None:
    per_day = day_quality(_activity_coverage_frame(), 0.25, 0.30)
    anchors = _days("2026-09-24", "2026-09-28")

    # The whole-panel filter is the AND over activities, so losing any one
    # stream is enough to drop the anchor.
    assert select_anchors(anchors, per_day, "label", 7, 30) == []


def test_panel_add_skips_unusable_activities() -> None:
    panel = Panel(max_delta=5, n_act=len(LABEL_COLS))
    n_users = 4
    dt_any = np.full(n_users, 2.0)
    dt_same = np.full((n_users, len(LABEL_COLS)), 2.0)
    y = np.ones((n_users, len(LABEL_COLS)), dtype=bool)
    usable = np.ones(len(LABEL_COLS), dtype=bool)
    usable[LABEL_COLS.index("push")] = False

    panel.add(
        dt_any, dt_same, y, cast("pd.Timestamp", pd.Timestamp("2026-10-10")), usable
    )

    j_push = LABEL_COLS.index("push")
    assert panel.n[:, j_push].sum() == 0, "a damaged stream contributes no rows"
    assert panel.pos[:, j_push].sum() == 0
    assert panel.n[AXES.index("any"), 0, 2] == n_users, "other activities survive"
    # Recorded as unobserved, not as a zero-positive window.
    assert np.isnan(panel.windows[0]["push"])
    assert panel.windows[0]["open_pr"] == n_users


def test_select_anchors_off_mode_keeps_everything() -> None:
    per_day = day_quality(_coverage_frame(), 0.25, 0.30)
    anchors = _days("2026-09-28", "2026-10-05")

    assert select_anchors(anchors, per_day, "off", 7, 30) == anchors


def test_load_coverage_drops_unreadable_files(tmp_path: Path) -> None:
    frame = _coverage_frame()
    frame["error"] = None
    frame.loc[0, "error"] = "BadGzipFile: Not a gzipped file"
    frame.loc[0, ["events", "PushEvent"]] = None
    path = tmp_path / "coverage.csv"
    frame.to_csv(path, index=False)

    loaded = load_coverage(path)

    assert len(loaded) == len(frame) - 1
    assert bool(loaded["events"].notna().all())
    assert str(loaded["day"].iloc[0]) == "2026-09-20"


def test_runs_returns_half_open_intervals() -> None:
    flags = np.array([False, True, True, False, True])
    assert _runs(flags) == [(1, 3), (4, 5)]
    assert _runs(np.zeros(4, dtype=bool)) == []
    assert _runs(np.ones(3, dtype=bool)) == [(0, 3)]
