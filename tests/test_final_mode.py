"""Tests for the submission run: mode dispatch, anchor list, observability, theta.

Everything here is CPU-only and finishes in milliseconds; the training path
itself is covered by the rehearsal documented in ``docs/FINAL_RUN.md``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.data.label_coverage import count_sanity_flags, observed_activities
from src.data.temporal_split import LABEL_COLS
from src.utils.threshold_optimizer import optimal_thresholds_per_activity


def _ts(value: str) -> pd.Timestamp:
    """`pd.Timestamp` narrowed for the type stubs (they include NaTType)."""
    return cast(pd.Timestamp, pd.Timestamp(value))


class TestModeDispatch:
    def test_bare_command_is_the_submission_run(self) -> None:
        from scripts.run_training import parse_args, resolve_mode

        assert resolve_mode(parse_args([])) == "final"
        assert resolve_mode(parse_args(["--final"])) == "final"

    def test_validate_mode_survives(self) -> None:
        from scripts.run_training import parse_args, resolve_mode

        assert resolve_mode(parse_args(["--mode", "validate"])) == "validate"

    def test_observability_default_follows_the_mode(self) -> None:
        from scripts.run_training import parse_args, resolve_anchor_observed

        # Final runs need the hybrid rule: the audit alone calls 2026-10/11
        # unusable for every activity, which is where the test week lives.
        assert resolve_anchor_observed(parse_args([])) == "hybrid"
        assert resolve_anchor_observed(parse_args(["--mode", "validate"])) == "label"
        assert (
            resolve_anchor_observed(parse_args(["--anchor-observed", "label"])) == "label"
        )

    def test_test_cutoff_is_the_graded_week_start(self) -> None:
        from scripts.run_training import parse_args

        # README section 4: 2026-11-22 00:00 Taiwan time == 2026-11-21 16:00 UTC.
        assert pd.Timestamp(parse_args([]).test_cutoff) == pd.Timestamp(
            "2026-11-21 16:00:00"
        )


class TestFinalAnchorList:
    def test_weekly_then_strided(self) -> None:
        from scripts.run_training import final_anchor_cutoffs

        cutoffs = final_anchor_cutoffs(
            start=_ts("2025-12-22"),
            dense_end=_ts("2026-02-09"),
            end=_ts("2026-11-14"),
            stride=28,
        )
        # weekly dense block, inclusive of both ends
        assert cutoffs[:8] == list(pd.date_range("2025-12-22", "2026-02-09", freq="7D"))
        # then a 28-day grid, and the last legal anchor is always included
        assert cutoffs[8:] == [
            pd.Timestamp(d)
            for d in (
                "2026-03-09",
                "2026-04-06",
                "2026-05-04",
                "2026-06-01",
                "2026-06-29",
                "2026-07-27",
                "2026-08-24",
                "2026-09-21",
                "2026-10-19",
                "2026-11-14",
            )
        ]
        assert cutoffs[-1] == _ts("2026-11-14")
        assert cutoffs == sorted(cutoffs)

    def test_end_before_dense_end_stays_weekly(self) -> None:
        from scripts.run_training import final_anchor_cutoffs

        cutoffs = final_anchor_cutoffs(
            start=_ts("2025-12-22"),
            dense_end=_ts("2026-02-09"),
            end=_ts("2026-01-26"),
            stride=28,
        )
        assert cutoffs == list(pd.date_range("2025-12-22", "2026-01-26", freq="7D"))

    def test_rejects_backwards_range(self) -> None:
        from scripts.run_training import final_anchor_cutoffs

        with pytest.raises(SystemExit):
            final_anchor_cutoffs(
                start=_ts("2026-02-09"),
                dense_end=_ts("2026-01-01"),
                end=_ts("2026-01-01"),
                stride=28,
            )


class TestCountSanityFlags:
    def test_flags_a_collapsed_stream(self) -> None:
        # 10 anchors with a steady stream, then the last one loses 80% of it.
        positives = np.full((10, 2), 1000)
        positives[-1, 0] = 50  # collapsed
        positives[-1, 1] = 900  # intact
        flags = count_sanity_flags(positives)
        assert not flags[-1, 0]
        assert flags[-1, 1]
        assert flags[:-1].all()

    def test_zero_positives_is_never_sane(self) -> None:
        positives = np.full((6, 2), 1000)
        positives[2, 0] = 0
        flags = count_sanity_flags(positives)
        assert not flags[2, 0]
        assert flags[2, 1]

    def test_flags_a_redistribution_dump(self) -> None:
        positives = np.full((8, 1), 500)
        positives[4, 0] = 2000  # 4x the neighbourhood: not a real surge
        flags = count_sanity_flags(positives)
        assert not flags[4, 0]

    def test_november_2026_shape(self) -> None:
        """The live case on the production anchor grid: push dies, PR cells live.

        Positive counts measured from ``processed_tables/db`` for the 28-day
        anchors the submission run uses (``final_anchor_cutoffs``).  The point of
        the test is that the *ratio* test catches the push collapse while leaving
        the four pull-request activities alone, and that it needs the anchor
        spacing of the real grid to do it: within a 7-day grid the four
        neighbours of 2026-11-14 are all post-collapse weeks and the median
        follows them down.
        """
        rows = [
            # open_pr, open_issue, issue_comment, pr_review, prc, push
            (1288, 550, 934, 864, 606, 3828),  # 2026-03-09
            (1211, 573, 1041, 962, 671, 5037),  # 2026-04-06
            (1496, 643, 843, 679, 463, 5175),  # 2026-05-04
            (866, 399, 511, 387, 239, 5155),  # 2026-06-01
            (435, 204, 270, 249, 146, 3999),  # 2026-06-29
            (95, 99, 79, 63, 42, 2725),  # 2026-07-27
            (175, 177, 131, 110, 66, 1399),  # 2026-08-24
            (455, 269, 343, 317, 249, 2581),  # 2026-09-21
            (306, 221, 181, 217, 162, 2058),  # 2026-10-19
            (356, 210, 214, 243, 77, 457),  # 2026-11-14
        ]
        flags = count_sanity_flags(np.array(rows, dtype=float))
        last = dict(zip(LABEL_COLS, flags[-1], strict=True))
        assert last["open_pr"] and last["open_issue"]
        assert last["issue_comment"] and last["pr_review"]
        assert not last["push"], "November push is ~1/5 of October: unobservable"
        # `pr_review_comment` sits at 77/114 = 0.68x its neighbourhood: inside the
        # band, so it stays.  That is intentional -- the test-week labels come
        # from the same degraded feed, so matching its under-count is the
        # consistent choice rather than a bias to correct.
        assert last["pr_review_comment"]


class TestHybridObservability:
    def _day_flags(self) -> pd.DataFrame:
        days = pd.date_range("2026-01-01", "2026-02-28", freq="D")
        frame = pd.DataFrame(index=pd.DatetimeIndex(days))
        for act in LABEL_COLS:
            frame[f"usable_{act}"] = True
        # The audit calls 2026-02-10..28 unusable for everything: this stands in
        # for the 2026-10/11 density artifact.
        frame.loc["2026-02-10":, :] = False
        return frame

    def test_hybrid_rescues_count_sane_cells(self) -> None:
        cutoffs = [_ts("2026-02-02"), _ts("2026-02-16")]
        positives = np.array([[100] * 6, [100] * 6], dtype=float)
        audit = observed_activities(cutoffs, self._day_flags(), days=7, mode="label")
        hybrid = observed_activities(
            cutoffs, self._day_flags(), days=7, mode="hybrid", positives=positives
        )
        assert not audit.loc[_ts("2026-02-16")].any()
        assert hybrid.loc[_ts("2026-02-16")].all()

    def test_hybrid_still_drops_collapsed_cells(self) -> None:
        cutoffs = list(pd.date_range("2026-02-02", periods=5, freq="7D"))
        positives = np.full((5, 6), 100.0)
        positives[-1, 5] = 5  # push collapsed on the last anchor only
        hybrid = observed_activities(
            cutoffs, self._day_flags(), days=7, mode="hybrid", positives=positives
        )
        row = hybrid.loc[cutoffs[-1]]
        assert row["open_pr"]
        assert not row["push"]

    def test_hybrid_needs_positives(self) -> None:
        with pytest.raises(ValueError):
            observed_activities(
                [_ts("2026-02-02")], self._day_flags(), mode="hybrid"
            )


class TestPerActivityThresholds:
    """`optimal_thresholds_per_activity` is the final run's theta fitter.

    It replaces the joint CMA-ES objective for that one mode because the late
    anchors have unobserved columns, which the TA score cannot score at all.
    """

    EVENTS = ("open_pr", "open_issue")

    def test_separates_a_clean_signal(self) -> None:
        labels = np.array([[1, 0], [1, 0], [0, 0], [0, 0]], dtype=np.int64)
        probs = np.array([[0.9, 0.2], [0.8, 0.1], [0.2, 0.3], [0.1, 0.25]])
        theta, metrics = optimal_thresholds_per_activity(probs, labels, events=self.EVENTS)
        assert 0.2 < theta[0] < 0.8
        assert metrics["f1"]["open_pr"] == pytest.approx(1.0)
        # the second activity has no positives: never predict it
        assert theta[1] == pytest.approx(0.99)
        assert metrics["base_rate"]["open_issue"] == 0.0

    def test_ignores_unobserved_cells(self) -> None:
        """A masked cell must not be read as a negative."""
        probs = np.array([[0.9], [0.85], [0.1], [0.05]])
        labels = np.array([[1], [1], [0], [0]], dtype=np.int64)
        observed = np.array([[True], [True], [False], [False]])
        theta_all, _ = optimal_thresholds_per_activity(
            probs, labels, events=("open_pr",)
        )
        theta_masked, metrics = optimal_thresholds_per_activity(
            probs, labels, observed, events=("open_pr",)
        )
        assert metrics["n_rows"]["open_pr"] == 2
        assert metrics["base_rate"]["open_pr"] == pytest.approx(1.0)
        # With only positives observed, every score is a true positive, so the
        # threshold must sit at the lower bound rather than splitting the
        # unobserved zeros.
        assert theta_masked[0] < theta_all[0]

    def test_empty_column_predicts_nothing(self) -> None:
        probs = np.array([[0.5], [0.6]])
        labels = np.array([[0], [1]], dtype=np.int64)
        observed = np.array([[False], [False]])
        theta, metrics = optimal_thresholds_per_activity(
            probs, labels, observed, events=("open_pr",)
        )
        assert theta[0] == pytest.approx(0.99)
        assert metrics["n_rows"]["open_pr"] == 0
