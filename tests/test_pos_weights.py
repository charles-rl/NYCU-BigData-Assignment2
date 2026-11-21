"""Tests for ``--pos-weights`` resolution, especially the ``auto`` mode.

``auto`` exists because the static ``DOC_POS_WEIGHTS`` come from a 252-anchor
panel that is wider than any single run's training set.  The two properties that
matter, and that a regression would quietly break:

1. the value it returns is exactly ``(N_observed - N_pos) / N_pos`` -- the
   textbook BCE ``pos_weight``, computed where the model is actually fitted;
2. the denominator counts *observed* cells, not rows.  A label cell the coverage
   audit marked unobserved never enters the loss, so counting it as a negative
   would deflate the weight for exactly the activities that lost days.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.ablations.run_training_ablation import (
    DOC_POS_WEIGHTS,
    resolve_pos_weights,
)
from src.utils.metrics import ACTIVITY_NAMES, N_ACTIVITIES


def test_auto_matches_observed_negatives_over_positives() -> None:
    positives = np.array([20, 5, 40, 10, 2, 100], dtype=np.int64)
    observed = np.array([200, 100, 300, 250, 50, 400], dtype=np.int64)

    weights = resolve_pos_weights("auto", positives, int(observed.sum()), observed)

    expected = (observed - positives) / positives
    assert weights == pytest.approx(expected.tolist())


def test_auto_divides_by_observed_not_by_rows() -> None:
    """A masked activity must not inherit the un-masked row count as negatives."""
    positives = np.array([10] * N_ACTIVITIES, dtype=np.int64)
    observed = np.array([100, 100, 100, 100, 100, 100], dtype=np.int64)
    # 1,000 rows in the anchors, but only 100 observed cells per activity: with
    # the row count as denominator the weight would come out at 99.0 instead of 9.0.
    masked = observed.copy()
    masked[4] = 50  # pr_review_comment lost half its cells to the audit

    weights = resolve_pos_weights("auto", positives, 1000, masked)

    assert weights[4] == pytest.approx((50 - 10) / 10)
    assert weights[0] == pytest.approx((100 - 10) / 10)
    assert weights[4] != pytest.approx((1000 - 10) / 10)


def test_auto_falls_back_to_row_count_when_observed_is_unknown() -> None:
    """Old callers that only know the row count still get the same formula."""
    positives = np.array([25, 25, 25, 25, 25, 25], dtype=np.int64)
    weights = resolve_pos_weights("auto", positives, 100)
    assert weights == pytest.approx([(100 - 25) / 25] * N_ACTIVITIES)


def test_auto_stays_finite_with_a_zero_positive_activity() -> None:
    positives = np.array([0, 5, 5, 5, 5, 5], dtype=np.int64)
    observed = np.array([100] * N_ACTIVITIES, dtype=np.int64)
    weights = resolve_pos_weights("auto", positives, 600, observed)
    assert np.all(np.isfinite(weights))
    assert weights[0] > 1.0


def test_doc_mode_returns_the_published_constants() -> None:
    positives = np.array([1] * N_ACTIVITIES, dtype=np.int64)
    observed = np.array([100] * N_ACTIVITIES, dtype=np.int64)
    assert resolve_pos_weights("doc", positives, 600, observed) == list(DOC_POS_WEIGHTS)


def test_explicit_values_round_trip() -> None:
    weights = resolve_pos_weights("7.3,18,11,12.8,19.3,1.3", np.zeros(6), 0)
    assert weights == pytest.approx([7.3, 18.0, 11.0, 12.8, 19.3, 1.3])


def test_explicit_values_need_six_entries() -> None:
    with pytest.raises(SystemExit, match="comma-separated"):
        resolve_pos_weights("1,2,3", np.zeros(6), 0)


def test_explicit_values_must_be_numeric() -> None:
    with pytest.raises(SystemExit, match="must be numbers"):
        resolve_pos_weights("1,2,3,4,5,oops", np.zeros(6), 0)


def test_auto_needs_one_count_per_activity() -> None:
    with pytest.raises(SystemExit, match="per-activity"):
        resolve_pos_weights("auto", np.zeros(3), 100, np.full(6, 100))


def test_activity_order_is_the_metric_order() -> None:
    """The weights are zipped against ACTIVITY_NAMES positionally."""
    assert len(ACTIVITY_NAMES) == N_ACTIVITIES == len(DOC_POS_WEIGHTS)


# ---------------------------------------------------------------------------
# The production driver must agree with the ablation arm it was validated by.
#
# `scripts/run_training.py` carries its own copy of `resolve_pos_weights` (the
# ablation drivers are deliberately frozen copies of the base pipeline), so the
# two can drift.  They already did once: the ablation arm was fixed to divide by
# observed cells while the production driver kept dividing by raw rows, which
# under masking inflates every weight -- for `open_pr` on the default 8-anchor
# range it produced 6.80 instead of 3.46, i.e. 94% of the doc-constant inflation
# the ablation grid measures as worth -4.26 score @0.5.
# ---------------------------------------------------------------------------


def test_production_resolver_matches_the_ablation_arm() -> None:
    from scripts.run_training import resolve_pos_weights as production

    positives = np.array([20, 5, 40, 10, 2, 100], dtype=np.int64)
    observed = np.array([200, 100, 300, 250, 50, 400], dtype=np.int64)
    cases = [
        ("doc", positives, int(observed.sum()), observed),
        ("auto", positives, int(observed.sum()), observed),
        ("auto", positives, 1000, None),
        ("7.3,18,11,12.8,19.3,1.3", positives, 0, None),
    ]
    for spec, pos, n_rows, obs in cases:
        assert production(spec, pos, n_rows, obs) == pytest.approx(
            resolve_pos_weights(spec, pos, n_rows, obs)
        ), spec


def test_production_resolver_uses_observed_cells_not_rows() -> None:
    from scripts.run_training import resolve_pos_weights as production

    positives = np.array([10] * N_ACTIVITIES, dtype=np.int64)
    observed = np.array([100] * N_ACTIVITIES, dtype=np.int64)
    observed[4] = 50
    weights = production("auto", positives, 1000, observed)
    assert weights[4] == pytest.approx((50 - 10) / 10)
    assert weights[0] == pytest.approx((100 - 10) / 10)


def test_final_training_defaults_come_from_the_ablation_grid() -> None:
    """Lock the three settings the ablation grid settled, so they cannot drift back."""
    from scripts.run_training import parse_args

    args = parse_args([])
    assert args.pos_weights == "auto"  # +4.26 @0.5, 3/3 seeds (experiments/ablations/RESULTS.md)
    assert args.num_gnn_layers == 1  # 2 layers: +0.11 @CMA on 2/3 seeds = noise, +65% time
    assert args.epochs == 310  # every ablation cell and all 60 HPO trials ran 310
