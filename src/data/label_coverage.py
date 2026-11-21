"""Which (anchor, activity) label cells the coverage audit says are observable.

A label is a count of events in a 7-day window.  If GH Archive dropped one
activity's stream on one of those days, the count is wrong in a specific
direction: it can only lose positives, so it silently turns "did open a PR"
into "did not".  Training on that is worse than training on less data, because
it teaches the model that the activity never happens.

`docs/VALIDATION_DATA.md` section 5 prescribes the fix: judge the *cell*
(anchor, activity), not the day.  A day whose `push` stream survived is a valid
push cell even when its `pr_review` stream collapsed.  This module turns
``raw_data/coverage_activity.csv`` into that verdict, so the trainer can mark
the unusable cells as *unobserved* (dropped from the loss) instead of zero.

The source of truth for the day verdict is
:func:`src.data.activity_decay_analysis.anchor_usability`, which prefers the
``usable`` band over the stricter ``clean`` one -- with 19/24 hours typically
intact, P(24/24) is under 1% and a whole month reads as unusable otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.activity_decay_analysis import anchor_usability
from src.data.temporal_split import LABEL_COLS

DEFAULT_COVERAGE = Path("raw_data/coverage_activity.csv")

# Label-count sanity rule (`mode="hybrid"`).
#
# The coverage audit judges an hour by the *share* of events that are of one
# type, against the local (±7 day) median share.  That test is defeated by one
# specific failure mode GH Archive shows in 2026-10/11: `PushEvent` disappears
# while the pull-request / issue / comment / review streams stay at their normal
# absolute volume.  Total events fall ~10x (push is ~90% of a normal hour), so
# every surviving type's *share* rises 4-6x and the audit marks the whole month
# unusable -- even though the label side of those activities lost nothing.
#
# Measured on 2026-11 (9,000 cohort users, 7-day windows): `open_pr` 356-371
# positives against 306-447 in October, `open_issue` 203-210 vs 181-242,
# `issue_comment` 200-244 vs 181-293, `pr_review` 230-261 vs 217-297.  Only `push`
# (315-457 vs 2,058-2,183) and `pr_review_comment` (65-98 vs 143-197) really
# collapse.
#
# So a second, independent test runs on the quantity that actually matters:
# how many positives the 7-day window produced, compared with the same anchor's
# neighbours (±4 anchors, excluding itself).  A cell is count-sane when that
# ratio is inside the band below.  `hybrid` mode keeps a cell the audit trusts
# OR one the counts show is intact, which is what lets the sparse late regime --
# the one the test week lives in -- enter training at all.
COUNT_RATIO_BAND: tuple[float, float] = (0.6, 2.0)
COUNT_NEIGHBOURS = 4


def load_day_flags(path: str | Path = DEFAULT_COVERAGE) -> pd.DataFrame | None:
    """`(day, activity) -> usable` as a wide frame, or None when the audit is absent."""
    coverage_path = Path(path)
    if not coverage_path.exists():
        return None

    coverage = pd.read_csv(coverage_path)
    if coverage.empty or "usable" not in coverage.columns:
        return None

    wide = coverage.pivot_table(index="day", columns="activity", values="usable")
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index))
    frame = pd.DataFrame(index=wide.index)
    for act in LABEL_COLS:
        if act in wide.columns:
            frame[f"usable_{act}"] = wide[act].fillna(False).astype(bool)
        else:
            frame[f"usable_{act}"] = False
    return frame


def count_sanity_flags(
    positives: np.ndarray,
    *,
    neighbours: int = COUNT_NEIGHBOURS,
    band: tuple[float, float] = COUNT_RATIO_BAND,
) -> np.ndarray:
    """`(anchor, activity) -> bool` from the label counts of neighbouring anchors.

    ``positives`` is an ``(n_anchors, n_activities)`` count of positives per
    anchor.  The comparison is against the median of the ``neighbours`` anchors
    on either side, excluding the anchor itself, so a steady regime drift
    cancels out and only a discontinuity is flagged -- the same reasoning as the
    audit's local density baseline, applied to the label side.

    A cell with zero positives is never count-sane: that is the signature of a
    dropped stream, not of a quiet week.
    """
    counts = np.asarray(positives, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"positives must be 2-D (anchors, activities); got {counts.shape}")
    n_anchors = counts.shape[0]
    lo, hi = float(band[0]), float(band[1])
    flags = np.zeros(counts.shape, dtype=bool)
    for i in range(n_anchors):
        left = max(0, i - neighbours)
        right = min(n_anchors, i + neighbours + 1)
        pool = np.concatenate([counts[left:i], counts[i + 1 : right]], axis=0)
        if pool.shape[0] < 2:
            # Too few neighbours at the corpus edge to measure anything.
            flags[i] = counts[i] > 0
            continue
        median = np.median(pool, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(median > 0, counts[i] / median, 1.0)
        flags[i] = (counts[i] > 0) & (ratio >= lo) & (ratio <= hi)
    return flags


def observed_activities(
    cutoffs: list[pd.Timestamp],
    day_flags: pd.DataFrame | None,
    days: int = 7,
    mode: str = "label",
    max_delta: int = 0,
    positives: np.ndarray | None = None,
) -> pd.DataFrame:
    """`(anchor, activity) -> bool`: may this label cell enter the loss?

    ``mode="label"`` requires every day of the label window to be usable for
    that activity; ``mode="strict"`` additionally requires the ``max_delta``
    days before the cutoff (the feature lookback) to be usable; ``mode="off"``
    trusts everything; ``mode="hybrid"`` is ``label`` OR
    :func:`count_sanity_flags`, and needs ``positives``.
    """
    anchors = pd.DatetimeIndex(pd.to_datetime(cutoffs))
    if mode == "hybrid":
        if positives is None:
            raise ValueError("mode='hybrid' needs the per-anchor positive counts")
        audit = (
            pd.DataFrame(True, index=anchors, columns=pd.Index(LABEL_COLS))
            if day_flags is None
            else anchor_usability(
                list(anchors), day_flags, mode="label", horizon_days=days, max_delta=max_delta
            )
        )
        counts = count_sanity_flags(np.asarray(positives))
        if counts.shape != audit.to_numpy().shape:
            raise ValueError(
                f"positives shape {counts.shape} does not match the anchor grid "
                f"{audit.to_numpy().shape}"
            )
        return pd.DataFrame(
            audit.to_numpy(dtype=bool) | counts, index=anchors, columns=audit.columns
        )
    if day_flags is None:
        return pd.DataFrame(True, index=anchors, columns=pd.Index(LABEL_COLS))
    return anchor_usability(
        list(anchors), day_flags, mode=mode, horizon_days=days, max_delta=max_delta
    )


def observed_matrix(
    observed_row: pd.Series | np.ndarray,
) -> np.ndarray:
    """Broadcast a `(6,)` verdict into a `(n_users, 6)` observation mask."""
    flags = np.asarray(observed_row, dtype=bool).reshape(1, -1)
    return np.repeat(flags, 1, axis=0)
