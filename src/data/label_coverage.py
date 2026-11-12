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


def observed_activities(
    cutoffs: list[pd.Timestamp],
    day_flags: pd.DataFrame | None,
    days: int = 7,
    mode: str = "label",
    max_delta: int = 0,
) -> pd.DataFrame:
    """`(anchor, activity) -> bool`: may this label cell enter the loss?

    ``mode="label"`` requires every day of the label window to be usable for
    that activity; ``mode="strict"`` additionally requires the ``max_delta``
    days before the cutoff (the feature lookback) to be usable; ``mode="off"``
    trusts everything.
    """
    anchors = pd.DatetimeIndex(pd.to_datetime(cutoffs))
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
