"""In-memory reproduction of the TA scorer in ``scripts/evaluate.py``.

The official script computes, for each of the six activities,

    F1 = 2*tp / (2*tp + fp + fn)   (1.0 when tp + fp + fn == 0)

then averages those six F1 values (macro-F1), computes one pooled F1 over all
activities (micro-F1), and returns ``(macro_f1 + micro_f1) / 2``.

This module performs exactly that arithmetic on arrays already held in RAM.
It performs no file I/O and accepts NumPy arrays or PyTorch tensors.  Callers
are responsible for aligning prediction rows with truth rows first, as the TA
script does with ``pred.set_index(UserID).loc[truth[UserID]]``.

Example
-------
>>> import numpy as np
>>> from src.utils.metrics import score_binary
>>> truth = np.array([[1, 0], [0, 1]])
>>> pred = np.array([[1, 0], [1, 0]])
>>> score_binary(truth, pred, events=("a", "b"))["score"]
0.5
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np

ACTIVITY_NAMES: tuple[str, ...] = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)
N_ACTIVITIES = len(ACTIVITY_NAMES)

__all__ = [
    "ACTIVITY_NAMES",
    "N_ACTIVITIES",
    "ScoreResult",
    "f1",
    "score_binary",
    "score_from_probs",
    "score_in_memory",
]


class ScoreResult(TypedDict):
    """Return type of :func:`score_binary`."""

    per_event: dict[str, float]
    per_class_f1: list[float]
    macro_f1: float
    micro_f1: float
    score: float


def _as_numpy(x, name: str) -> np.ndarray:
    """Convert a NumPy array, torch tensor, or array-like to a 2-D ndarray."""
    if isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        # torch.Tensor (or anything mimicking its conversion API)
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D with shape (rows, activities); got {arr.shape}")
    return arr


def _as_binary_matrix(x, name: str) -> np.ndarray:
    """Return an int64 0/1 matrix, mirroring the TA script's format checks."""
    arr = _as_numpy(x, name)

    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.int64)

    if np.issubdtype(arr.dtype, np.integer):
        out = arr.astype(np.int64, copy=False)
    else:
        try:
            as_float = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only 0 or 1") from exc
        if not np.all(np.isfinite(as_float)):
            raise ValueError(f"{name} contains NaN or infinity")
        if not np.all(as_float == np.floor(as_float)):
            raise ValueError(f"{name} must contain only 0 or 1")
        out = as_float.astype(np.int64)

    if out.size and not np.isin(out, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 or 1")
    return out


def f1(tp, fp, fn):
    """Exact copy of ``scripts/evaluate.py::f1``.

    A class with no positives and no predicted positives scores 1.0, not 0.0.
    """
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def score_binary(
    y_true,
    y_pred,
    events: Sequence[str] | None = None,
) -> ScoreResult:
    """Compute per-activity, macro, micro F1 and Score on aligned binary arrays.

    Parameters
    ----------
    y_true, y_pred:
        Shape ``(n_rows, n_events)`` arrays of 0/1 values. NumPy arrays and
        torch tensors are both accepted. Rows must already be aligned.
    events:
        Activity names in column order. Defaults to the six TA activities.

    Returns
    -------
    dict with ``per_event`` (name -> F1), ``per_class_f1`` (ordered list),
    ``macro_f1``, ``micro_f1`` and ``score``. Values match the TA script to
    floating-point precision, hence to 6 decimal places for any normal input.
    """
    names = tuple(ACTIVITY_NAMES if events is None else events)
    if not names:
        raise ValueError("events must contain at least one activity")

    true = _as_binary_matrix(y_true, "y_true")
    pred = _as_binary_matrix(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.shape[1] != len(names):
        raise ValueError(
            f"expected {len(names)} activity columns, got {true.shape[1]}"
        )

    # Same masks and reductions as scripts/evaluate.py::scores.
    tp = ((pred == 1) & (true == 1)).sum(axis=0)
    fp = ((pred == 1) & (true == 0)).sum(axis=0)
    fn = ((pred == 0) & (true == 1)).sum(axis=0)

    tp_i = [int(v) for v in tp]
    fp_i = [int(v) for v in fp]
    fn_i = [int(v) for v in fn]

    per_event = {
        name: f1(t, p, n) for name, t, p, n in zip(names, tp_i, fp_i, fn_i, strict=True)
    }
    macro_f1 = sum(per_event.values()) / len(names)  # identical order to the TA loop
    micro_f1 = f1(sum(tp_i), sum(fp_i), sum(fn_i))
    score = (macro_f1 + micro_f1) / 2

    return {
        "per_event": per_event,
        "per_class_f1": [per_event[name] for name in names],
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "score": float(score),
    }


def score_from_probs(
    y_true,
    y_probs,
    thresholds,
    events: Sequence[str] | None = None,
) -> ScoreResult:
    """Binarize probabilities with per-activity thresholds, then score.

    ``thresholds`` may be a scalar or one value per activity. A probability
    equal to its threshold counts as positive (``>=``).
    """
    probs = _as_numpy(y_probs, "y_probs")
    if not np.all(np.isfinite(probs)):
        raise ValueError("y_probs contains NaN or infinity")

    theta = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if theta.size == 1:
        theta = np.full(probs.shape[1], theta[0], dtype=np.float64)
    if theta.size != probs.shape[1]:
        raise ValueError(
            f"expected 1 or {probs.shape[1]} thresholds, got {theta.size}"
        )

    pred = (probs >= theta[None, :]).astype(np.int64)
    return score_binary(y_true, pred, events=events)


# Backwards-compatible alias used by training code.
score_in_memory = score_binary
