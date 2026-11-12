"""CMA-ES optimizer for per-activity decision thresholds.

The search variable is ``theta`` in ``[0.01, 0.99]^6``.  Candidate theta values
are applied as ``prediction = (probability >= theta)`` and scored with
:func:`src.utils.metrics.score_binary`, i.e. exactly the TA score
``(macro_f1 + micro_f1) / 2``.  CMA-ES minimizes, so the objective returns the
negative score.

Default budget (as requested): population 16, ``maxfevals=3000``, which is
about 188 generations with ``x0=[0.5]*6`` and ``sigma0=0.15``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import cma
import numpy as np

from src.utils.metrics import ACTIVITY_NAMES, N_ACTIVITIES, ScoreResult, score_binary

__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_MAXFEVALS",
    "DEFAULT_POPSIZE",
    "DEFAULT_SIGMA0",
    "DEFAULT_X0",
    "optimize_thresholds_cmaes",
    "predict_with_thresholds",
]

DEFAULT_X0: tuple[float, ...] = (0.5,) * N_ACTIVITIES
DEFAULT_SIGMA0 = 0.15
DEFAULT_POPSIZE = 16
DEFAULT_MAXFEVALS = 3000
DEFAULT_BOUNDS = (0.01, 0.99)


def _to_numpy(x, name: str) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_probability_matrix(y_probs, n_events: int) -> np.ndarray:
    probs = _to_numpy(y_probs, "y_probs")
    if probs.ndim != 2:
        raise ValueError(f"y_probs must be 2-D (rows, activities); got {probs.shape}")
    if probs.shape[1] != n_events:
        raise ValueError(f"expected {n_events} activity columns, got {probs.shape[1]}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("y_probs contains NaN or infinity")
    if probs.size and (probs.min() < 0.0 or probs.max() > 1.0):
        raise ValueError("y_probs must be probabilities in [0, 1]")
    return probs


def predict_with_thresholds(y_probs, thresholds) -> np.ndarray:
    """Binarize a probability matrix with a scalar or per-activity theta."""
    probs = _to_numpy(y_probs, "y_probs")
    if probs.ndim != 2:
        raise ValueError(f"y_probs must be 2-D (rows, activities); got {probs.shape}")
    theta = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if theta.size == 1:
        theta = np.full(probs.shape[1], theta[0], dtype=np.float64)
    if theta.size != probs.shape[1]:
        raise ValueError(f"expected 1 or {probs.shape[1]} thresholds, got {theta.size}")
    return (probs >= theta[None, :]).astype(np.int64)


def optimize_thresholds_cmaes(
    y_true,
    y_probs,
    *,
    x0: Sequence[float] | np.ndarray | None = None,
    sigma0: float = DEFAULT_SIGMA0,
    popsize: int = DEFAULT_POPSIZE,
    maxfevals: int = DEFAULT_MAXFEVALS,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
    events: Sequence[str] | None = None,
    seed: int = 42,
    verbose: int = -9,
    generation_callback: Callable[[int, float, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximize the TA combined score over six per-activity thresholds.

    Parameters
    ----------
    y_true:
        Binary ground truth, shape ``(n_rows, n_events)``. NumPy or torch.
    y_probs:
        Predicted probabilities, same shape. NumPy or torch.
    x0:
        Initial guess; defaults to ``[0.5] * n_events``.
    sigma0:
        Initial CMA-ES step size (default 0.15).
    popsize:
        CMA-ES population size (default 16).
    maxfevals:
        Evaluation budget (default 3000, ~188 generations at popsize 16).
    bounds:
        Inclusive ``(lower, upper)`` bounds for every theta entry.
    generation_callback:
        Optional ``fn(generation, best_score_so_far, best_theta)`` called after
        each generation, for W&B or console progress.

    Returns
    -------
    (theta, metrics)
        ``theta`` is a float64 array of length ``n_events`` inside ``bounds``.
        ``metrics`` contains the TA keys (``per_event``, ``per_class_f1``,
        ``macro_f1``, ``micro_f1``, ``score``) plus optimization diagnostics
        ``n_generations``, ``n_evaluations``, ``initial_score`` and ``stop``.
    """
    names = tuple(ACTIVITY_NAMES if events is None else events)
    if not names:
        raise ValueError("events must contain at least one activity")
    n_events = len(names)

    if int(popsize) < 1:
        raise ValueError(f"popsize must be >= 1, got {popsize}")
    if int(maxfevals) < 1:
        raise ValueError(f"maxfevals must be >= 1, got {maxfevals}")
    if not np.isfinite(sigma0) or float(sigma0) <= 0.0:
        raise ValueError(f"sigma0 must be finite and > 0, got {sigma0}")

    lo, hi = float(bounds[0]), float(bounds[1])
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"bounds must satisfy 0 <= lower < upper <= 1; got {bounds}")

    probs = _as_probability_matrix(y_probs, n_events)
    truth = _to_numpy(y_true, "y_true")
    if truth.shape != probs.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape} vs y_probs {probs.shape}")

    if x0 is None:
        x0_arr = np.full(n_events, 0.5, dtype=np.float64)
    else:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0_arr.size != n_events:
            raise ValueError(f"x0 must have {n_events} entries, got {x0_arr.size}")
        if not np.all(np.isfinite(x0_arr)):
            raise ValueError(f"x0 must be finite; got {x0_arr}")
        if np.any(x0_arr < lo) or np.any(x0_arr > hi):
            raise ValueError(f"x0 must lie inside bounds {bounds}; got {x0_arr}")

    def evaluate(theta: np.ndarray) -> ScoreResult:
        theta_eval = np.clip(np.asarray(theta, dtype=np.float64), lo, hi)
        preds = (probs >= theta_eval[None, :]).astype(np.int64)
        return score_binary(truth, preds, events=names)

    def objective(theta: np.ndarray) -> float:
        # CMA-ES minimizes, so negate the score it should maximize.
        return -evaluate(theta)["score"]

    initial_metrics = evaluate(x0_arr)

    es = cma.CMAEvolutionStrategy(
        x0_arr.tolist(),
        float(sigma0),
        options={
            "bounds": [lo, hi],
            "maxfevals": int(maxfevals),
            "popsize": int(popsize),
            "seed": int(seed),
            "verbose": int(verbose),
            # The score is piecewise constant in theta (it changes only when a
            # probability crosses a threshold), so CMA's default convergence
            # tests see a perfectly flat fitness landscape and can stop after
            # ~110 generations. Disable those early stops so the requested
            # ~3000-evaluation / ~200-generation budget is actually used.
            "tolflatfitness": int(maxfevals),
            "tolfun": 0,
            "tolfunrel": 0,
            "tolfunhist": 0,
            "tolx": 0,
            "tolstagnation": 0,
            "tolupsigma": 0,
            "tolconditioncov": 0,
            "maxiter": int(maxfevals) // max(1, int(popsize)) + 2,
        },
    )

    n_generations = 0
    n_evaluations = 0
    best_score = float(initial_metrics["score"])
    best_theta = x0_arr.copy()

    stop_info: Any = {}
    while not es.stop():
        solutions = es.ask()
        values: list[float] = []
        for solution in solutions:
            value = objective(solution)
            values.append(value)
            n_evaluations += 1
            score = -value
            if score > best_score:
                best_score = score
                best_theta = np.clip(np.asarray(solution, dtype=np.float64), lo, hi)
        es.tell(solutions, values)
        n_generations += 1

        if generation_callback is not None:
            generation_callback(n_generations, best_score, best_theta.copy())

    # Include es.result.xbest in the comparison: the archive may hold a better
    # point than the last generation's best.
    xbest = x0_arr if es.result.xbest is None else es.result.xbest
    candidate = np.clip(np.asarray(xbest, dtype=np.float64), lo, hi)
    candidate_metrics = evaluate(candidate)
    if float(candidate_metrics["score"]) >= best_score:
        best_theta = candidate
        best_score = float(candidate_metrics["score"])

    stop_info = es.stop() or {}
    final_metrics = evaluate(best_theta)
    metrics: dict[str, Any] = dict(final_metrics)
    metrics.update(
        {
            "n_generations": n_generations,
            "n_evaluations": n_evaluations,
            "initial_score": initial_metrics["score"],
            "stop": {str(k): str(v) for k, v in dict(stop_info).items()},
        }
    )
    return np.asarray(best_theta, dtype=np.float64), metrics
