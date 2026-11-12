"""CMA-ES threshold optimizer smoke tests on synthetic data (no disk I/O)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import ACTIVITY_NAMES, score_binary, score_from_probs
from src.utils.threshold_optimizer import (
    DEFAULT_BOUNDS,
    DEFAULT_MAXFEVALS,
    DEFAULT_POPSIZE,
    DEFAULT_SIGMA0,
    DEFAULT_X0,
    optimize_thresholds_cmaes,
    predict_with_thresholds,
)


def _synthetic(n_rows: int = 9000, seed: int = 0):
    """Skewed per-activity probabilities with different optimal cutoffs."""
    rng = np.random.default_rng(seed)
    # per-activity base rates, deliberately very different (4.9% .. 43%)
    base = np.array([0.08, 0.07, 0.06, 0.05, 0.049, 0.43])
    truth = (rng.random((n_rows, 6)) < base).astype(np.int64)

    # informative scores: positives get a beta bump, negatives stay low
    probs = rng.beta(2.0, 8.0, size=(n_rows, 6))
    probs[truth == 1] = rng.beta(5.0, 3.0, size=int(truth.sum()))
    probs = np.clip(probs, 0.0, 1.0)
    return truth, probs


def test_default_budget_constants():
    assert DEFAULT_POPSIZE == 16
    assert DEFAULT_MAXFEVALS == 3000
    assert DEFAULT_BOUNDS == (0.01, 0.99)
    assert DEFAULT_X0 == (0.5,) * 6
    assert DEFAULT_SIGMA0 == 0.15
    # ~200 generations at popsize 16
    assert 180 <= DEFAULT_MAXFEVALS / DEFAULT_POPSIZE <= 220


def test_runs_on_dummy_9000x6_matrix():
    truth, probs = _synthetic()
    assert probs.shape == (9000, 6)

    theta, metrics = optimize_thresholds_cmaes(truth, probs)

    assert theta.shape == (6,)
    assert np.all(theta >= 0.01) and np.all(theta <= 0.99)
    assert np.all(np.isfinite(theta))

    # n_generations = budget / popsize (last generation may be partially spent)
    assert metrics["n_evaluations"] >= 16
    assert metrics["n_generations"] >= 150

    default_score = score_from_probs(truth, probs, 0.5)["score"]
    assert metrics["score"] >= default_score, (metrics["score"], default_score)

    # reported metrics must equal a fresh recomputation from theta
    recomputed = score_binary(truth, predict_with_thresholds(probs, theta))
    assert recomputed["score"] == pytest.approx(metrics["score"], abs=1e-12)
    assert recomputed["macro_f1"] == pytest.approx(metrics["macro_f1"], abs=1e-12)
    assert recomputed["micro_f1"] == pytest.approx(metrics["micro_f1"], abs=1e-12)
    assert set(metrics["per_event"]) == set(ACTIVITY_NAMES)


def test_is_deterministic_with_seed():
    truth, probs = _synthetic(n_rows=600)
    theta_a, metrics_a = optimize_thresholds_cmaes(truth, probs, maxfevals=320, seed=123)
    theta_b, metrics_b = optimize_thresholds_cmaes(truth, probs, maxfevals=320, seed=123)
    np.testing.assert_allclose(theta_a, theta_b)
    assert metrics_a["score"] == pytest.approx(metrics_b["score"], abs=1e-12)


def test_reproducible_across_sigma_and_popsize_overrides():
    truth, probs = _synthetic(n_rows=800)
    theta, metrics = optimize_thresholds_cmaes(
        truth, probs, sigma0=0.15, popsize=8, maxfevals=160
    )
    assert theta.shape == (6,)
    assert 0.0 <= metrics["score"] <= 1.0


def test_generation_callback_fires():
    truth, probs = _synthetic(n_rows=400)
    seen: list[tuple[int, float]] = []
    optimize_thresholds_cmaes(
        truth,
        probs,
        maxfevals=96,
        generation_callback=lambda gen, best, _theta: seen.append((gen, best)),
    )
    assert seen
    assert seen[-1][0] == len(seen)
    assert all(0.0 <= best <= 1.0 for _, best in seen)


def test_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    truth, probs = _synthetic(n_rows=300)
    theta, metrics = optimize_thresholds_cmaes(
        torch.from_numpy(truth), torch.from_numpy(probs), maxfevals=96
    )
    assert theta.shape == (6,)
    assert np.isfinite(metrics["score"])


def test_rejects_bad_inputs():
    truth, probs = _synthetic(n_rows=50)
    with pytest.raises(ValueError):
        optimize_thresholds_cmaes(truth, probs[:, :5], maxfevals=32)
    with pytest.raises(ValueError):
        optimize_thresholds_cmaes(truth[:, :5], probs, maxfevals=32)
    with pytest.raises(ValueError):
        optimize_thresholds_cmaes(truth, probs * 2.0, maxfevals=32)
    with pytest.raises(ValueError):
        optimize_thresholds_cmaes(truth, probs, x0=[0.0] * 6, maxfevals=32)
    with pytest.raises(ValueError):
        optimize_thresholds_cmaes(truth, probs, x0=[0.5] * 5, maxfevals=32)
    with pytest.raises(ValueError):
        optimize_thresholds_cmaes(truth, probs, x0=[0.5, 0.5, np.nan, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        optimize_thresholds_cmaes(truth, probs, sigma0=0.0)
    with pytest.raises(ValueError):
        optimize_thresholds_cmaes(truth, probs, popsize=0)
