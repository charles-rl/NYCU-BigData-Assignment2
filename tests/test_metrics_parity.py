"""Parity tests: src.utils.metrics must match scripts/evaluate.py numerically.

The TA scorer is imported (never modified) and its ``scores()`` function is
compared against the in-memory implementation on randomized inputs, including
degenerate columns (all-zero truth, all-zero predictions).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import ACTIVITY_NAMES, score_binary, score_from_probs

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_PY = REPO_ROOT / "scripts" / "evaluate.py"


def _load_ta_module():
    spec = importlib.util.spec_from_file_location("ta_evaluate", EVALUATE_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EVALUATE_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ta_evaluate", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ta():
    return _load_ta_module()


def _frames(pred: np.ndarray, truth: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = np.arange(len(pred))
    cols = np.asarray(ACTIVITY_NAMES, dtype=object)
    pred_df = pd.DataFrame(pred, columns=cols)
    pred_df.insert(0, "UserID", users)
    truth_df = pd.DataFrame(truth, columns=cols)
    truth_df.insert(0, "UserID", users)
    return pred_df, truth_df


@pytest.mark.parametrize(
    "n_rows, prevalence, seed",
    [
        (9000, 0.05, 0),
        (9000, 0.43, 1),
        (137, 0.5, 2),
        (3, 0.0, 3),  # all-zero truth: TA scores F1 = 1.0 per empty class
        (4, 1.0, 4),  # all-one truth
    ],
)
def test_matches_ta_script(ta, n_rows, prevalence, seed):
    rng = np.random.default_rng(seed)
    truth = (rng.random((n_rows, len(ACTIVITY_NAMES))) < prevalence).astype(np.int64)
    pred = (rng.random((n_rows, len(ACTIVITY_NAMES))) < max(prevalence, 0.02)).astype(np.int64)

    pred_df, truth_df = _frames(pred, truth)
    expected = ta.scores(pred_df, truth_df, list(ACTIVITY_NAMES))
    got = score_binary(truth, pred)

    for name in ACTIVITY_NAMES:
        assert got["per_event"][name] == pytest.approx(expected["per_event"][name], abs=1e-12)
    assert got["macro_f1"] == pytest.approx(expected["macro_f1"], abs=1e-12)
    assert got["micro_f1"] == pytest.approx(expected["micro_f1"], abs=1e-12)
    assert got["score"] == pytest.approx(expected["score"], abs=1e-12)

    # the reported requirement is 6 decimal places
    assert round(got["score"], 6) == round(expected["score"], 6)


def test_matches_ta_script_on_shuffled_row_order(ta):
    rng = np.random.default_rng(7)
    truth = (rng.random((500, 6)) < 0.2).astype(np.int64)
    pred = (rng.random((500, 6)) < 0.2).astype(np.int64)

    # TA aligns predictions to truth's UserID order; do the same with a permutation.
    order = rng.permutation(500)
    pred_df, truth_df = _frames(pred[order], truth)
    expected = ta.scores(pred_df, truth_df, list(ACTIVITY_NAMES))  # reindexes internally
    got = score_binary(truth, pred[order])
    assert got["score"] == pytest.approx(expected["score"], abs=1e-12)


def test_accepts_torch_tensors_and_probs():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    truth = (rng.random((64, 6)) < 0.3).astype(np.int64)
    probs = rng.random((64, 6))

    from_np = score_from_probs(truth, probs, 0.5)
    from_torch = score_from_probs(
        torch.from_numpy(truth), torch.from_numpy(probs), torch.tensor(0.5)
    )
    assert from_torch["score"] == pytest.approx(from_np["score"], abs=1e-12)
    assert from_torch["per_event"] == from_np["per_event"]


def test_rejects_non_binary_and_misaligned_shapes():
    ok = np.zeros((4, 6), dtype=np.int64)
    with pytest.raises(ValueError):
        score_binary(ok, ok + 2)
    with pytest.raises(ValueError):
        score_binary(ok, np.zeros((5, 6), dtype=np.int64))
    with pytest.raises(ValueError):
        score_binary(ok, np.zeros((4, 5), dtype=np.int64))
    with pytest.raises(ValueError):
        score_from_probs(ok, np.full((4, 6), 0.5), [0.5, 0.5])
