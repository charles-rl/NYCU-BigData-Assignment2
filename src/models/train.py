"""Training pipeline with Weights & Biases logging and CMA-ES threshold optimization.

Validation scoring is imported from :mod:`src.utils.metrics`, an in-memory
replica of the TA ``scripts/evaluate.py`` scorer.  Threshold search lives in
:mod:`src.utils.threshold_optimizer`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from src.models.hetero_model import GitHubHeteroHGT
from src.utils.metrics import ACTIVITY_NAMES, score_in_memory
from src.utils.threshold_optimizer import optimize_thresholds_cmaes

__all__ = [
    "ACTIVITY_NAMES",
    "Anchor",
    "optimize_thresholds_cmaes",
    "score_in_memory",
    "train_model",
    "train_model_multi_anchor",
]

# Where wandb writes its `wandb/` run directory.  It must NOT default to the
# repository root: a top-level directory named `wandb` makes every isort-style
# linter classify the third-party `wandb` package as local code and demand the
# import be moved, which flaps depending on whether a run has happened yet.
WANDB_DIR = os.environ.get("WANDB_DIR", "logs")


@dataclass
class Anchor:
    """One (cutoff, label window) pair, aligned to its own graph's node order.

    ``labels`` holds the binary outcome and ``observed`` says whether that
    (user, activity) label was actually measurable -- an activity whose GH
    Archive stream was damaged on a given day is *unobserved*, not negative.
    The loss only sees observed entries, which is what lets a broken day
    contribute four usable activities instead of poisoning all six.

    Invariant: ``labels`` and ``observed`` are both ``(n_users, 6)`` and
    ``labels`` is zero wherever ``observed`` is false.
    """

    name: str
    cutoff: pd.Timestamp
    graph: Any
    labels: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    positives: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.mask.sum())

    @property
    def graph_megabytes(self) -> float:
        """Approximate CPU footprint of the graph tensors (features + edges)."""
        total = 0
        for node_type in self.graph.node_types:
            x = getattr(self.graph[node_type], "x", None)
            if x is not None:
                total += x.numel() * x.element_size()
        for edge_type in self.graph.edge_types:
            store = self.graph[edge_type]
            total += store.edge_index.numel() * store.edge_index.element_size()
            attr = getattr(store, "edge_attr", None)
            if attr is not None:
                total += attr.numel() * attr.element_size()
        return total / 2**20

    @property
    def n_observed(self) -> int:
        return int(self.observed[self.mask].sum())


def _anchor_batch(anchor: Anchor, device: str):
    """Flattened (logits target, observed weight, row mask) for one anchor."""
    y = torch.from_numpy(anchor.labels[anchor.mask]).float().to(device)
    obs = torch.from_numpy(anchor.observed[anchor.mask]).float().to(device)
    return y, obs


def _masked_bce(logits: torch.Tensor, y: torch.Tensor, obs: torch.Tensor, weight) -> torch.Tensor:
    """BCEWithLogits restricted to the observed entries of the label matrix."""
    per_entry = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=weight, reduction="none"
    )
    return (per_entry * obs).sum() / obs.sum().clamp_min(1.0)


def _as_numpy_bool(x) -> np.ndarray:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return arr.astype(bool)


def _eta_min(base_lr: float) -> float:
    """Floor for the schedule: 1% of the base lr, never below 1e-5."""
    return max(1e-5, base_lr * 0.01)


def _scheduler_config(
    schedule: str, epochs: int, base_lr: float, t0: int, t_mult: int
) -> dict[str, object]:
    """Describe the learning-rate schedule for ``wandb.config``.

    Validation happens here so both trainers share one definition of the
    allowed values.
    """
    if schedule not in {"cosine", "warm-restarts", "none"}:
        raise ValueError(
            f"lr_schedule must be 'cosine', 'warm-restarts' or 'none', got {schedule!r}"
        )
    if schedule == "warm-restarts" and t0 < 1:
        raise ValueError(f"lr_t0 must be >= 1, got {t0}")
    cfg: dict[str, object] = {
        "scheduler": {
            "cosine": "CosineAnnealingLR",
            "warm-restarts": "CosineAnnealingWarmRestarts",
            "none": "constant",
        }[schedule],
        "scheduler_eta_min": _eta_min(base_lr),
    }
    if schedule == "cosine":
        cfg["scheduler_t_max"] = epochs
    elif schedule == "warm-restarts":
        cfg["scheduler_t0"] = t0
        cfg["scheduler_t_mult"] = t_mult
    return cfg


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    epochs: int,
    base_lr: float,
    t0: int,
    t_mult: int,
):
    """Build the lr scheduler (``None`` means a constant lr).

    ``"cosine"`` is one monotone cosine from ``base_lr`` to ``eta_min`` spread
    over the *whole* run, so raising ``epochs`` stretches a single decay rather
    than repeating it.  ``"warm-restarts"`` is the sawtooth: lr snaps back to
    ``base_lr`` at the end of every cycle, which lets the model escape a
    plateau after the cosine has bottomed out.  Cycle ``k`` lasts
    ``t0 * t_mult**k`` epochs, so ``t_mult > 1`` grows them geometrically
    (5, 10, 20, 40, ...) while ``t_mult == 1`` keeps them constant.
    """
    _scheduler_config(schedule, epochs, base_lr, t0, t_mult)
    if schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=_eta_min(base_lr))
    if schedule == "warm-restarts":
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult, eta_min=_eta_min(base_lr)
        )
    return None


def _current_lr(scheduler, optimizer: torch.optim.Optimizer) -> float:
    """Learning rate actually in effect for the next step."""
    if scheduler is None:
        return float(optimizer.param_groups[0]["lr"])
    return float(scheduler.get_last_lr()[0])


def _fit_thresholds_on_train(
    *,
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    sigma0: float,
    popsize: int,
    maxfevals: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Fit the six thresholds on TRAINING rows, then apply them to validation.

    The thresholds are parameters fitted by optimization (3000 evaluations over
    6 variables), so fitting them on the validation week and reporting the score
    of that same week would be train-on-test leakage.  The optimizer therefore
    never sees a validation label or probability; the returned validation
    metrics are the honest held-out score of the fitted theta.

    Returns ``(theta, val_metrics, fit_metrics)`` where ``val_metrics`` is the
    score of ``theta`` on the validation rows and ``fit_metrics`` the score on
    the rows theta was fitted on.
    """
    if train_probs is None or train_labels is None or len(train_probs) == 0:
        raise ValueError("cannot fit thresholds: no training probabilities")

    theta, fit_metrics = optimize_thresholds_cmaes(
        train_labels,
        train_probs,
        sigma0=sigma0,
        popsize=popsize,
        maxfevals=maxfevals,
    )

    val_metrics = score_in_memory(
        val_labels, (val_probs >= theta[None, :]).astype(np.int64)
    )
    return theta, dict(val_metrics), dict(fit_metrics)


def train_model(
    model: GitHubHeteroHGT,
    train_graph,
    train_labels: torch.Tensor,
    val_graph,
    val_labels: np.ndarray,
    val_user_ids: np.ndarray,
    pos_weights: list[float],
    cohort_mask_train: torch.Tensor,
    cohort_mask_val: torch.Tensor,
    epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_interval: int = 5,
    grad_clip: float = 1.0,
    lr_schedule: str = "warm-restarts",
    lr_t0: int = 5,
    lr_t_mult: int = 2,
    cma_sigma0: float = 0.15,
    cma_popsize: int = 16,
    cma_maxfevals: int = 3000,
    project_name: str = "nycu-bigdata-asg2",
    run_name: str = "hgt-exp-decay-cmaes",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online",
    extra_config: dict[str, object] | None = None,
    truth_path: str = "splits/answer.csv",
    val_pred_path: str = "splits/val_best_pred.csv",
    save_model_path: str | None = None,
) -> tuple[GitHubHeteroHGT, np.ndarray, dict[str, object]]:
    """Train the HGT model and tune per-activity thresholds with CMA-ES.

    ``pos_weights`` must contain one positive-class weight per activity.  They
    are passed directly to ``BCEWithLogitsLoss(pos_weight=...)``, which expects
    a tensor broadcastable to the ``(batch, 6)`` logits/targets.
    """
    if len(pos_weights) != len(ACTIVITY_NAMES):
        raise ValueError(
            f"pos_weights must have {len(ACTIVITY_NAMES)} entries, got {len(pos_weights)}"
        )

    # 1. Initialize Weights & Biases.  wandb.log below is asynchronous by
    #    default (no network flush is awaited), so it does not block training.
    wandb.init(
        project=project_name,
        name=run_name,
        mode=wandb_mode,
        config={
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "pos_weights": pos_weights,
            "optimizer": "AdamW",
            **_scheduler_config(lr_schedule, epochs, lr, lr_t0, lr_t_mult),
            "grad_clip_max_norm": grad_clip,
            "threshold_optimizer": "CMA-ES",
            "cma_sigma0": cma_sigma0,
            "cma_popsize": cma_popsize,
            "cma_maxfevals": cma_maxfevals,
            "device": device,
            **(extra_config or {}),
        },
        dir=WANDB_DIR,
    )

    model = model.to(device)
    train_graph = train_graph.to(device)
    val_graph = val_graph.to(device)

    # normalize masks once: torch mask for train, numpy mask for validation
    train_mask = cohort_mask_train.to(device=device, dtype=torch.bool)
    val_mask = (
        cohort_mask_val.detach().cpu().numpy()
        if torch.is_tensor(cohort_mask_val)
        else np.asarray(cohort_mask_val)
    )
    val_mask = val_mask.astype(bool)

    # pos_weight shape (6,) multiplies the positive term of each of the six
    # independent sigmoid outputs; it does not need to be expanded to (N, 6).
    weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # One scheduler step per epoch.  "cosine" decays lr from `lr` to eta_min
    # over the whole run (so a bigger `epochs` stretches one cosine, it does not
    # repeat it); "warm-restarts" snaps back to `lr` every `lr_t0` epochs.
    scheduler = _build_scheduler(optimizer, lr_schedule, epochs, lr, lr_t0, lr_t_mult)

    y_train = train_labels.to(device).float()

    best_val_score = -1.0
    best_thresholds = np.full(len(ACTIVITY_NAMES), 0.5, dtype=np.float64)
    best_val_metrics: dict[str, object] = {}
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(train_graph)
        loss = criterion(logits[train_mask], y_train[train_mask])
        loss.backward()

        # Clip the total gradient norm BEFORE optimizer.step(); the returned
        # value is the pre-clipping norm, useful for spotting instability.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        log_dict: dict[str, object] = {
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": _current_lr(scheduler, optimizer),
            "epoch": epoch,
        }

        # Train-side scores at the static 0.5 cutoff, so the W&B charts show a
        # train-vs-validation comparison on the same metric, not just loss.
        with torch.no_grad():
            train_probs = torch.sigmoid(logits[train_mask]).detach().cpu().numpy()
        train_true = y_train[train_mask].detach().cpu().numpy().astype(np.int64)
        train_metrics = score_in_memory(
            train_true, (train_probs >= 0.5).astype(np.int64)
        )
        log_dict.update(
            {
                "train/score": train_metrics["score"],
                "train/macro_f1": train_metrics["macro_f1"],
                "train/micro_f1": train_metrics["micro_f1"],
            }
        )

        # Run validation + CMA-ES threshold optimization
        if epoch % val_interval == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_graph)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_loss = criterion(
                val_logits[val_mask],
                torch.as_tensor(val_labels, dtype=torch.float32, device=device),
            )
            val_probs_cohort = val_probs[val_mask]

            # Baseline score using the static 0.5 threshold
            default_preds = (val_probs_cohort >= 0.5).astype(np.int64)
            default_metrics = score_in_memory(val_labels, default_preds)

            # Theta is fitted on TRAINING rows only and merely applied here, so
            # the validation score is held out rather than tuned-on-test.  The
            # in-loop train probabilities carry dropout noise, so recompute them
            # in eval mode before fitting.
            model.eval()
            with torch.no_grad():
                eval_train_probs = (
                    torch.sigmoid(model(train_graph)[train_mask]).cpu().numpy()
                )
            th, cma_metrics, fit_metrics = _fit_thresholds_on_train(
                train_probs=eval_train_probs,
                train_labels=train_true,
                val_probs=val_probs_cohort,
                val_labels=val_labels,
                sigma0=cma_sigma0,
                popsize=cma_popsize,
                maxfevals=cma_maxfevals,
            )

            log_dict.update(
                {
                    "val/loss": val_loss.item(),
                    "val/score_default_0.5": default_metrics["score"],
                    "val/score_cmaes": cma_metrics["score"],
                    "val/score_theta_on_fit_set": fit_metrics["score"],
                    "val/macro_f1": cma_metrics["macro_f1"],
                    "val/micro_f1": cma_metrics["micro_f1"],
                    "val/cma_generations": fit_metrics["n_generations"],
                    "val/cma_evaluations": fit_metrics["n_evaluations"],
                }
            )
            # How much of the in-sample gain survives the move to unseen rows.
            log_dict["val/score_cmaes_optimistic"] = (
                fit_metrics["score"] - cma_metrics["score"]
            )

            # Log per-class F1 and learned thresholds to W&B
            for idx, act in enumerate(ACTIVITY_NAMES):
                log_dict[f"val_f1/{act}"] = cma_metrics["per_class_f1"][idx]
                log_dict[f"val_threshold/{act}"] = float(th[idx])

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"loss train {loss.item():.4f} / val {val_loss.item():.4f} | "
                f"score train {train_metrics['score'] * 100:.2f}% / "
                f"val {default_metrics['score'] * 100:.2f}% @0.5 -> "
                f"{cma_metrics['score'] * 100:.2f}% @tuned "
                f"(macro {cma_metrics['macro_f1'] * 100:.2f}%, "
                f"micro {cma_metrics['micro_f1'] * 100:.2f}%)",
                flush=True,
            )

            if float(cma_metrics["score"]) > best_val_score:
                best_val_score = float(cma_metrics["score"])
                best_thresholds = th
                best_val_metrics = cma_metrics
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        # Asynchronous by default: this returns without waiting on the network.
        wandb.log(log_dict)

    # Restore best model weights (epoch == epochs guarantees a validation pass)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final verification using the official TA evaluate.py script on disk
    _report_and_save(
        model=model,
        val_graph=val_graph,
        val_mask=val_mask,
        thresholds=best_thresholds,
        val_user_ids=val_user_ids,
        best_val_score=best_val_score,
        val_pred_path=val_pred_path,
        truth_path=truth_path,
        save_model_path=save_model_path,
    )

    return model, best_thresholds, best_val_metrics


def _report_and_save(
    *,
    model: GitHubHeteroHGT,
    val_graph,
    val_mask: np.ndarray,
    thresholds: np.ndarray,
    val_user_ids: np.ndarray,
    best_val_score: float,
    val_pred_path: str,
    truth_path: str,
    save_model_path: str | None,
) -> None:
    """Write the validation predictions, the checkpoint, and the TA cross-check."""
    pred_file = Path(val_pred_path)
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        final_probs = model.predict_proba(val_graph).cpu().numpy()[val_mask]
    final_binary = (final_probs >= thresholds).astype(np.int64)

    val_pred_df = pd.DataFrame(
        final_binary, columns=np.asarray(ACTIVITY_NAMES, dtype=object)
    )
    val_pred_df.insert(0, "UserID", val_user_ids)
    val_pred_df.to_csv(pred_file, index=False)

    if save_model_path:
        model_file = Path(save_model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "thresholds": thresholds.tolist(),
                "best_val_score": best_val_score,
            },
            model_file,
        )
        print(f"Saved best checkpoint to {model_file}")

    if Path(truth_path).exists():
        print(
            f"\nRunning official scripts/evaluate.py on {pred_file} "
            f"against {truth_path}:"
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--pred",
                str(pred_file),
                "--truth",
                truth_path,
            ],
            check=False,
        )
    else:
        print(f"\nSkipping evaluate.py: truth file {truth_path} not found")

    wandb.summary["best_val_score"] = best_val_score
    wandb.summary["best_thresholds"] = thresholds.tolist()
    wandb.finish()


def _to_device(graph, device: str) -> tuple[Any, bool]:
    """Move a HeteroData to `device` in place; report whether a move happened."""
    first = next(iter(graph.node_types))
    target = torch.device(device)
    if graph[first].x.device == target:
        return graph, False
    return graph.to(device), True


def train_model_multi_anchor(
    model: GitHubHeteroHGT,
    train_anchors: list[Anchor],
    val_anchor: Anchor,
    val_user_ids: np.ndarray,
    pos_weights: list[float],
    *,
    epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_interval: int = 5,
    grad_clip: float = 1.0,
    lr_schedule: str = "warm-restarts",
    lr_t0: int = 5,
    lr_t_mult: int = 2,
    cma_sigma0: float = 0.15,
    cma_popsize: int = 16,
    cma_maxfevals: int = 3000,
    project_name: str = "nycu-bigdata-asg2",
    run_name: str = "hgt-multi-anchor",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online",
    extra_config: dict[str, object] | None = None,
    truth_path: str = "splits/answer.csv",
    val_pred_path: str = "splits/val_best_pred.csv",
    save_model_path: str | None = None,
    shuffle_anchors: bool = True,
    anchor_seed: int = 42,
) -> tuple[GitHubHeteroHGT, np.ndarray, dict[str, object]]:
    """Train on many (cutoff, label-window) anchors, validating on one clean week.

    Each anchor is its own graph.  One *epoch* visits every train anchor once
    and applies one optimizer step on the sum of the anchors' gradients, so the
    learning-rate schedule still has meaning; the objective is the exact mean
    BCE over all *observed* (user, activity) entries, because anchor ``i``
    contributes weight ``n_observed_i / sum_j n_observed_j``.

    An activity the coverage audit marked unusable contributes no entries at
    all (see :class:`Anchor`), so a broken `push` day removes push labels from
    that anchor instead of asserting "nobody pushed".
    """
    if len(pos_weights) != len(ACTIVITY_NAMES):
        raise ValueError(
            f"pos_weights must have {len(ACTIVITY_NAMES)} entries, got {len(pos_weights)}"
        )
    if not train_anchors:
        raise ValueError("train_anchors is empty")
    missing = [a.name for a in train_anchors if a.n_observed == 0]
    if missing:
        raise ValueError(f"anchors with no observed label entries: {missing}")
    if not val_anchor.observed[val_anchor.mask].all():
        raise ValueError(
            f"validation anchor {val_anchor.name} has unobserved label cells; "
            "pick a fully usable week"
        )

    wandb.init(
        project=project_name,
        name=run_name,
        mode=wandb_mode,
        config={
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "pos_weights": pos_weights,
            "optimizer": "AdamW",
            **_scheduler_config(lr_schedule, epochs, lr, lr_t0, lr_t_mult),
            "grad_clip_max_norm": grad_clip,
            "threshold_optimizer": "CMA-ES",
            "cma_sigma0": cma_sigma0,
            "cma_popsize": cma_popsize,
            "cma_maxfevals": cma_maxfevals,
            "device": device,
            "n_anchors": len(train_anchors),
            "anchors": [str(a.cutoff.date()) for a in train_anchors],
            "anchor_observed_entries": [a.n_observed for a in train_anchors],
            **(extra_config or {}),
        },
        dir=WANDB_DIR,
    )

    model = model.to(device)
    val_graph = val_anchor.graph.to(device)
    val_mask = val_anchor.mask.astype(bool)
    val_labels = val_anchor.labels[val_mask]

    weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _build_scheduler(optimizer, lr_schedule, epochs, lr, lr_t0, lr_t_mult)

    shares = np.array([a.n_observed for a in train_anchors], dtype=np.float64)
    shares /= shares.sum()

    best_val_score = -1.0
    best_thresholds = np.full(len(ACTIVITY_NAMES), 0.5, dtype=np.float64)
    best_val_metrics: dict[str, object] = {}
    best_state_dict: dict[str, torch.Tensor] | None = None
    rng = np.random.default_rng(anchor_seed)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        order = (
            rng.permutation(len(train_anchors))
            if shuffle_anchors
            else np.arange(len(train_anchors))
        )
        epoch_loss = 0.0
        prob_parts: list[np.ndarray] = []
        truth_parts: list[np.ndarray] = []

        for slot in order:
            idx = int(slot)
            anchor = train_anchors[idx]
            graph, moved = _to_device(anchor.graph, device)
            row_mask = torch.from_numpy(anchor.mask).to(device=device, dtype=torch.bool)

            logits = model(graph)[row_mask]
            y, obs = _anchor_batch(anchor, device)
            loss = _masked_bce(logits, y, obs, weight_tensor)
            (loss * float(shares[idx])).backward()
            epoch_loss += float(loss.detach()) * float(shares[idx])

            # Only fully observed rows can enter a TA-style score, because the
            # metric has no notion of an unobserved cell.
            with torch.no_grad():
                full = anchor.observed[anchor.mask].all(axis=1)
                if full.any():
                    probs = torch.sigmoid(logits.detach()).cpu().numpy()
                    prob_parts.append(probs[full])
                    truth_parts.append(anchor.labels[anchor.mask][full])

            del logits, y, obs, loss
            if moved:
                anchor.graph = anchor.graph.to("cpu")

        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
            grad_norm_value = float(grad_norm)
        else:
            grad_norm_value = 0.0
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        log_dict: dict[str, object] = {
            "train/loss": epoch_loss,
            "train/grad_norm": grad_norm_value,
            "train/lr": _current_lr(scheduler, optimizer),
            "train/anchors": len(train_anchors),
            "epoch": epoch,
        }
        train_score = float("nan")
        if prob_parts:
            train_metrics = score_in_memory(
                np.concatenate(truth_parts),
                (np.concatenate(prob_parts) >= 0.5).astype(np.int64),
            )
            train_score = float(train_metrics["score"])
            log_dict.update(
                {
                    "train/score": train_metrics["score"],
                    "train/macro_f1": train_metrics["macro_f1"],
                    "train/micro_f1": train_metrics["micro_f1"],
                }
            )

        if epoch % val_interval == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_graph)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_loss = criterion(
                val_logits[val_mask],
                torch.as_tensor(val_labels, dtype=torch.float32, device=device),
            )
            val_probs_cohort = val_probs[val_mask]
            default_metrics = score_in_memory(
                val_labels, (val_probs_cohort >= 0.5).astype(np.int64)
            )

            # Thresholds are fitted on TRAINING rows only, then applied here.
            # The in-loop train probabilities carry dropout noise, so recompute
            # them in eval mode before fitting.
            model.eval()
            parts_p: list[np.ndarray] = []
            parts_y: list[np.ndarray] = []
            for anchor in train_anchors:
                graph, moved = _to_device(anchor.graph, device)
                row_mask = torch.from_numpy(anchor.mask).to(
                    device=device, dtype=torch.bool
                )
                with torch.no_grad():
                    anchor_logits = model(graph)[row_mask]
                    full = anchor.observed[anchor.mask].all(axis=1)
                    if full.any():
                        parts_p.append(torch.sigmoid(anchor_logits).cpu().numpy()[full])
                        parts_y.append(anchor.labels[anchor.mask][full])
                del anchor_logits
                if moved:
                    anchor.graph = anchor.graph.to("cpu")

            th, cma_metrics, fit_metrics = _fit_thresholds_on_train(
                train_probs=np.concatenate(parts_p),
                train_labels=np.concatenate(parts_y),
                val_probs=val_probs_cohort,
                val_labels=val_labels,
                sigma0=cma_sigma0,
                popsize=cma_popsize,
                maxfevals=cma_maxfevals,
            )

            log_dict.update(
                {
                    "val/loss": val_loss.item(),
                    "val/score_default_0.5": default_metrics["score"],
                    "val/score_cmaes": cma_metrics["score"],
                    "val/score_theta_on_fit_set": fit_metrics["score"],
                    "val/macro_f1": cma_metrics["macro_f1"],
                    "val/micro_f1": cma_metrics["micro_f1"],
                    "val/cma_generations": fit_metrics["n_generations"],
                    "val/cma_evaluations": fit_metrics["n_evaluations"],
                }
            )
            log_dict["val/score_cmaes_optimistic"] = (
                fit_metrics["score"] - cma_metrics["score"]
            )

            for i, act in enumerate(ACTIVITY_NAMES):
                log_dict[f"val_f1/{act}"] = cma_metrics["per_class_f1"][i]
                log_dict[f"val_threshold/{act}"] = float(th[i])

            print(
                f"Epoch {epoch:03d}/{epochs} | anchors {len(train_anchors)} | "
                f"loss train {epoch_loss:.4f} / val {val_loss.item():.4f} | "
                f"score train {train_score * 100:.2f}% / "
                f"val {default_metrics['score'] * 100:.2f}% @0.5 -> "
                f"{cma_metrics['score'] * 100:.2f}% @tuned "
                f"(macro {cma_metrics['macro_f1'] * 100:.2f}%, "
                f"micro {cma_metrics['micro_f1'] * 100:.2f}%)",
                flush=True,
            )

            if float(cma_metrics["score"]) > best_val_score:
                best_val_score = float(cma_metrics["score"])
                best_thresholds = th
                best_val_metrics = cma_metrics
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        wandb.log(log_dict)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    _report_and_save(
        model=model,
        val_graph=val_graph,
        val_mask=val_mask,
        thresholds=best_thresholds,
        val_user_ids=val_user_ids,
        best_val_score=best_val_score,
        val_pred_path=val_pred_path,
        truth_path=truth_path,
        save_model_path=save_model_path,
    )

    return model, best_thresholds, best_val_metrics
