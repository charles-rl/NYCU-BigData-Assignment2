"""Training pipeline with Weights & Biases logging and CMA-ES threshold optimization.

Validation scoring is imported from :mod:`src.utils.metrics`, an in-memory
replica of the TA ``scripts/evaluate.py`` scorer.  Threshold search lives in
:mod:`src.utils.threshold_optimizer`.
"""

from __future__ import annotations

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
from torch.optim.lr_scheduler import CosineAnnealingLR

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
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": epochs,
            "grad_clip_max_norm": grad_clip,
            "threshold_optimizer": "CMA-ES",
            "cma_sigma0": cma_sigma0,
            "cma_popsize": cma_popsize,
            "cma_maxfevals": cma_maxfevals,
            "device": device,
            **(extra_config or {}),
        },
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
    # One scheduler step per epoch for `epochs` epochs: cosine decays lr from
    # `lr` to `eta_min` over the whole run.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

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
        scheduler.step()

        log_dict: dict[str, object] = {
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": scheduler.get_last_lr()[0],
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

            # Joint threshold optimization on the aligned cohort rows
            th, cma_metrics = optimize_thresholds_cmaes(
                val_labels,
                val_probs_cohort,
                sigma0=cma_sigma0,
                popsize=cma_popsize,
                maxfevals=cma_maxfevals,
            )

            log_dict.update(
                {
                    "val/loss": val_loss.item(),
                    "val/score_default_0.5": default_metrics["score"],
                    "val/score_cmaes": cma_metrics["score"],
                    "val/score_cmaes_initial": cma_metrics["initial_score"],
                    "val/macro_f1": cma_metrics["macro_f1"],
                    "val/micro_f1": cma_metrics["micro_f1"],
                    "val/cma_generations": cma_metrics["n_generations"],
                    "val/cma_evaluations": cma_metrics["n_evaluations"],
                }
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
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": epochs,
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
    )

    model = model.to(device)
    val_graph = val_anchor.graph.to(device)
    val_mask = val_anchor.mask.astype(bool)
    val_labels = val_anchor.labels[val_mask]

    weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

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
        scheduler.step()

        log_dict: dict[str, object] = {
            "train/loss": epoch_loss,
            "train/grad_norm": grad_norm_value,
            "train/lr": scheduler.get_last_lr()[0],
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
            th, cma_metrics = optimize_thresholds_cmaes(
                val_labels,
                val_probs_cohort,
                sigma0=cma_sigma0,
                popsize=cma_popsize,
                maxfevals=cma_maxfevals,
            )

            log_dict.update(
                {
                    "val/loss": val_loss.item(),
                    "val/score_default_0.5": default_metrics["score"],
                    "val/score_cmaes": cma_metrics["score"],
                    "val/score_cmaes_initial": cma_metrics["initial_score"],
                    "val/macro_f1": cma_metrics["macro_f1"],
                    "val/micro_f1": cma_metrics["micro_f1"],
                    "val/cma_generations": cma_metrics["n_generations"],
                    "val/cma_evaluations": cma_metrics["n_evaluations"],
                }
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
