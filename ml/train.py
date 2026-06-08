"""Reproducible training driver for the multi-task match-statistics predictor.

Run:
    uv run python -m ml.train               # build features if needed, then train
    uv run python -m ml.train --rebuild     # force-rebuild the feature table first

The script is fully seeded (Python, NumPy, PyTorch, cuDNN) so that a given
configuration reproduces bit-for-bit on the same hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn

from ml.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    HEAD_SLICES,
    NUM_CLASSES,
    NUM_REGRESSION_TARGETS,
    PROJECT_ROOT,
    REGRESSION_TARGETS,
    FeatureConfig,
    LossWeights,
    ModelConfig,
    MongoConfig,
    SplitConfig,
    TrainConfig,
)
from ml.data_loader import (
    MatchFeatureDataset,
    compute_class_weights,
    compute_normalization_stats,
    make_dataloader,
    resolve_split_ranges,
)
from ml.features import Vocabulary, build_feature_table, numeric_feature_names
from ml.model import MatchStatsMultiTaskPredictor, ModelDimensions

logger = logging.getLogger(__name__)

_VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
_SCALER_PATH = ARTIFACTS_DIR / "scaler.json"
_CHECKPOINT_PATH = ARTIFACTS_DIR / "model.pt"
_METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def set_seed(seed: int) -> None:
    """Seed every RNG and force deterministic cuDNN kernels.

    Reproducibility requires seeding Python's ``random`` (shuffle buffer), NumPy
    (scaler/array ops) and PyTorch (weight init, dropout). ``cudnn.deterministic``
    plus disabling ``benchmark`` removes the last source of run-to-run variance on
    GPU at a small throughput cost, the right trade-off for an auditable pipeline.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CompositeLoss:
    """Composite multi-task objective with masked regression terms.

    Total loss::

        L = L_outcome
            + lambda_sot     * L_sot
            + lambda_corners * L_corners
            + lambda_cards   * L_cards

    ``L_outcome`` is class-weighted cross-entropy on raw logits. Each regression
    term is a *masked* Huber loss: targets that are absent (no WhoScored
    counterpart, ``mask = 0``) must not contribute gradient. For a head with
    prediction ``p_i`` and target ``t_i`` over a batch, the masked loss is

        L_head = ( sum_i m_i * huber(p_i, t_i) ) / max(sum_i m_i, 1),

    i.e. the mean Huber over *present* samples only. Dividing by the present count
    (not the batch size) keeps the per-sample scale constant regardless of how
    many targets are masked, so the loss magnitude -- and thus the effective
    learning rate -- does not drift with target coverage. When a batch contains no
    present targets the term is exactly 0 (no NaN). Only the scalar total is
    returned for ``.backward()``; the per-task components are detached floats for
    logging.

    Args:
        class_weights: ``(NUM_CLASSES,)`` inverse-frequency weights for the
            cross-entropy term.
        loss_weights: Static lambda scalars and the Huber transition point.
    """

    def __init__(self, class_weights: torch.Tensor, loss_weights: LossWeights) -> None:
        self._ce = nn.CrossEntropyLoss(weight=class_weights)
        self._huber = nn.HuberLoss(delta=loss_weights.huber_delta, reduction="none")
        self._w = loss_weights

    def _masked_head_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean Huber loss over present samples for one (B, 2) regression head."""
        per_element = self._huber(prediction, target)  # (B, 2)
        per_sample = per_element.mean(dim=1)            # (B,)
        denominator = mask.sum().clamp_min(1.0)
        return (per_sample * mask).sum() / denominator

    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        outcome: torch.Tensor,
        regression: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_ce = self._ce(outputs["outcome"], outcome)
        sot_lo, sot_hi = HEAD_SLICES["sot"]
        cor_lo, cor_hi = HEAD_SLICES["corners"]
        car_lo, car_hi = HEAD_SLICES["cards"]
        loss_sot = self._masked_head_loss(outputs["sot"], regression[:, sot_lo:sot_hi], mask)
        loss_corners = self._masked_head_loss(outputs["corners"], regression[:, cor_lo:cor_hi], mask)
        loss_cards = self._masked_head_loss(outputs["cards"], regression[:, car_lo:car_hi], mask)

        total = (
            loss_ce
            + self._w.lambda_sot * loss_sot
            + self._w.lambda_corners * loss_corners
            + self._w.lambda_cards * loss_cards
        )
        components = {
            "ce": float(loss_ce.item()),
            "sot": float(loss_sot.item()),
            "corners": float(loss_corners.item()),
            "cards": float(loss_cards.item()),
        }
        return total, components


def _ensure_feature_table(
    mongo: MongoConfig,
    feature_cfg: FeatureConfig,
    rebuild: bool,
) -> Tuple[Vocabulary, List[str], int]:
    """Build the feature table if missing/forced, else load cached metadata."""
    needs_build = rebuild or not (_VOCAB_PATH.exists() and _METADATA_PATH.exists())
    if needs_build:
        logger.info("Building feature table (rebuild=%s).", rebuild)
        vocab, feature_names, n_rows = build_feature_table(mongo, feature_cfg)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        _VOCAB_PATH.write_text(json.dumps(vocab.to_dict()), encoding="utf-8")
        _METADATA_PATH.write_text(
            json.dumps({"n_rows": n_rows, "feature_names": feature_names}),
            encoding="utf-8",
        )
        return vocab, feature_names, n_rows

    logger.info("Loading cached vocabulary and metadata.")
    vocab = Vocabulary.from_dict(json.loads(_VOCAB_PATH.read_text(encoding="utf-8")))
    meta = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    return vocab, meta["feature_names"], int(meta["n_rows"])


def _confusion_counts(logits: torch.Tensor, labels: torch.Tensor, matrix: np.ndarray) -> None:
    """Accumulate outcome predictions into a confusion matrix in place."""
    preds = logits.argmax(dim=1)
    for true, pred in zip(labels.tolist(), preds.tolist()):
        matrix[true, pred] += 1


def _metrics_from_confusion(matrix: np.ndarray) -> Dict[str, float]:
    """Derive accuracy and macro precision/recall/F1 from a confusion matrix.

    Macro-F1 weights every class equally, so (unlike raw accuracy) it penalises a
    model that ignores the minority "draw" class. It is the primary selection
    metric for the imbalanced 1X2 task.
    """
    total = matrix.sum()
    accuracy = float(np.trace(matrix)) / total if total > 0 else 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls])
        fp = float(matrix[:, cls].sum() - tp)
        fn = float(matrix[cls, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def _run_epoch(
    model: MatchStatsMultiTaskPredictor,
    loader: torch.utils.data.DataLoader,
    criterion: CompositeLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Run one pass over ``loader``. Trains if ``optimizer`` is given, else evaluates.

    Returns the average total loss, a metrics dict (outcome classification metrics,
    per-task loss components, and masked MAE per regression target) and the outcome
    confusion matrix.
    """
    is_train = optimizer is not None
    model.train(is_train)

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loss_sum = 0.0
    seen = 0
    component_sums = {"ce": 0.0, "sot": 0.0, "corners": 0.0, "cards": 0.0}
    abs_error = np.zeros(NUM_REGRESSION_TARGETS, dtype=np.float64)
    present_count = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for numeric, categorical, outcome, regression, mask in loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            outcome = outcome.to(device)
            regression = regression.to(device)
            mask = mask.to(device)

            outputs = model(numeric, categorical)
            total, components = criterion(outputs, outcome, regression, mask)

            if optimizer is not None:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

            batch_n = outcome.size(0)
            loss_sum += float(total.item()) * batch_n
            seen += batch_n
            for key, value in components.items():
                component_sums[key] += value * batch_n
            _confusion_counts(outputs["outcome"].detach().cpu(), outcome.cpu(), matrix)

            # Masked MAE accumulation over present regression targets only.
            predicted = torch.cat([outputs["sot"], outputs["corners"], outputs["cards"]], dim=1)
            masked_abs = (predicted - regression).abs() * mask.unsqueeze(1)
            abs_error += masked_abs.sum(dim=0).detach().cpu().numpy()
            present_count += float(mask.sum().item())

    avg_loss = loss_sum / seen if seen > 0 else 0.0
    metrics = _metrics_from_confusion(matrix)
    for key, value in component_sums.items():
        metrics[f"loss_{key}"] = value / seen if seen > 0 else 0.0

    denominator = max(present_count, 1.0)
    per_target_mae = abs_error / denominator
    for name, value in zip(REGRESSION_TARGETS, per_target_mae):
        metrics[f"mae_{name}"] = float(value)
    # Head-level MAE: average of the home and away columns within each head.
    for head, (lo, hi) in HEAD_SLICES.items():
        metrics[f"mae_{head}"] = float(np.mean(per_target_mae[lo:hi]))
    return avg_loss, metrics, matrix


def train(args: argparse.Namespace) -> None:
    """End-to-end training entry point."""
    load_dotenv(PROJECT_ROOT / ".env")

    mongo = MongoConfig()
    feature_cfg = FeatureConfig()
    split_cfg = SplitConfig()
    model_cfg = ModelConfig()
    loss_weights = LossWeights()
    train_cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    vocab, feature_names, n_rows = _ensure_feature_table(mongo, feature_cfg, args.rebuild)
    if feature_names != numeric_feature_names():
        logger.warning("Cached feature names differ from current code; consider --rebuild.")
    logger.info("Feature table: %d rows, %d numeric features.", n_rows, len(feature_names))

    ranges = resolve_split_ranges(n_rows, split_cfg.train_fraction, split_cfg.val_fraction)
    logger.info(
        "Temporal split (row_idx): train=%s val=%s test=%s",
        ranges["train"], ranges["val"], ranges["test"],
    )

    # Scaler and class weights are fit on the TRAIN range only (no leakage).
    mean, std = compute_normalization_stats(mongo, feature_names, ranges["train"])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _SCALER_PATH.write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist(), "feature_names": feature_names}),
        encoding="utf-8",
    )
    class_weights = compute_class_weights(mongo, ranges["train"]).to(device)
    logger.info("Class weights (home/draw/away): %s", class_weights.tolist())

    train_ds = MatchFeatureDataset(
        mongo, feature_names, ranges["train"], mean, std,
        shuffle_buffer=train_cfg.shuffle_buffer, seed=train_cfg.seed,
        page_size=train_cfg.page_size,
    )
    val_ds = MatchFeatureDataset(
        mongo, feature_names, ranges["val"], mean, std,
        shuffle_buffer=0, seed=train_cfg.seed, page_size=train_cfg.page_size,
    )
    test_ds = MatchFeatureDataset(
        mongo, feature_names, ranges["test"], mean, std,
        shuffle_buffer=0, seed=train_cfg.seed, page_size=train_cfg.page_size,
    )

    train_loader = make_dataloader(train_ds, train_cfg.batch_size, train_cfg.num_workers_train)
    val_loader = make_dataloader(val_ds, train_cfg.batch_size, train_cfg.num_workers_eval)
    test_loader = make_dataloader(test_ds, train_cfg.batch_size, train_cfg.num_workers_eval)

    dims = ModelDimensions(
        num_teams=vocab.num_teams,
        num_leagues=vocab.num_leagues,
        num_seasons=vocab.num_seasons,
        num_numeric_features=len(feature_names),
    )
    model = MatchStatsMultiTaskPredictor(
        dims=dims,
        team_embedding_dim=model_cfg.team_embedding_dim,
        league_embedding_dim=model_cfg.league_embedding_dim,
        season_embedding_dim=model_cfg.season_embedding_dim,
        hidden_dims=model_cfg.hidden_dims,
        dropout=model_cfg.dropout,
    ).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # AdamW decouples weight decay from the adaptive step, giving cleaner L2
    # regularisation than classic Adam.
    criterion = CompositeLoss(class_weights, loss_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_metrics, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics, _ = _run_epoch(model, val_loader, criterion, device)

        logger.info(
            "Epoch %03d | train loss %.4f f1 %.3f | val loss %.4f f1 %.3f acc %.3f | "
            "val MAE sot %.2f cor %.2f card %.2f",
            epoch, train_loss, train_metrics["macro_f1"],
            val_loss, val_metrics["macro_f1"], val_metrics["accuracy"],
            val_metrics["mae_sot"], val_metrics["mae_corners"], val_metrics["mae_cards"],
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_class": "MatchStatsMultiTaskPredictor",
                    "model_state": best_state,
                    "dims": dims.__dict__,
                    "model_cfg": model_cfg.__dict__,
                    "loss_weights": loss_weights.__dict__,
                },
                _CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d (no val improvement).", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics, test_matrix = _run_epoch(model, test_loader, criterion, device)
    logger.info("=" * 70)
    logger.info("TEST  total loss %.4f", test_loss)
    logger.info(
        "TEST  outcome: accuracy %.3f | macro-P %.3f macro-R %.3f macro-F1 %.3f",
        test_metrics["accuracy"], test_metrics["macro_precision"],
        test_metrics["macro_recall"], test_metrics["macro_f1"],
    )
    logger.info("TEST  regression MAE (lower is better):")
    for name in REGRESSION_TARGETS:
        logger.info("  %-14s %.3f", name, test_metrics[f"mae_{name}"])
    logger.info("TEST  confusion matrix (rows=true, cols=pred) %s:", CLASS_NAMES)
    for cls, row in zip(CLASS_NAMES, test_matrix.tolist()):
        logger.info("  %-9s %s", cls, row)
    logger.info("Best checkpoint saved to %s", _CHECKPOINT_PATH)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-task match-statistics predictor.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the feature table before training.")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    train(_parse_args())


if __name__ == "__main__":
    main()
