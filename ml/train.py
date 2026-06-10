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
from torch.nn import functional as F

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
from ml.calibration import apply_temperature, fit_temperature
from ml.data_loader import (
    MatchFeatureDataset,
    compute_class_weights,
    compute_max_date,
    compute_normalization_stats,
    make_dataloader,
    resolve_split_ranges,
)
from ml.features import LABEL_DRAW, Vocabulary, build_feature_table, numeric_feature_names
from ml.model import (
    MatchStatsMultiTaskPredictor,
    ModelDimensions,
    poisson_outcome_probs,
)

logger = logging.getLogger(__name__)

_VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
_SCALER_PATH = ARTIFACTS_DIR / "scaler.json"
_CHECKPOINT_PATH = ARTIFACTS_DIR / "model.pt"
_METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
_METRICS_HISTORY_PATH = ARTIFACTS_DIR / "metrics_history.jsonl"


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
    """Composite multi-task objective built on a Poisson goal model.

    Total loss::

        L = L_goals
            + lambda_outcome * L_outcome
            + lambda_sot     * L_sot
            + lambda_corners * L_corners
            + lambda_cards   * L_cards

    ``L_goals`` is the Poisson negative log-likelihood of the observed home/away
    goal counts given the predicted rates (``input - target*log(input)``); it is
    the primary objective and anchors the scale. The 1X2 outcome is *derived* from
    those rates (:func:`ml.model.poisson_outcome_probs`), and ``L_outcome`` is a
    class-weighted negative log-likelihood on that derived simplex. This is the
    explicit draw-recovery term: the Poisson diagonal already lends the draw a
    real mechanism, and the inverse-frequency class weights counter the residual
    under-prediction of draws inherent to an independent (correlation-free)
    Poisson. Goals are present for every match, so ``L_goals`` and ``L_outcome``
    are unmasked.

    Each regression term is a *masked* Huber loss: WhoScored-derived targets that
    are absent (``mask = 0``) must not contribute gradient. For a head with
    prediction ``p_i`` and target ``t_i``,

        L_head = ( sum_i m_i * huber(p_i, t_i) ) / max(sum_i m_i, 1),

    the mean Huber over *present* samples only. Dividing by the present count (not
    the batch size) keeps the per-sample scale constant regardless of coverage, so
    the loss magnitude -- and thus the effective learning rate -- does not drift.
    When a batch has no present targets the term is exactly 0 (no NaN). Only the
    scalar total is returned for ``.backward()``; the per-task components are
    detached floats for logging.

    **Dixon-Coles time decay.** When a per-row ``date_days`` tensor is supplied,
    the goal NLL is weighted by ``0.5 ** ((t_ref - t_match) / half_life)`` --
    the exponential down-weighting of stale matches from Dixon & Coles (1997).
    Recent matches then dominate the rate estimates, mirroring the recency bias
    the EMA features already encode but at the *likelihood* level. Weights are
    renormalised to mean 1 per batch so the loss scale (and the effective
    learning rate) is unchanged. Evaluation passes ``date_days=None``, keeping
    validation/test losses unweighted and comparable across configurations.

    Args:
        class_weights: ``(NUM_CLASSES,)`` inverse-frequency weights for the derived
            outcome cross-entropy.
        loss_weights: Static lambda scalars and the Huber transition point.
        time_decay_half_life_days: Half-life of the goal-NLL recency weight;
            ``<= 0`` disables the decay entirely.
        reference_day: ``t_ref`` as proleptic ordinal days (latest train-range
            date). Required when the decay is enabled.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        loss_weights: LossWeights,
        time_decay_half_life_days: float = 0.0,
        reference_day: Optional[float] = None,
    ) -> None:
        self._poisson = nn.PoissonNLLLoss(log_input=False, full=False, reduction="none")
        self._class_weights = class_weights
        self._huber = nn.HuberLoss(delta=loss_weights.huber_delta, reduction="none")
        self._w = loss_weights
        self._half_life = time_decay_half_life_days
        self._reference_day = reference_day

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
        goals: torch.Tensor,
        regression: torch.Tensor,
        mask: torch.Tensor,
        date_days: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        per_sample_goals = self._poisson(outputs["goals"], goals).mean(dim=1)  # (B,)
        if date_days is not None and self._half_life > 0 and self._reference_day is not None:
            age_days = (self._reference_day - date_days).clamp_min(0.0)
            decay = torch.pow(0.5, age_days / self._half_life)
            decay = decay * (decay.numel() / decay.sum().clamp_min(1e-8))
            loss_goals = (per_sample_goals * decay).mean()
        else:
            loss_goals = per_sample_goals.mean()

        probs = poisson_outcome_probs(outputs["goals"], rho=outputs.get("rho"))
        loss_outcome = F.nll_loss(
            torch.log(probs.clamp_min(1e-9)), outcome, weight=self._class_weights
        )

        sot_lo, sot_hi = HEAD_SLICES["sot"]
        cor_lo, cor_hi = HEAD_SLICES["corners"]
        car_lo, car_hi = HEAD_SLICES["cards"]
        loss_sot = self._masked_head_loss(outputs["sot"], regression[:, sot_lo:sot_hi], mask)
        loss_corners = self._masked_head_loss(outputs["corners"], regression[:, cor_lo:cor_hi], mask)
        loss_cards = self._masked_head_loss(outputs["cards"], regression[:, car_lo:car_hi], mask)

        total = (
            loss_goals
            + self._w.lambda_outcome * loss_outcome
            + self._w.lambda_sot * loss_sot
            + self._w.lambda_corners * loss_corners
            + self._w.lambda_cards * loss_cards
        )
        components = {
            "goals": float(loss_goals.item()),
            "outcome": float(loss_outcome.item()),
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


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true class (multiclass log-loss).

    The proper scoring rule for probability quality: unlike argmax accuracy it
    rewards calibrated confidence, so it is the right scalar for comparing the
    derived 1X2 simplex across model variants.
    """
    true_probs = np.clip(probs[np.arange(labels.shape[0]), labels], eps, 1.0)
    return float(-np.log(true_probs).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared distance to the one-hot target."""
    onehot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    return float(np.square(probs - onehot).sum(axis=1).mean())


def ranked_probability_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean RPS over the ordered 1X2 classes (home win < draw < away win).

    The RPS compares cumulative distributions, so predicting a home win when the
    away side wins costs more than when the match is drawn. Log-loss and Brier
    ignore this ordinal structure of the 1X2 market; RPS is its standard metric.
    """
    onehot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    cum_diff = np.cumsum(probs, axis=1) - np.cumsum(onehot, axis=1)
    return float((np.square(cum_diff).sum(axis=1) / (probs.shape[1] - 1)).mean())


def reliability_table(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> List[Dict[str, float]]:
    """Bin predictions by top-class confidence against empirical accuracy.

    A calibrated model has ``mean_confidence ~= accuracy`` within each bin; the
    gap profile localises over/under-confidence (e.g. an overconfident favourite
    bias) that a single scalar like log-loss cannot show.
    """
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence >= lo) & ((confidence < hi) if hi < 1.0 else (confidence <= hi))
        count = int(in_bin.sum())
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": float(count),
                "mean_confidence": float(confidence[in_bin].mean()) if count else 0.0,
                "accuracy": float(correct[in_bin].mean()) if count else 0.0,
            }
        )
    return rows


def probability_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Bundle the proper-scoring metrics for one (probs, labels) split."""
    return {
        "log_loss": log_loss(probs, labels),
        "brier": brier_score(probs, labels),
        "rps": ranked_probability_score(probs, labels),
    }


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


def _collect_outcome_probs(
    model: MatchStatsMultiTaskPredictor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, labels)`` of the Poisson-derived 1X2 over a loader."""
    model.eval()
    probs_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for numeric, categorical, outcome, _goals, _regression, _mask, _date_days in loader:
            outputs = model(numeric.to(device), categorical.to(device))
            probs = poisson_outcome_probs(outputs["goals"], rho=outputs.get("rho"))
            probs_chunks.append(probs.cpu().numpy())
            label_chunks.append(outcome.numpy())
    return np.concatenate(probs_chunks), np.concatenate(label_chunks)


def _confusion_with_draw_boost(
    probs: np.ndarray, labels: np.ndarray, draw_boost: float
) -> np.ndarray:
    """Confusion matrix after scaling the draw probability before the argmax."""
    adjusted = probs.copy()
    adjusted[:, LABEL_DRAW] *= draw_boost
    preds = adjusted.argmax(axis=1)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true, pred in zip(labels.tolist(), preds.tolist()):
        matrix[true, pred] += 1
    return matrix


def tune_draw_boost(
    probs: np.ndarray,
    labels: np.ndarray,
    grid: Optional[np.ndarray] = None,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Pick the draw-probability multiplier that maximises a metric on a split.

    Independent/Dixon-Coles Poisson yields well-shaped probabilities, but the
    plain ``argmax`` almost never returns a draw because P(draw) rarely exceeds
    both P(home) and P(away). Scaling P(draw) by a constant before the argmax is a
    cost-sensitive decision rule (equivalent to a class-specific decision
    threshold). Tuned for ``"f1"`` it trades raw accuracy for balanced macro-F1;
    tuned for ``"accuracy"`` it almost always settles near 1.0 (no boost), which
    is exactly the honest answer for an accuracy-primary objective. The
    multiplier is selected on the validation split only -- never on test -- so
    the reported test figure remains an honest out-of-sample estimate.

    Args:
        probs: ``(N, 3)`` outcome probabilities.
        labels: ``(N,)`` true class indices.
        grid: Candidate multipliers (defaults to ``0.5 .. 3.0`` step ``0.05``;
            sub-1.0 values allow draw *deflation* when that helps the metric).
        metric: ``"f1"`` (macro-F1) or ``"accuracy"``.

    Returns:
        ``(best_boost, best_score)``.
    """
    if metric not in ("f1", "accuracy"):
        raise ValueError(f"Unsupported decision metric: {metric!r}")
    key = "macro_f1" if metric == "f1" else "accuracy"
    if grid is None:
        grid = np.round(np.arange(0.5, 3.001, 0.05), 2)
    best_boost, best_score = 1.0, -1.0
    for boost in grid:
        matrix = _confusion_with_draw_boost(probs, labels, float(boost))
        score = _metrics_from_confusion(matrix)[key]
        if score > best_score:
            best_score, best_boost = score, float(boost)
    return best_boost, best_score


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
    component_sums = {"goals": 0.0, "outcome": 0.0, "sot": 0.0, "corners": 0.0, "cards": 0.0}
    abs_error = np.zeros(NUM_REGRESSION_TARGETS, dtype=np.float64)
    goal_abs_error = np.zeros(2, dtype=np.float64)  # home/away expected-goals MAE
    present_count = 0.0
    outcome_nll_sum = 0.0  # plain (class-unweighted) NLL of the derived 1X2

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for numeric, categorical, outcome, goals, regression, mask, date_days in loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            outcome = outcome.to(device)
            goals = goals.to(device)
            regression = regression.to(device)
            mask = mask.to(device)

            outputs = model(numeric, categorical)
            # Time decay applies to the optimised objective only; evaluation
            # losses stay unweighted so epochs/configurations remain comparable.
            total, components = criterion(
                outputs, outcome, goals, regression, mask,
                date_days=date_days.to(device) if is_train else None,
            )

            if optimizer is not None:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

            batch_n = outcome.size(0)
            loss_sum += float(total.item()) * batch_n
            seen += batch_n
            for key, value in components.items():
                component_sums[key] += value * batch_n

            # Outcome metrics use the Poisson-derived 1X2 distribution.
            probs = poisson_outcome_probs(outputs["goals"], rho=outputs.get("rho"))
            _confusion_counts(probs.detach().cpu(), outcome.cpu(), matrix)
            outcome_nll_sum += float(
                F.nll_loss(
                    torch.log(probs.detach().clamp_min(1e-9)), outcome, reduction="sum"
                ).item()
            )

            # Masked MAE accumulation over present regression targets only.
            predicted = torch.cat([outputs["sot"], outputs["corners"], outputs["cards"]], dim=1)
            masked_abs = (predicted - regression).abs() * mask.unsqueeze(1)
            abs_error += masked_abs.sum(dim=0).detach().cpu().numpy()
            present_count += float(mask.sum().item())

            # Expected-goals MAE (always present, no mask).
            goal_abs_error += (outputs["goals"] - goals).abs().sum(dim=0).detach().cpu().numpy()

    avg_loss = loss_sum / seen if seen > 0 else 0.0
    metrics = _metrics_from_confusion(matrix)
    metrics["outcome_nll"] = outcome_nll_sum / seen if seen > 0 else 0.0
    for key, value in component_sums.items():
        metrics[f"loss_{key}"] = value / seen if seen > 0 else 0.0

    goal_mae = goal_abs_error / seen if seen > 0 else goal_abs_error
    metrics["mae_home_goals"] = float(goal_mae[0])
    metrics["mae_away_goals"] = float(goal_mae[1])
    metrics["mae_goals"] = float(np.mean(goal_mae))

    denominator = max(present_count, 1.0)
    per_target_mae = abs_error / denominator
    for name, value in zip(REGRESSION_TARGETS, per_target_mae):
        metrics[f"mae_{name}"] = float(value)
    # Head-level MAE: average of the home and away columns within each head.
    for head, (lo, hi) in HEAD_SLICES.items():
        metrics[f"mae_{head}"] = float(np.mean(per_target_mae[lo:hi]))
    return avg_loss, metrics, matrix


def _build_model(dims: ModelDimensions, model_cfg: ModelConfig) -> MatchStatsMultiTaskPredictor:
    """Construct the predictor from config (one place for all member builds)."""
    return MatchStatsMultiTaskPredictor(
        dims=dims,
        team_embedding_dim=model_cfg.team_embedding_dim,
        league_embedding_dim=model_cfg.league_embedding_dim,
        season_embedding_dim=model_cfg.season_embedding_dim,
        hidden_dims=model_cfg.hidden_dims,
        dropout=model_cfg.dropout,
    )


def _fit_member(
    seed: int,
    mongo: MongoConfig,
    feature_names: List[str],
    ranges: Dict[str, Tuple[int, int]],
    mean: np.ndarray,
    std: np.ndarray,
    criterion: "CompositeLoss",
    dims: ModelDimensions,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
) -> Dict[str, torch.Tensor]:
    """Train one (ensemble) member from its own seed; return its best state.

    Each member gets its own RNG seed -- weight init, dropout masks and the
    shuffle order all differ -- which is exactly the decorrelation a seed
    ensemble averages over. Model selection follows the validation outcome
    log-loss (plain NLL of the derived 1X2), not the composite loss: the
    composite is contaminated by the auxiliary regression lambdas, whereas the
    outcome NLL is the smooth surrogate of the primary objective.
    """
    set_seed(seed)
    train_ds = MatchFeatureDataset(
        mongo, feature_names, ranges["train"], mean, std,
        shuffle_buffer=train_cfg.shuffle_buffer, seed=seed,
        page_size=train_cfg.page_size,
    )
    train_loader = make_dataloader(train_ds, train_cfg.batch_size, train_cfg.num_workers_train)

    model = _build_model(dims, model_cfg).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # AdamW decouples weight decay from the adaptive step, giving cleaner L2
    # regularisation than classic Adam.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    best_val_nll = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_metrics, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics, _ = _run_epoch(model, val_loader, criterion, device)

        logger.info(
            "Epoch %03d | train loss %.4f f1 %.3f | val loss %.4f nll %.4f f1 %.3f acc %.3f | "
            "val MAE goals %.2f sot %.2f cor %.2f card %.2f",
            epoch, train_loss, train_metrics["macro_f1"],
            val_loss, val_metrics["outcome_nll"], val_metrics["macro_f1"],
            val_metrics["accuracy"], val_metrics["mae_goals"], val_metrics["mae_sot"],
            val_metrics["mae_corners"], val_metrics["mae_cards"],
        )

        if val_metrics["outcome_nll"] < best_val_nll - 1e-5:
            best_val_nll = val_metrics["outcome_nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d (no val outcome-NLL improvement).", epoch)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state


def _ensemble_probs(
    states: List[Dict[str, torch.Tensor]],
    loader: torch.utils.data.DataLoader,
    dims: ModelDimensions,
    model_cfg: ModelConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average the Poisson-derived 1X2 probabilities of every member.

    Averaging probabilities (not parameters) makes the ensemble a mixture of
    Dixon-Coles grids -- the correct combination rule for calibrated simplexes.
    """
    model = _build_model(dims, model_cfg).to(device)
    probs_sum: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    for state in states:
        model.load_state_dict(state)
        member_probs, labels = _collect_outcome_probs(model, loader, device)
        probs_sum = member_probs if probs_sum is None else probs_sum + member_probs
    assert probs_sum is not None and labels is not None
    return probs_sum / float(len(states)), labels


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
    class_weights = compute_class_weights(mongo, ranges["train"])
    # Exponent-shaped rebalancing: w^p renormalised to mean 1 spans uniform CE
    # (p=0) to full inverse-frequency weighting (p=1).
    class_weights = torch.pow(class_weights, train_cfg.class_weight_power)
    class_weights = (class_weights / class_weights.mean()).to(device)
    logger.info(
        "Class weights (home/draw/away, power %.2f): %s",
        train_cfg.class_weight_power, class_weights.tolist(),
    )

    val_ds = MatchFeatureDataset(
        mongo, feature_names, ranges["val"], mean, std,
        shuffle_buffer=0, seed=train_cfg.seed, page_size=train_cfg.page_size,
    )
    test_ds = MatchFeatureDataset(
        mongo, feature_names, ranges["test"], mean, std,
        shuffle_buffer=0, seed=train_cfg.seed, page_size=train_cfg.page_size,
    )
    val_loader = make_dataloader(val_ds, train_cfg.batch_size, train_cfg.num_workers_eval)
    test_loader = make_dataloader(test_ds, train_cfg.batch_size, train_cfg.num_workers_eval)

    dims = ModelDimensions(
        num_teams=vocab.num_teams,
        num_leagues=vocab.num_leagues,
        num_seasons=vocab.num_seasons,
        num_numeric_features=len(feature_names),
    )
    reference_day = compute_max_date(mongo, ranges["train"])
    criterion = CompositeLoss(
        class_weights,
        loss_weights,
        time_decay_half_life_days=train_cfg.time_decay_half_life_days,
        reference_day=reference_day,
    )

    # Seed ensemble: K independently initialised/shuffled members whose derived
    # 1X2 probabilities are averaged (a mixture of Poisson-grid models). The
    # averaging cancels seed-level variance and improves calibration; K=1 is the
    # plain single-model path.
    ensemble_size = max(1, int(getattr(args, "ensemble", 1)))
    member_seeds = [train_cfg.seed + offset for offset in range(ensemble_size)]
    states: List[Dict[str, torch.Tensor]] = []
    for member_index, member_seed in enumerate(member_seeds, start=1):
        logger.info("--- Ensemble member %d/%d (seed %d) ---",
                    member_index, ensemble_size, member_seed)
        states.append(
            _fit_member(
                member_seed, mongo, feature_names, ranges, mean, std,
                criterion, dims, model_cfg, train_cfg, device, val_loader,
            )
        )

    model = _build_model(dims, model_cfg).to(device)
    model.load_state_dict(states[0])

    # Decision layer, fit on VALIDATION only (never test): first calibrate the
    # (ensemble-averaged) probabilities with temperature scaling
    # (argmax-invariant), then tune the draw boost on the calibrated simplex for
    # both objectives. The accuracy-tuned boost is the default at inference; the
    # F1-tuned boost stays available for balanced-recall use cases.
    val_probs, val_labels = _ensemble_probs(states, val_loader, dims, model_cfg, device)
    # Robustness guards: a decision-layer parameter is adopted only when its
    # validation gain clears a significance margin. Argmax of calibrated
    # probabilities is already the Bayes-optimal accuracy rule, so a tuned
    # accuracy boost (or a temperature) that wins by a hair on validation is
    # overwhelmingly noise and tends to LOSE on test.
    temperature = fit_temperature(val_probs, val_labels)
    if log_loss(val_probs, val_labels) - log_loss(
        apply_temperature(val_probs, temperature), val_labels
    ) < 0.005:
        temperature = 1.0
    val_probs_cal = apply_temperature(val_probs, temperature)
    val_argmax_on_cal = _metrics_from_confusion(
        _confusion_with_draw_boost(val_probs_cal, val_labels, 1.0)
    )
    draw_boost_acc, val_acc_boosted = tune_draw_boost(val_probs_cal, val_labels, metric="accuracy")
    if val_acc_boosted < val_argmax_on_cal["accuracy"] + 0.01:
        draw_boost_acc, val_acc_boosted = 1.0, val_argmax_on_cal["accuracy"]
    draw_boost_f1, val_f1_boosted = tune_draw_boost(val_probs_cal, val_labels, metric="f1")
    val_f1_argmax = val_argmax_on_cal["macro_f1"]
    logger.info(
        "Calibration: temperature %.2f (val log-loss %.4f -> %.4f)",
        temperature, log_loss(val_probs, val_labels), log_loss(val_probs_cal, val_labels),
    )
    logger.info(
        "Decision boosts on val: accuracy x%.2f (acc %.3f) | f1 x%.2f (macro-F1 %.3f argmax -> %.3f)",
        draw_boost_acc, val_acc_boosted, draw_boost_f1, val_f1_argmax, val_f1_boosted,
    )

    torch.save(
        {
            "model_class": "MatchStatsMultiTaskPredictor",
            "model_state": states[0],
            # All member states; inference averages their derived probabilities.
            "model_states": states,
            "dims": dims.__dict__,
            "model_cfg": model_cfg.__dict__,
            "loss_weights": loss_weights.__dict__,
            "temperature": temperature,
            "draw_boost_acc": draw_boost_acc,
            "draw_boost_f1": draw_boost_f1,
            # Legacy key kept for older consumers; matches the F1 boost.
            "draw_boost": draw_boost_f1,
        },
        _CHECKPOINT_PATH,
    )

    # Regression MAEs / composite loss are reported from member 0; all outcome
    # metrics use the ensemble-averaged probabilities.
    test_loss, test_metrics, _ = _run_epoch(model, test_loader, criterion, device)
    test_probs, test_labels = _ensemble_probs(states, test_loader, dims, model_cfg, device)
    test_matrix = _confusion_with_draw_boost(test_probs, test_labels, 1.0)
    test_argmax_metrics = _metrics_from_confusion(test_matrix)
    test_probs_cal = apply_temperature(test_probs, temperature)
    boosted_acc_matrix = _confusion_with_draw_boost(test_probs_cal, test_labels, draw_boost_acc)
    boosted_acc_metrics = _metrics_from_confusion(boosted_acc_matrix)
    boosted_f1_matrix = _confusion_with_draw_boost(test_probs_cal, test_labels, draw_boost_f1)
    boosted_f1_metrics = _metrics_from_confusion(boosted_f1_matrix)

    logger.info("=" * 70)
    logger.info("TEST  total loss %.4f (member 0) | ensemble size %d", test_loss, ensemble_size)
    logger.info(
        "TEST  outcome (argmax, headline): accuracy %.3f | macro-F1 %.3f",
        test_argmax_metrics["accuracy"], test_argmax_metrics["macro_f1"],
    )
    logger.info(
        "TEST  outcome (acc boost x%.2f):  accuracy %.3f | macro-F1 %.3f",
        draw_boost_acc, boosted_acc_metrics["accuracy"], boosted_acc_metrics["macro_f1"],
    )
    logger.info(
        "TEST  outcome (f1 boost x%.2f):   accuracy %.3f | macro-F1 %.3f",
        draw_boost_f1, boosted_f1_metrics["accuracy"], boosted_f1_metrics["macro_f1"],
    )
    logger.info("TEST  expected-goals MAE: home %.3f away %.3f",
                test_metrics["mae_home_goals"], test_metrics["mae_away_goals"])
    logger.info("TEST  regression MAE (lower is better):")
    for name in REGRESSION_TARGETS:
        logger.info("  %-14s %.3f", name, test_metrics[f"mae_{name}"])
    logger.info("TEST  confusion matrix, argmax (rows=true, cols=pred) %s:", CLASS_NAMES)
    for cls, row in zip(CLASS_NAMES, test_matrix.tolist()):
        logger.info("  %-9s %s", cls, row)
    logger.info(
        "Best checkpoint saved to %s (T=%.2f, boost_acc=%.2f, boost_f1=%.2f)",
        _CHECKPOINT_PATH, temperature, draw_boost_acc, draw_boost_f1,
    )

    # Machine-readable snapshot for cross-phase comparison: ``metrics.json`` holds
    # the latest run; ``metrics_history.jsonl`` accumulates one line per run so
    # multi-seed means can be computed per ``--tag`` without manual bookkeeping.
    val_prob_metrics = probability_metrics(val_probs, val_labels)
    val_prob_metrics_cal = probability_metrics(val_probs_cal, val_labels)
    test_prob_metrics = probability_metrics(test_probs, test_labels)
    test_prob_metrics_cal = probability_metrics(test_probs_cal, test_labels)
    val_argmax_metrics = _metrics_from_confusion(
        _confusion_with_draw_boost(val_probs, val_labels, 1.0)
    )
    summary: Dict[str, object] = {
        "tag": args.tag,
        "seed": train_cfg.seed,
        "ensemble": ensemble_size,
        "num_numeric_features": len(feature_names),
        "n_rows": n_rows,
        "temperature": temperature,
        "draw_boost_acc": draw_boost_acc,
        "draw_boost_f1": draw_boost_f1,
        "val": {**val_argmax_metrics, **val_prob_metrics},
        "val_prob_calibrated": val_prob_metrics_cal,
        "test_argmax": {
            key: test_argmax_metrics[key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        },
        "test_boosted_acc": boosted_acc_metrics,
        "test_boosted": boosted_f1_metrics,
        "test_prob": test_prob_metrics,
        "test_prob_calibrated": test_prob_metrics_cal,
        "test_mae_goals": {
            "home": test_metrics["mae_home_goals"],
            "away": test_metrics["mae_away_goals"],
        },
        "test_mae": {name: test_metrics[f"mae_{name}"] for name in REGRESSION_TARGETS},
        "test_confusion_argmax": test_matrix.tolist(),
        "test_confusion_boosted": boosted_f1_matrix.tolist(),
        "test_reliability": reliability_table(test_probs_cal, test_labels),
    }
    _METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with _METRICS_HISTORY_PATH.open("a", encoding="utf-8") as history:
        history.write(json.dumps(summary) + "\n")
    logger.info(
        "VAL   acc %.3f | log-loss %.4f (cal %.4f) | brier %.4f | rps %.4f",
        val_argmax_metrics["accuracy"], val_prob_metrics["log_loss"],
        val_prob_metrics_cal["log_loss"], val_prob_metrics["brier"], val_prob_metrics["rps"],
    )
    logger.info(
        "TEST  log-loss %.4f (cal %.4f) | brier %.4f | rps %.4f",
        test_prob_metrics["log_loss"], test_prob_metrics_cal["log_loss"],
        test_prob_metrics["brier"], test_prob_metrics["rps"],
    )
    logger.info("Metrics snapshot written to %s (tag=%r).", _METRICS_PATH, args.tag)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-task match-statistics predictor.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the feature table before training.")
    parser.add_argument("--tag", default="", help="Free-form label stored in the metrics snapshot/history.")
    parser.add_argument(
        "--ensemble", type=int, default=1,
        help="Number of seed-ensemble members (consecutive seeds from --seed); "
        "their derived 1X2 probabilities are averaged. 1 = single model.",
    )
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
