"""Optuna hyperparameter search for the multi-task match-statistics predictor.

Run:
    uv run python -m ml.tune                       # 30 trials, 15 epochs each
    uv run python -m ml.tune --trials 60 --epochs 20
    uv run python -m ml.tune --objective combined  # penalise regression MAE too

This driver tunes the network and the composite-loss weights *without* touching
the ETL or rebuilding the processed ``ml_match_features`` collection. The feature
table, vocabulary and chronological row order are treated as fixed inputs produced
by a prior ``ml.train`` run; only the optimisation and architecture knobs are
explored.

Search space
------------
* ``learning_rate`` -- log-uniform on ``[1e-4, 1e-2]``. The AdamW step size spans
  orders of magnitude in its effect, so a log-uniform prior places equal search
  density per decade rather than over-sampling the large-step region.
* ``dropout_p`` -- uniform on ``[0.1, 0.5]``. Regularisation strength is roughly
  linear in its effect, so a uniform prior is appropriate.
* ``lambda_sot`` / ``lambda_corners`` / ``lambda_cards`` -- uniform on
  ``[0.1, 2.0]``. These scale the auxiliary regression terms of the composite
  loss. Tuning them searches the multi-task trade-off: how much auxiliary
  gradient signal best regularises the shared backbone for the primary 1X2 task
  without the regression heads dominating the shared representation.

No data leakage
---------------
The temporal ``row_idx`` split from :func:`resolve_split_ranges` is reused
verbatim, and the standardisation statistics are fit on the train range only
(:func:`compute_normalization_stats`). Every trial selects on the validation
split, which lies strictly in the future of the train split; the test split is
never read here, so it remains an untouched hold-out for final evaluation.

Reproducibility
---------------
The TPE sampler is seeded, and every trial re-seeds all RNGs via
:func:`ml.train.set_seed` before building its model. Consequently weight
initialisation, dropout masks and the shuffle-buffer order are identical across
trials, so observed differences in validation score are attributable to the
sampled hyperparameters rather than stochastic noise.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import optuna
import torch
from dotenv import load_dotenv
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from ml.config import (
    ARTIFACTS_DIR,
    HEAD_SLICES,
    PROJECT_ROOT,
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
from ml.features import Vocabulary, numeric_feature_names
from ml.model import MatchStatsMultiTaskPredictor, ModelDimensions
from ml.train import CompositeLoss, _run_epoch, set_seed

logger = logging.getLogger(__name__)

_VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
_METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# Approximate per-head target magnitudes (per team, per match) used only to make
# the heterogeneous regression MAEs dimensionless when the optional combined
# objective is selected. Shots on target and corners sit around five per team,
# cards around two. These are fixed reference scales, not tuned quantities, so
# the combined objective stays deterministic and is never circular with the
# tuned lambda weights.
_MAE_REFERENCE: Dict[str, float] = {"sot": 5.0, "corners": 5.0, "cards": 2.0}

# Weight of the regression penalty in the combined objective. Kept small so the
# bounded macro-F1 term (in [0, 1]) dominates: the 1X2 outcome is the primary
# task and the regression heads act chiefly as a regulariser.
_COMBINED_REG_GAMMA: float = 0.10


@dataclass(frozen=True)
class TuningContext:
    """Immutable, trial-independent state shared across the whole study.

    Building the data loaders, normalisation statistics, class weights and model
    dimensions exactly once (rather than per trial) is the dominant efficiency
    lever: only the model and optimiser are rebuilt inside each trial.

    Attributes:
        device: Compute device.
        train_loader: Streaming loader over the train row range.
        val_loader: Deterministic (unshuffled) loader over the validation range.
        dims: Vocabulary sizes and numeric input width for the model.
        class_weights: Inverse-frequency 1X2 weights, fit on the train range.
        base_model_cfg: Architecture defaults; only ``dropout`` is overridden.
        epochs: Shortened per-trial training budget.
        seed: Global seed re-applied at the start of every trial.
        objective_kind: ``"f1"`` (maximise macro-F1) or ``"combined"``.
    """

    device: torch.device
    train_loader: DataLoader
    val_loader: DataLoader
    dims: ModelDimensions
    class_weights: torch.Tensor
    base_model_cfg: ModelConfig
    epochs: int
    seed: int
    objective_kind: str


def _trial_score(val_metrics: Dict[str, float], objective_kind: str) -> float:
    """Map a validation-metrics dict to the scalar Optuna maximises.

    For ``"f1"`` the score is the macro-F1 of the outcome head, a bounded,
    scale-free measure of the primary task. For ``"combined"`` a normalised
    regression penalty is subtracted::

        score = macro_f1 - gamma * mean_head( MAE_head / reference_head )

    Each head MAE is divided by a fixed reference magnitude to render the
    heterogeneous-scale errors dimensionless before averaging, so no single head
    dominates the penalty purely because its counts are larger.
    """
    macro_f1 = val_metrics["macro_f1"]
    if objective_kind == "f1":
        return macro_f1
    normalized = [
        val_metrics[f"mae_{head}"] / reference
        for head, reference in _MAE_REFERENCE.items()
    ]
    penalty = float(np.mean(normalized))
    return macro_f1 - _COMBINED_REG_GAMMA * penalty


def build_objective(ctx: TuningContext) -> Callable[[optuna.Trial], float]:
    """Create the closure Optuna calls once per trial.

    The returned objective samples the hyperparameters, runs a shortened training
    loop on the shared loaders, reports the per-epoch validation score for pruning
    and returns the best score observed across the trial's epochs (epoch-level
    early selection, mirroring the model-selection logic in ``ml.train``).
    """

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5)
        lambda_sot = trial.suggest_float("lambda_sot", 0.1, 2.0)
        lambda_corners = trial.suggest_float("lambda_corners", 0.1, 2.0)
        lambda_cards = trial.suggest_float("lambda_cards", 0.1, 2.0)

        # Identical initial conditions per trial: only the sampled knobs vary.
        set_seed(ctx.seed)

        model = MatchStatsMultiTaskPredictor(
            dims=ctx.dims,
            team_embedding_dim=ctx.base_model_cfg.team_embedding_dim,
            league_embedding_dim=ctx.base_model_cfg.league_embedding_dim,
            season_embedding_dim=ctx.base_model_cfg.season_embedding_dim,
            hidden_dims=ctx.base_model_cfg.hidden_dims,
            dropout=dropout_p,
        ).to(ctx.device)

        loss_weights = LossWeights(
            lambda_sot=lambda_sot,
            lambda_corners=lambda_corners,
            lambda_cards=lambda_cards,
        )
        criterion = CompositeLoss(ctx.class_weights, loss_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=TrainConfig.weight_decay,
        )

        best_score = float("-inf")
        best_metrics: Dict[str, float] = {}
        for epoch in range(1, ctx.epochs + 1):
            _run_epoch(model, ctx.train_loader, criterion, ctx.device, optimizer)
            _, val_metrics, _ = _run_epoch(model, ctx.val_loader, criterion, ctx.device)

            score = _trial_score(val_metrics, ctx.objective_kind)
            if score > best_score:
                best_score = score
                best_metrics = val_metrics

            # Report the running best so the pruner compares trials on a
            # monotonic, non-noisy curve and terminates clearly inferior ones.
            trial.report(best_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Persist diagnostics so the study log explains *why* a trial scored well.
        trial.set_user_attr("macro_f1", best_metrics["macro_f1"])
        trial.set_user_attr("accuracy", best_metrics["accuracy"])
        for head in HEAD_SLICES:
            trial.set_user_attr(f"mae_{head}", best_metrics[f"mae_{head}"])
        return best_score

    return objective


def _load_feature_metadata() -> Vocabulary:
    """Load the cached vocabulary and row metadata, refusing to rebuild.

    The tuner must operate on the existing ``ml_match_features`` collection. If
    the artifacts are absent the feature table has never been built, so we fail
    loudly with the exact command to run rather than silently triggering an
    expensive rebuild.
    """
    if not (_VOCAB_PATH.exists() and _METADATA_PATH.exists()):
        raise FileNotFoundError(
            "Missing vocab.json/metadata.json. Build the feature table first with "
            "'uv run python -m ml.train --rebuild' before tuning."
        )
    vocab = Vocabulary.from_dict(json.loads(_VOCAB_PATH.read_text(encoding="utf-8")))
    return vocab


def _build_context(args: argparse.Namespace) -> TuningContext:
    """Assemble the trial-independent state once (loaders, stats, dims)."""
    mongo = MongoConfig()
    split_cfg = SplitConfig()
    train_cfg = TrainConfig()

    vocab = _load_feature_metadata()
    meta = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = meta["feature_names"]
    n_rows = int(meta["n_rows"])
    if feature_names != numeric_feature_names():
        logger.warning("Cached feature names differ from current code; consider a rebuild via ml.train.")

    ranges = resolve_split_ranges(n_rows, split_cfg.train_fraction, split_cfg.val_fraction)
    logger.info("Temporal split (row_idx): train=%s val=%s", ranges["train"], ranges["val"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | %d rows | %d numeric features", device, n_rows, len(feature_names))

    # Fit on the train range only; identical to ml.train, so no leakage.
    mean, std = compute_normalization_stats(mongo, feature_names, ranges["train"])
    class_weights = compute_class_weights(mongo, ranges["train"]).to(device)

    train_ds = MatchFeatureDataset(
        mongo, feature_names, ranges["train"], mean, std,
        shuffle_buffer=train_cfg.shuffle_buffer, seed=args.seed,
        page_size=train_cfg.page_size,
    )
    val_ds = MatchFeatureDataset(
        mongo, feature_names, ranges["val"], mean, std,
        shuffle_buffer=0, seed=args.seed, page_size=train_cfg.page_size,
    )
    train_loader = make_dataloader(train_ds, args.batch_size, train_cfg.num_workers_train)
    val_loader = make_dataloader(val_ds, args.batch_size, train_cfg.num_workers_eval)

    dims = ModelDimensions(
        num_teams=vocab.num_teams,
        num_leagues=vocab.num_leagues,
        num_seasons=vocab.num_seasons,
        num_numeric_features=len(feature_names),
    )
    return TuningContext(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        dims=dims,
        class_weights=class_weights,
        base_model_cfg=ModelConfig(),
        epochs=args.epochs,
        seed=args.seed,
        objective_kind=args.objective,
    )


def _report_study(study: optuna.Study, objective_kind: str) -> None:
    """Print the best trial and a ready-to-paste ``config.py`` snippet."""
    best = study.best_trial
    logger.info("=" * 70)
    logger.info("Study finished: %d trials (%d pruned).",
                len(study.trials),
                len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))
    logger.info("Best %s score: %.4f", objective_kind, best.value)
    logger.info("Best trial diagnostics: macro_f1=%.4f acc=%.4f mae_sot=%.3f mae_cor=%.3f mae_card=%.3f",
                best.user_attrs.get("macro_f1", float("nan")),
                best.user_attrs.get("accuracy", float("nan")),
                best.user_attrs.get("mae_sot", float("nan")),
                best.user_attrs.get("mae_corners", float("nan")),
                best.user_attrs.get("mae_cards", float("nan")))
    logger.info("Best hyperparameters:")
    for key, value in best.params.items():
        logger.info("  %-16s = %s", key, value)

    params = best.params
    logger.info("-" * 70)
    logger.info("Apply manually in ml/config.py:")
    logger.info("  TrainConfig.learning_rate = %.6g", params["learning_rate"])
    logger.info("  ModelConfig.dropout       = %.4g", params["dropout_p"])
    logger.info("  LossWeights.lambda_sot     = %.4g", params["lambda_sot"])
    logger.info("  LossWeights.lambda_corners = %.4g", params["lambda_corners"])
    logger.info("  LossWeights.lambda_cards   = %.4g", params["lambda_cards"])
    logger.info("=" * 70)


def tune(args: argparse.Namespace) -> optuna.Study:
    """Run the Optuna study and report the best configuration."""
    load_dotenv(PROJECT_ROOT / ".env")
    set_seed(args.seed)
    ctx = _build_context(args)

    # TPE is a sample-efficient Bayesian sampler suited to the low-dimensional,
    # continuous search space; the median pruner stops trials whose running best
    # falls below the median of completed trials at the same epoch, after a warm-up
    # so early-epoch noise does not prune promising-but-slow configurations.
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = build_objective(ctx)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)

    _report_study(study, args.objective)
    return study


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for the MTL predictor.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per trial (shortened budget).")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed, help="Global RNG and sampler seed.")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="Mini-batch size.")
    parser.add_argument(
        "--objective",
        choices=("f1", "combined"),
        default="f1",
        help="Maximise outcome macro-F1, or a combined macro-F1 minus normalised regression MAE.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional wall-clock limit in seconds for the whole study.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Optuna's own trial logs are routed through the same handler.
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    tune(_parse_args())


if __name__ == "__main__":
    main()
