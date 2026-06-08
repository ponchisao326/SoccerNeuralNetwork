"""Command-line inference for the trained ``MatchStatsMultiTaskPredictor``.

This script reconstructs the trained network from the artifacts written by
:mod:`ml.train` (``vocab.json``, ``scaler.json`` and the ``model.pt`` checkpoint)
and produces a full multi-task forecast for a single fixture: the 1X2 outcome
distribution plus the per-team regression heads (shots on target as the xG proxy,
corners and cards).

Feature provenance
-------------------
The model consumes a standardised continuous block of rolling-form features. In
production these would be recomputed from the team's most recent fixtures exactly
as in :func:`ml.features.build_feature_table`. That online feature store is out of
scope here, so :func:`build_feature_vector` emits a neutral placeholder: a vector
of standardised zeros. Because the scaler centres every feature on its training
mean, a standardised-zero vector is mathematically equivalent to "the
league-average match" -- the least-biased prior when no form is supplied. The
function is structured to make swapping in real rolling features a one-line change
(replace the ``raw`` array), so the placeholder does not constrain the interface.

Determinism
-----------
The model is placed in evaluation mode (``model.eval()``) so ``BatchNorm1d`` uses
its accumulated running statistics rather than batch statistics. This is essential
for single-sample inference: with a batch of one, batch statistics would be
degenerate (zero variance), whereas the running estimates yield the same transform
the network saw during training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from ml.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    UNK_INDEX,
    ModelConfig,
)
from ml.model import MatchStatsMultiTaskPredictor, ModelDimensions

logger = logging.getLogger(__name__)

_VOCAB_PATH: Path = ARTIFACTS_DIR / "vocab.json"
_SCALER_PATH: Path = ARTIFACTS_DIR / "scaler.json"
_CHECKPOINT_PATH: Path = ARTIFACTS_DIR / "model.pt"


def _load_json(path: Path) -> Dict[str, object]:
    """Load and parse a JSON artifact, raising a clear error if it is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}. Train the model first with "
            f"'uv run python -m ml.train --rebuild'."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts() -> Tuple[
    MatchStatsMultiTaskPredictor,
    Mapping[str, Mapping[str, int]],
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """Reconstruct the trained model and its preprocessing artifacts.

    The checkpoint carries the exact ``ModelDimensions`` and ``ModelConfig`` used
    at training time, so the architecture is rebuilt identically before the saved
    weights are loaded; nothing here depends on the live MongoDB schema.

    Returns:
        A tuple ``(model, vocab, mean, std, feature_names)`` where ``model`` is in
        evaluation mode, ``vocab`` maps each categorical field name to its
        ``name -> index`` table, ``mean`` / ``std`` are the per-feature
        standardisation statistics, and ``feature_names`` is their column order.
    """
    vocab = _load_json(_VOCAB_PATH)
    scaler = _load_json(_SCALER_PATH)
    checkpoint = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    dims = ModelDimensions(**checkpoint["dims"])
    saved_cfg = checkpoint.get("model_cfg", {})
    defaults = ModelConfig()
    model = MatchStatsMultiTaskPredictor(
        dims=dims,
        team_embedding_dim=int(saved_cfg.get("team_embedding_dim", defaults.team_embedding_dim)),
        league_embedding_dim=int(saved_cfg.get("league_embedding_dim", defaults.league_embedding_dim)),
        season_embedding_dim=int(saved_cfg.get("season_embedding_dim", defaults.season_embedding_dim)),
        # JSON serialises the tuple as a list; the constructor only iterates it.
        hidden_dims=tuple(saved_cfg.get("hidden_dims", defaults.hidden_dims)),
        dropout=float(saved_cfg.get("dropout", defaults.dropout)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    feature_names = [str(name) for name in scaler["feature_names"]]
    return model, vocab, mean, std, feature_names


def _lookup_index(table: Mapping[str, int], key: str) -> int:
    """Resolve a categorical value to its embedding index, falling back to <unk>.

    Unseen teams/leagues/seasons (e.g. a newly promoted club) map to the reserved
    ``UNK_INDEX`` rather than failing, mirroring the training-time encoding.
    """
    return int(table.get(key, UNK_INDEX))


def build_feature_vector(
    mean: np.ndarray,
    std: np.ndarray,
    raw: np.ndarray | None = None,
) -> torch.Tensor:
    """Standardise a raw rolling-form vector into the model's numeric input.

    Args:
        mean: Per-feature training means.
        std: Per-feature training standard deviations.
        raw: Optional raw (un-standardised) feature vector aligned with the
            scaler's column order. When ``None`` the training mean is used,
            yielding a standardised-zero vector (the neutral, average-match prior
            documented at module level).

    Returns:
        A ``(1, num_numeric_features)`` float tensor ready for the model.
    """
    if raw is None:
        raw = mean.copy()
    safe_std = std.copy()
    safe_std[safe_std == 0.0] = 1.0  # match the dataset's div-by-zero guard.
    standardised = (raw.astype(np.float32) - mean) / safe_std
    return torch.from_numpy(standardised).unsqueeze(0)


def predict_match(
    model: MatchStatsMultiTaskPredictor,
    vocab: Mapping[str, Mapping[str, int]],
    mean: np.ndarray,
    std: np.ndarray,
    home_team: str,
    away_team: str,
    league: str,
    season: str,
) -> Dict[str, object]:
    """Run a single forward pass and decode every task head.

    Args:
        model: Trained model in evaluation mode.
        vocab: Categorical ``name -> index`` tables.
        mean: Per-feature standardisation means.
        std: Per-feature standardisation standard deviations.
        home_team: Home team name (FBref spelling).
        away_team: Away team name (FBref spelling).
        league: League identifier (e.g. ``ENG-Premier League``).
        season: Season code (e.g. ``2425``).

    Returns:
        A dictionary holding the softmax outcome probabilities and the predicted
        home/away shots on target, corners and cards.
    """
    numeric = build_feature_vector(mean, std)
    categorical = torch.tensor(
        [[
            _lookup_index(vocab["teams"], home_team),
            _lookup_index(vocab["teams"], away_team),
            _lookup_index(vocab["leagues"], league),
            _lookup_index(vocab["seasons"], season),
        ]],
        dtype=torch.long,
    )

    with torch.no_grad():
        outputs = model(numeric, categorical)

    # CrossEntropyLoss trained raw logits; softmax recovers the calibrated 1X2
    # probability simplex for human-readable reporting.
    probabilities = F.softmax(outputs["outcome"], dim=1).squeeze(0).tolist()
    sot = outputs["sot"].squeeze(0).tolist()
    corners = outputs["corners"].squeeze(0).tolist()
    cards = outputs["cards"].squeeze(0).tolist()

    return {
        "outcome_probabilities": dict(zip(CLASS_NAMES, probabilities)),
        "home_sot": sot[0],
        "away_sot": sot[1],
        "home_corners": corners[0],
        "away_corners": corners[1],
        "home_cards": cards[0],
        "away_cards": cards[1],
    }


def format_report(
    home_team: str,
    away_team: str,
    league: str,
    season: str,
    prediction: Mapping[str, object],
) -> str:
    """Render the prediction dictionary as an aligned, human-readable report."""
    probs: Mapping[str, float] = prediction["outcome_probabilities"]  # type: ignore[assignment]
    lines: List[str] = [
        "=" * 56,
        f"  {home_team}  vs  {away_team}",
        f"  {league} | season {season}",
        "=" * 56,
        "  Outcome probabilities (1X2)",
        f"    Home win : {probs[CLASS_NAMES[0]] * 100:6.2f} %",
        f"    Draw     : {probs[CLASS_NAMES[1]] * 100:6.2f} %",
        f"    Away win : {probs[CLASS_NAMES[2]] * 100:6.2f} %",
        "-" * 56,
        f"  {'Metric':<24}{'Home':>14}{'Away':>14}",
        f"  {'Shots on target (xG)':<24}{prediction['home_sot']:>14.2f}{prediction['away_sot']:>14.2f}",
        f"  {'Corners':<24}{prediction['home_corners']:>14.2f}{prediction['away_corners']:>14.2f}",
        f"  {'Cards':<24}{prediction['home_cards']:>14.2f}{prediction['away_cards']:>14.2f}",
        "=" * 56,
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    """Parse the inference command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict full match statistics with the trained "
            "MatchStatsMultiTaskPredictor."
        )
    )
    parser.add_argument("--home", required=True, help="Home team name (FBref spelling).")
    parser.add_argument("--away", required=True, help="Away team name (FBref spelling).")
    parser.add_argument(
        "--league",
        default="ENG-Premier League",
        help="League identifier, e.g. 'ENG-Premier League'.",
    )
    parser.add_argument(
        "--season",
        default="2425",
        help="Season code, e.g. '2425' for 2024-25.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the artifacts, predict a single fixture and print the report."""
    args = _parse_args()
    model, vocab, mean, std, _ = load_artifacts()
    prediction = predict_match(
        model=model,
        vocab=vocab,
        mean=mean,
        std=std,
        home_team=args.home,
        away_team=args.away,
        league=args.league,
        season=args.season,
    )
    print(format_report(args.home, args.away, args.league, args.season, prediction))


if __name__ == "__main__":
    main()
