"""Command-line inference for the trained ``MatchStatsMultiTaskPredictor``.

This script reconstructs the trained network from the artifacts written by
:mod:`ml.train` (``vocab.json``, ``scaler.json`` and the ``model.pt`` checkpoint)
and produces a full multi-task forecast for a single fixture: the 1X2 outcome
distribution plus the per-team regression heads (shots on target as the xG proxy,
corners and cards).

Feature provenance
-------------------
By default the model consumes the team's *real* rolling-form features. The full
chronological history is replayed (:func:`ml.features.build_team_states`) to
recover both clubs' current ELO and exponential moving averages, and the upcoming
fixture is featurised with :meth:`ml.features.FeatureBuilder.peek_match_features`
-- the read-only counterpart of the training-time ``process`` -- so inference and
training share one feature construction. The ``--neutral`` flag instead emits a
standardised-zero vector (every feature at its training mean, i.e. the
league-average match), a fast prior that ignores current form.

Data caveat
-----------
The regression heads (shots on target, corners, cards) are only reliable for
fixtures whose clubs have WhoScored coverage. Competitions ingested from FBref
alone (e.g. the Spanish Segunda Division) carry ``sot_ema = 0`` and had their
regression targets masked during training, so for those matches only the 1X2
outcome -- driven by the goal-based EMAs and ELO -- should be trusted.

Determinism
-----------
The model is placed in evaluation mode (``model.eval()``) so ``BatchNorm1d`` uses
its accumulated running statistics rather than batch statistics. This is essential
for single-sample inference: with a batch of one, batch statistics would be
degenerate (zero variance), whereas the running estimates yield the same transform
the network saw during training.

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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from ml.calibration import apply_temperature
from ml.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    UNK_INDEX,
    ModelConfig,
)
from ml.model import (
    MatchStatsMultiTaskPredictor,
    ModelDimensions,
    poisson_outcome_probs,
)

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
    List[MatchStatsMultiTaskPredictor],
    Mapping[str, Mapping[str, int]],
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, float],
]:
    """Reconstruct the trained model ensemble and its preprocessing artifacts.

    The checkpoint carries the exact ``ModelDimensions`` and ``ModelConfig`` used
    at training time, so the architecture is rebuilt identically before the saved
    weights are loaded; nothing here depends on the live MongoDB schema. A
    checkpoint trained with ``--ensemble K`` carries ``model_states`` (K state
    dicts); older single-model checkpoints fall back to ``model_state``.

    Returns:
        A tuple ``(models, vocab, mean, std, feature_names, decision)`` where
        ``models`` are the ensemble members in evaluation mode, ``vocab`` maps
        each categorical field name to its ``name -> index`` table, ``mean`` /
        ``std`` are the per-feature standardisation statistics,
        ``feature_names`` is their column order, and ``decision`` holds the
        validation-fit decision-layer parameters (``temperature``,
        ``draw_boost_acc``, ``draw_boost_f1``).
    """
    vocab = _load_json(_VOCAB_PATH)
    scaler = _load_json(_SCALER_PATH)
    checkpoint = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    dims = ModelDimensions(**checkpoint["dims"])
    saved_cfg = checkpoint.get("model_cfg", {})
    defaults = ModelConfig()
    states = checkpoint.get("model_states") or [checkpoint["model_state"]]
    models: List[MatchStatsMultiTaskPredictor] = []
    for state in states:
        model = MatchStatsMultiTaskPredictor(
            dims=dims,
            team_embedding_dim=int(saved_cfg.get("team_embedding_dim", defaults.team_embedding_dim)),
            league_embedding_dim=int(saved_cfg.get("league_embedding_dim", defaults.league_embedding_dim)),
            season_embedding_dim=int(saved_cfg.get("season_embedding_dim", defaults.season_embedding_dim)),
            # JSON serialises the tuple as a list; the constructor only iterates it.
            hidden_dims=tuple(saved_cfg.get("hidden_dims", defaults.hidden_dims)),
            dropout=float(saved_cfg.get("dropout", defaults.dropout)),
        )
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    feature_names = [str(name) for name in scaler["feature_names"]]
    decision = {
        "temperature": float(checkpoint.get("temperature", 1.0)),
        "draw_boost_acc": float(checkpoint.get("draw_boost_acc", 1.0)),
        "draw_boost_f1": float(checkpoint.get("draw_boost_f1", checkpoint.get("draw_boost", 1.0))),
    }
    return models, vocab, mean, std, feature_names, decision


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


def compute_live_numeric(
    home_team: str,
    away_team: str,
    feature_names: List[str],
    match_date: datetime,
    season_progress: float,
    league: str,
    season: str,
) -> Tuple[np.ndarray, bool, bool]:
    """Build the real (un-standardised) feature vector for an upcoming fixture.

    Replays the full chronological history (:func:`ml.features.build_team_states`)
    to recover both clubs' current ELO and exponential moving averages, then reads
    their pre-match features via :meth:`FeatureBuilder.peek_match_features`. This
    guarantees the prediction is constructed exactly like a training row, so the
    forecast reflects genuine recent form and opponent-adjusted strength rather
    than a neutral prior.

    Args:
        home_team: Home team name (FBref spelling).
        away_team: Away team name (FBref spelling).
        feature_names: Ordered numeric columns to emit (scaler/model order).
        match_date: Date of the upcoming fixture (drives the rest-days feature).
        season_progress: Normalised season position in ``[0, 1]``; 1.0 denotes the
            end of the campaign (e.g. promotion play-offs).
        league: League identifier of the fixture (FBref spelling).
        season: Season code of the fixture (drives the promoted/newly-seen flag).

    Returns:
        ``(raw_vector, home_known, away_known)`` where ``raw_vector`` is aligned to
        ``feature_names`` and the booleans flag whether each club had prior matches
        in the database (a ``False`` means the club fell back to cold-start priors).
    """
    # Imported lazily so the module stays importable without a live MongoDB.
    from dotenv import load_dotenv

    from ml.config import PROJECT_ROOT, FeatureConfig, MongoConfig
    from ml.features import build_team_states, has_team_history

    # Load the project .env so MONGODB_URI resolves (mirrors ml.train).
    load_dotenv(PROJECT_ROOT / ".env")
    builder, _ = build_team_states(MongoConfig(), FeatureConfig())
    home_known = has_team_history(builder, home_team)
    away_known = has_team_history(builder, away_team)

    features = builder.peek_match_features(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        week=None,
        season_match_span=0.0,
        league=league,
        season=season,
    )
    # ``peek_match_features`` sets season_progress to 0.0 when no week is supplied;
    # override it with the caller's explicit value for the upcoming fixture.
    features["season_progress"] = season_progress

    raw = np.array([float(features[name]) for name in feature_names], dtype=np.float32)
    return raw, home_known, away_known


def predict_match(
    models: List[MatchStatsMultiTaskPredictor],
    vocab: Mapping[str, Mapping[str, int]],
    numeric: torch.Tensor,
    home_team: str,
    away_team: str,
    league: str,
    season: str,
    decision: Optional[Dict[str, float]] = None,
    objective: str = "accuracy",
) -> Dict[str, object]:
    """Run one forward pass per ensemble member and decode every task head.

    Args:
        models: Trained ensemble members in evaluation mode (one or more).
        vocab: Categorical ``name -> index`` tables.
        numeric: ``(1, num_numeric_features)`` standardised feature tensor.
        home_team: Home team name (FBref spelling).
        away_team: Away team name (FBref spelling).
        league: League identifier (e.g. ``ENG-Premier League``).
        season: Season code (e.g. ``2425``).
        decision: Validation-fit decision parameters (``temperature``,
            ``draw_boost_acc``, ``draw_boost_f1``); ``None`` means raw argmax.
        objective: Which tuned boost drives the headline pick: ``"accuracy"``
            (maximise hit rate, the default) or ``"f1"`` (balanced draw recall).

    Returns:
        A dictionary holding the calibrated Poisson-derived 1X2 probabilities,
        both decision picks, the expected home/away goals, and the predicted
        shots on target, corners and cards.
    """
    decision = decision or {"temperature": 1.0, "draw_boost_acc": 1.0, "draw_boost_f1": 1.0}
    categorical = torch.tensor(
        [[
            _lookup_index(vocab["teams"], home_team),
            _lookup_index(vocab["teams"], away_team),
            _lookup_index(vocab["leagues"], league),
            _lookup_index(vocab["seasons"], season),
        ]],
        dtype=torch.long,
    )

    # The 1X2 distribution is derived analytically from the two Poisson goal
    # rates (no separate softmax head); the draw is the diagonal mass P(H=A).
    # Ensemble members are combined by averaging their probability simplexes
    # (a mixture of Dixon-Coles grids) and their point predictions. Temperature
    # scaling then repairs the calibration without moving the argmax.
    prob_rows: List[np.ndarray] = []
    head_sums: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for model in models:
            outputs = model(numeric, categorical)
            prob_rows.append(
                poisson_outcome_probs(outputs["goals"], rho=outputs.get("rho")).numpy()
            )
            for head in ("goals", "sot", "corners", "cards"):
                values = outputs[head].squeeze(0).numpy()
                head_sums[head] = head_sums.get(head, 0.0) + values

    raw_probs = np.mean(prob_rows, axis=0)
    probabilities = apply_temperature(raw_probs, decision["temperature"])[0].tolist()
    member_count = float(len(models))
    goals = (head_sums["goals"] / member_count).tolist()
    sot = (head_sums["sot"] / member_count).tolist()
    corners = (head_sums["corners"] / member_count).tolist()
    cards = (head_sums["cards"] / member_count).tolist()

    # Cost-sensitive picks: scale the calibrated draw probability by the
    # validation-tuned boost before the argmax. The accuracy boost maximises
    # raw hit rate; the F1 boost rebalances the minority draw class. Reported
    # probabilities are left un-boosted so the user sees honest estimates.
    picks: Dict[str, str] = {}
    for name, boost_key in (("accuracy", "draw_boost_acc"), ("f1", "draw_boost_f1")):
        adjusted = list(probabilities)
        adjusted[1] *= decision[boost_key]
        picks[name] = CLASS_NAMES[int(np.argmax(adjusted))]

    return {
        "outcome_probabilities": dict(zip(CLASS_NAMES, probabilities)),
        "predicted_outcome": picks[objective],
        "objective": objective,
        "pick_accuracy": picks["accuracy"],
        "pick_f1": picks["f1"],
        "temperature": decision["temperature"],
        "draw_boost": decision["draw_boost_acc" if objective == "accuracy" else "draw_boost_f1"],
        "home_goals_exp": goals[0],
        "away_goals_exp": goals[1],
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
        "  Outcome probabilities (1X2, Poisson-derived, calibrated "
        f"T={prediction['temperature']:.2f})",
        f"    Home win : {probs[CLASS_NAMES[0]] * 100:6.2f} %",
        f"    Draw     : {probs[CLASS_NAMES[1]] * 100:6.2f} %",
        f"    Away win : {probs[CLASS_NAMES[2]] * 100:6.2f} %",
        f"    Pick     : {prediction['predicted_outcome']}"
        f"  (objective={prediction['objective']}, draw boost x{prediction['draw_boost']:.2f})",
        f"    Alt picks: accuracy={prediction['pick_accuracy']}, f1={prediction['pick_f1']}",
        "-" * 56,
        f"  {'Metric':<24}{'Home':>14}{'Away':>14}",
        f"  {'Expected goals':<24}{prediction['home_goals_exp']:>14.2f}{prediction['away_goals_exp']:>14.2f}",
        f"  {'Shots on target':<24}{prediction['home_sot']:>14.2f}{prediction['away_sot']:>14.2f}",
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
    parser.add_argument(
        "--date",
        default=None,
        help="Upcoming fixture date (ISO, e.g. 2026-06-15); defaults to today. "
        "Drives the rest-days feature. Ignored with --neutral.",
    )
    parser.add_argument(
        "--season-progress",
        type=float,
        default=1.0,
        help="Normalised season position in [0, 1] for the fixture (default 1.0, "
        "i.e. end of season / play-offs). Ignored with --neutral.",
    )
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Use a neutral (zeroed) form vector instead of replaying real "
        "ELO/EMA history. Faster, but ignores current form.",
    )
    parser.add_argument(
        "--objective",
        choices=("accuracy", "f1"),
        default="accuracy",
        help="Decision objective for the pick: 'accuracy' maximises raw hit "
        "rate (default); 'f1' applies the draw-recall-balancing boost.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the artifacts, predict a single fixture and print the report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    models, vocab, mean, std, feature_names, decision = load_artifacts()

    if args.neutral:
        numeric = build_feature_vector(mean, std)
    else:
        match_date = datetime.fromisoformat(args.date) if args.date else datetime.now()
        logger.info("Replaying match history to build live ELO/EMA features...")
        raw, home_known, away_known = compute_live_numeric(
            home_team=args.home,
            away_team=args.away,
            feature_names=feature_names,
            match_date=match_date,
            season_progress=args.season_progress,
            league=args.league,
            season=args.season,
        )
        for team, known in ((args.home, home_known), (args.away, away_known)):
            if not known:
                logger.warning(
                    "No match history for %r; it falls back to cold-start priors "
                    "(check the exact FBref spelling).",
                    team,
                )
        numeric = build_feature_vector(mean, std, raw=raw)

    prediction = predict_match(
        models=models,
        vocab=vocab,
        numeric=numeric,
        home_team=args.home,
        away_team=args.away,
        league=args.league,
        season=args.season,
        decision=decision,
        objective=args.objective,
    )
    print(format_report(args.home, args.away, args.league, args.season, prediction))


if __name__ == "__main__":
    main()
