"""Causal feature engineering for 1X2 match-outcome prediction.

Prediction granularity
----------------------
The supervised unit is a single **match**, and the label (home win / draw / away
win) is derived from the ``score`` field of ``fbref_schedule``. The much larger
``whoscored_events`` collection (~8M event rows) is *not* the label carrier: it
holds within-match event streams, not match outcomes. It is therefore treated as
an optional, future per-match feature source (e.g. aggregated xG, shot counts),
documented as an extension point in :func:`build_feature_table`. The streaming /
pagination design below keeps memory flat regardless of collection size, so it
scales to that larger source unchanged.

Leakage policy
--------------
Every feature for a given match is computed from that match's *strictly prior*
history only. The builder processes fixtures in global chronological order and,
for each match, (1) emits the pre-match rolling state of both teams, then
(2) updates that state with the realised result. Because the update always
happens *after* the emit, no feature can ever observe its own match or any future
match. This is the structural guarantee against data leakage; the temporal
train/val/test split (see :mod:`ml.config`) is the second line of defence.

Efficiency
----------
Rolling aggregates are maintained incrementally with a fixed-length
``collections.deque`` per team (O(1) per update, O(#teams) memory) rather than
recomputed by re-scanning history per match. The whole table is built in a single
streaming pass over a Mongo cursor, so peak memory is independent of the number
of matches.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

from pymongo import ASCENDING, MongoClient

from ml.config import (
    UNK_INDEX,
    UNK_TOKEN,
    FeatureConfig,
    MongoConfig,
)

logger = logging.getLogger(__name__)

# A score string such as "0–3" (en-dash) or occasionally "0-3" (hyphen).
# We extract the first two integers regardless of the exact separator glyph.
_SCORE_RE = re.compile(r"(\d+)\D+(\d+)")

# Label encoding for the multiclass 1X2 target.
LABEL_HOME_WIN: int = 0
LABEL_DRAW: int = 1
LABEL_AWAY_WIN: int = 2

# Ordered numeric feature names. The order is the contract between the feature
# table, the normalisation scaler and the model input layer; do not reorder
# without rebuilding the feature table.
#
# The per-side block holds exponential moving averages (EMA) of the team's recent
# performance rather than simple sliding-window means: an EMA weights the most
# recent match most heavily and decays older matches geometrically, encoding the
# recency bias that a flat window lacks.
_PER_SIDE_FEATURES: Tuple[str, ...] = (
    "gf_ema",      # EMA of goals scored
    "ga_ema",      # EMA of goals conceded
    "sot_ema",     # EMA of shots on target (the xG proxy), from WhoScored
    "pts_ema",     # EMA of league points (win=3, draw=1, loss=0)
    "rest_days",   # days since the team's previous match (clamped)
    "played",      # number of prior matches observed (EMA maturity signal)
)

# Absolute pre-match ELO ratings of each side. Named explicitly (not h_/a_
# prefixed) because ELO is a relational rating maintained on a shared scale rather
# than an independent per-side rolling statistic.
_ELO_FEATURES: Tuple[str, ...] = (
    "home_elo",
    "away_elo",
)

_DIFF_FEATURES: Tuple[str, ...] = (
    "gf_ema_diff",     # home gf_ema - away gf_ema
    "ga_ema_diff",     # home ga_ema - away ga_ema
    "pts_ema_diff",    # home pts_ema - away pts_ema
    "elo_diff",        # home_elo - away_elo (signed strength gap)
)

_GLOBAL_FEATURES: Tuple[str, ...] = (
    "season_progress",  # matchweek normalised to [0, 1]; captures fatigue / table pressure
)


def numeric_feature_names() -> List[str]:
    """Return the ordered list of numeric feature column names.

    The model's numeric input width equals ``len(numeric_feature_names())``.
    """
    names: List[str] = [f"h_{f}" for f in _PER_SIDE_FEATURES]
    names += [f"a_{f}" for f in _PER_SIDE_FEATURES]
    names += list(_ELO_FEATURES)
    names += list(_DIFF_FEATURES)
    names += list(_GLOBAL_FEATURES)
    return names


def parse_score(score: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse a raw FBref score string into ``(home_goals, away_goals)``.

    Handles the en-dash separator FBref uses and tolerates surrounding noise.

    Args:
        score: Raw score string (e.g. ``"2–1"``) or ``None``.

    Returns:
        ``(home_goals, away_goals)`` or ``None`` if the score is unparseable.
    """
    if not score:
        return None
    match = _SCORE_RE.search(score)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def outcome_label(home_goals: int, away_goals: int) -> int:
    """Map a final score to the 1X2 class index."""
    if home_goals > away_goals:
        return LABEL_HOME_WIN
    if home_goals < away_goals:
        return LABEL_AWAY_WIN
    return LABEL_DRAW


@dataclass
class _TeamState:
    """Incrementally maintained recency-weighted state for one team.

    Each statistic is stored as an exponential moving average (EMA) scalar instead
    of a sliding window of raw values, so the per-team memory is O(1) and the most
    recent fixture dominates the estimate. The EMA trackers are ``Optional`` and
    start as ``None`` so that the first observed value can seed the average
    exactly (no spurious decay toward zero on match one). The ELO rating instead
    starts at a finite base rating, because an unknown team has a meaningful
    neutral prior whereas its EMA form does not.

    Attributes:
        elo: Current ELO rating; initialised to ``FeatureConfig.elo_base``.
        gf_ema: EMA of goals scored, or ``None`` before the first match.
        ga_ema: EMA of goals conceded.
        sot_ema: EMA of shots on target (absent for matches without WhoScored
            coverage, so it may stay ``None`` longer than the goal EMAs).
        pts_ema: EMA of league points earned (3/1/0).
        played: Count of prior matches folded in, used as an EMA-maturity signal.
        last_date: Date of the most recent processed match, for rest-day deltas.
    """

    elo: float
    gf_ema: Optional[float] = None
    ga_ema: Optional[float] = None
    sot_ema: Optional[float] = None
    pts_ema: Optional[float] = None
    played: int = 0
    last_date: Optional[datetime] = None

    @classmethod
    def empty(cls, elo_base: float) -> "_TeamState":
        return cls(elo=elo_base)


class FeatureBuilder:
    """Builds leak-free per-match feature rows from chronological fixtures.

    The builder is stateful: feed it matches in ascending date order via
    :meth:`process`. It is *not* safe to feed out-of-order matches, as that would
    corrupt the rolling state and violate the causality guarantee.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self._cfg = config
        self._states: Dict[str, _TeamState] = {}

    def _state(self, team: str) -> _TeamState:
        state = self._states.get(team)
        if state is None:
            state = _TeamState.empty(self._cfg.elo_base)
            self._states[team] = state
        return state

    @staticmethod
    def _emit(value: Optional[float]) -> float:
        """Map an EMA tracker to a feature value, using 0.0 as the cold-start prior."""
        return float(value) if value is not None else 0.0

    @staticmethod
    def _ema_step(old: Optional[float], value: Optional[float], alpha: float) -> Optional[float]:
        """Advance one EMA tracker by a single observation.

        Implements ``s_t = alpha * x_t + (1 - alpha) * s_{t-1}``. The first
        observation seeds the average directly (``s_0 = x_0``) so the series is not
        biased toward zero, and a missing observation (``value is None``) leaves
        the running average untouched rather than corrupting it with a placeholder.
        """
        if value is None:
            return old
        if old is None:
            return float(value)
        return float(value) * alpha + old * (1.0 - alpha)

    def _side_features(self, team: str, match_date: datetime) -> Dict[str, float]:
        """Compute the pre-match recency-weighted feature block for one team."""
        state = self._state(team)
        if state.played == 0:
            # Cold start: neutral priors. ``played=0`` lets the network learn to
            # discount these unreliable rows rather than us imputing strong values.
            return {
                "gf_ema": 0.0,
                "ga_ema": 0.0,
                "sot_ema": 0.0,
                "pts_ema": 0.0,
                "rest_days": self._cfg.default_rest_days,
                "played": 0.0,
            }

        if state.last_date is not None:
            rest = (match_date - state.last_date).days
            rest = max(0.0, min(float(rest), self._cfg.max_rest_days))
        else:
            rest = self._cfg.default_rest_days

        return {
            "gf_ema": self._emit(state.gf_ema),
            "ga_ema": self._emit(state.ga_ema),
            "sot_ema": self._emit(state.sot_ema),
            "pts_ema": self._emit(state.pts_ema),
            "rest_days": rest,
            "played": float(state.played),
        }

    def _update_team_state(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_sot: Optional[float],
        away_sot: Optional[float],
        match_date: datetime,
    ) -> None:
        """Fold a realised result into both teams' state (strictly post-emit).

        The update is performed jointly because the ELO step is a head-to-head
        computation. With pre-match ratings ``R_h`` and ``R_a`` the home team's
        expected score is the logistic

            E_h = 1 / (1 + 10 ** ((R_a - R_h) / 400)),

        and ``E_a = 1 - E_h``. Given the actual home score ``S_h`` in ``{1, 0.5, 0}``
        for win/draw/loss, both ratings move by the K-scaled surprise

            R' = R + K * (S - E).

        Because expectations are read from the pre-match ratings of *both* sides
        before either is mutated, the update is order-independent and conserves
        total rating (the home team's gain equals the away team's loss). The EMA
        form trackers are advanced afterwards with the same realised statistics.
        """
        home_state = self._state(home_team)
        away_state = self._state(away_team)

        home_elo = home_state.elo
        away_elo = away_state.elo
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
        expected_away = 1.0 - expected_home

        if home_goals > away_goals:
            actual_home = 1.0
            home_points, away_points = 3.0, 0.0
        elif home_goals == away_goals:
            actual_home = 0.5
            home_points, away_points = 1.0, 1.0
        else:
            actual_home = 0.0
            home_points, away_points = 0.0, 3.0
        actual_away = 1.0 - actual_home

        k = self._cfg.elo_k
        home_state.elo = home_elo + k * (actual_home - expected_home)
        away_state.elo = away_elo + k * (actual_away - expected_away)

        alpha = self._cfg.ema_alpha
        home_state.gf_ema = self._ema_step(home_state.gf_ema, float(home_goals), alpha)
        home_state.ga_ema = self._ema_step(home_state.ga_ema, float(away_goals), alpha)
        home_state.sot_ema = self._ema_step(home_state.sot_ema, home_sot, alpha)
        home_state.pts_ema = self._ema_step(home_state.pts_ema, home_points, alpha)
        home_state.played += 1
        home_state.last_date = match_date

        away_state.gf_ema = self._ema_step(away_state.gf_ema, float(away_goals), alpha)
        away_state.ga_ema = self._ema_step(away_state.ga_ema, float(home_goals), alpha)
        away_state.sot_ema = self._ema_step(away_state.sot_ema, away_sot, alpha)
        away_state.pts_ema = self._ema_step(away_state.pts_ema, away_points, alpha)
        away_state.played += 1
        away_state.last_date = match_date

    def peek_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        week: Optional[float],
        season_match_span: float,
    ) -> Dict[str, float]:
        """Compute the pre-match feature vector WITHOUT mutating any team state.

        This is the read-only half of :meth:`process`: it produces the identical
        19 features a training row receives, but leaves both teams' rolling state
        untouched. It is the inference entry point, where the upcoming fixture has
        no result to fold in. Calling it is therefore idempotent and order-free.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            match_date: Kickoff date (for the rest-days delta).
            week: Matchweek number, or ``None`` if absent.
            season_match_span: Matchweeks in the season, for ``season_progress``.

        Returns:
            Ordered numeric feature dict keyed by :func:`numeric_feature_names`.
        """
        # Pre-match ratings are read straight from current state (no mutation).
        home_elo = self._state(home_team).elo
        away_elo = self._state(away_team).elo

        home = self._side_features(home_team, match_date)
        away = self._side_features(away_team, match_date)

        features: Dict[str, float] = {}
        for name, value in home.items():
            features[f"h_{name}"] = value
        for name, value in away.items():
            features[f"a_{name}"] = value

        features["home_elo"] = home_elo
        features["away_elo"] = away_elo

        features["gf_ema_diff"] = home["gf_ema"] - away["gf_ema"]
        features["ga_ema_diff"] = home["ga_ema"] - away["ga_ema"]
        features["pts_ema_diff"] = home["pts_ema"] - away["pts_ema"]
        features["elo_diff"] = home_elo - away_elo

        if week is not None and season_match_span > 0:
            features["season_progress"] = max(0.0, min(week / season_match_span, 1.0))
        else:
            features["season_progress"] = 0.0

        return features

    def process(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_sot: Optional[float],
        away_sot: Optional[float],
        match_date: datetime,
        week: Optional[float],
        season_match_span: float,
    ) -> Dict[str, float]:
        """Emit the pre-match feature vector, then update both teams' state.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            home_goals: Realised home goals (used only for the post-emit update).
            away_goals: Realised away goals (post-emit update only).
            home_sot: Realised home shots on target, or ``None`` when the match has
                no WhoScored coverage (post-emit EMA update only).
            away_sot: Realised away shots on target, or ``None`` (post-emit only).
            match_date: Kickoff date.
            week: Matchweek number, or ``None`` if absent.
            season_match_span: Number of matchweeks in the season, for
                normalising ``season_progress``.

        Returns:
            Ordered numeric feature dict keyed by :func:`numeric_feature_names`.
        """
        features = self.peek_match_features(home_team, away_team, match_date, week, season_match_span)

        # Post-emit state update: strictly after features are read.
        self._update_team_state(
            home_team, away_team, home_goals, away_goals, home_sot, away_sot, match_date
        )

        return features


@dataclass(frozen=True)
class Vocabulary:
    """Deterministic string-to-index maps for categorical embedding inputs.

    Index 0 is reserved for ``<unk>`` in every map so that values unseen during
    feature building map to a trainable "unknown" embedding rather than crashing.

    Attributes:
        teams: Team-name -> index.
        leagues: League-name -> index.
        seasons: Season-string -> index.
    """

    teams: Dict[str, int]
    leagues: Dict[str, int]
    seasons: Dict[str, int]

    def team_index(self, name: str) -> int:
        return self.teams.get(name, UNK_INDEX)

    def league_index(self, name: str) -> int:
        return self.leagues.get(name, UNK_INDEX)

    def season_index(self, name: str) -> int:
        return self.seasons.get(name, UNK_INDEX)

    @property
    def num_teams(self) -> int:
        return len(self.teams)

    @property
    def num_leagues(self) -> int:
        return len(self.leagues)

    @property
    def num_seasons(self) -> int:
        return len(self.seasons)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {"teams": self.teams, "leagues": self.leagues, "seasons": self.seasons}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, int]]) -> "Vocabulary":
        return cls(teams=data["teams"], leagues=data["leagues"], seasons=data["seasons"])


def _coerce_date(raw: object) -> Optional[datetime]:
    """Parse the schedule ``date`` field (stored as ISO string) into datetime."""
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _iter_schedule(
    client: MongoClient, mongo: MongoConfig
) -> Iterator[Dict[str, object]]:
    """Stream every schedule document in global chronological order.

    Sorting by ``date`` then ``game`` yields a stable, fully ordered fixture
    stream. The cursor is consumed lazily, so memory stays flat.
    """
    collection = client[mongo.db][mongo.source_collection]
    cursor = collection.find(
        {},
        projection={
            "_id": 0,
            "league": 1,
            "season": 1,
            "game": 1,
            "game_id": 1,
            "date": 1,
            "week": 1,
            "home_team": 1,
            "away_team": 1,
            "score": 1,
        },
    ).sort([("date", ASCENDING), ("game", ASCENDING)])
    cursor.batch_size(1000)
    for doc in cursor:
        yield doc


def _build_vocabulary(rows: Iterable[Dict[str, object]]) -> Vocabulary:
    """Construct deterministic categorical vocabularies from schedule rows.

    Sorting names before indexing makes the mapping reproducible across runs,
    which matters for checkpoint compatibility. Building the team vocabulary over
    *all* rows (not just the train split) is intentional: the embedding table
    must have a slot for every team that can appear at inference. Embeddings for
    teams absent from the train period simply stay near their initialisation,
    which is the correct cold-start behaviour and is not leakage (no label or
    future-feature information is used to build the vocabulary).
    """
    teams: set[str] = set()
    leagues: set[str] = set()
    seasons: set[str] = set()
    for row in rows:
        home = row.get("home_team")
        away = row.get("away_team")
        league = row.get("league")
        season = row.get("season")
        if isinstance(home, str):
            teams.add(home)
        if isinstance(away, str):
            teams.add(away)
        if isinstance(league, str):
            leagues.add(league)
        if isinstance(season, str):
            seasons.add(season)

    def index_map(values: set[str]) -> Dict[str, int]:
        mapping = {UNK_TOKEN: UNK_INDEX}
        for offset, value in enumerate(sorted(values), start=1):
            mapping[value] = offset
        return mapping

    return Vocabulary(
        teams=index_map(teams),
        leagues=index_map(leagues),
        seasons=index_map(seasons),
    )


def _season_match_spans(client: MongoClient, mongo: MongoConfig) -> Dict[Tuple[str, str], float]:
    """Compute the max matchweek per (league, season) for progress normalisation."""
    collection = client[mongo.db][mongo.source_collection]
    spans: Dict[Tuple[str, str], float] = {}
    pipeline = [
        {
            "$group": {
                "_id": {"league": "$league", "season": "$season"},
                "max_week": {"$max": "$week"},
            }
        }
    ]
    for doc in collection.aggregate(pipeline):
        key = (doc["_id"]["league"], doc["_id"]["season"])
        max_week = doc.get("max_week")
        spans[key] = float(max_week) if max_week else 0.0
    return spans


# ---------------------------------------------------------------------------
# WhoScored regression-target extraction
# ---------------------------------------------------------------------------

# Event-type tokens that define each regression target. Shots on target follow
# the Opta/WhoScored convention: a shot that scored (Goal) or forced a save
# (SavedShot). Shots that missed (MissedShots) or hit the woodwork (ShotOnPost)
# are off target and excluded.
_SOT_TYPES: FrozenSet[str] = frozenset({"Goal", "SavedShot"})
_CORNER_TYPE: str = "CornerAwarded"
_CARD_TYPE: str = "Card"
# Each corner is logged twice by Opta: "Successful" for the attacking team that
# won it and "Unsuccessful" for the defending team. Counting only the successful
# event attributes the corner to the correct (attacking) team and avoids double
# counting (which otherwise inflates corners to ~2x the true per-team total).
_CORNER_OUTCOME: str = "Successful"

# Tokens stripped before comparing team names: club-form qualifiers and sponsor
# noise that differ between FBref and WhoScored spellings while carrying no
# disambiguating information.
_TEAM_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "fc", "afc", "cf", "cd", "ac", "as", "ssc", "rc", "sc", "ud", "sd", "us",
        "calcio", "club", "de", "the", "1", "04", "05", "08", "09",
        "1846", "1899", "1900",
    }
)

# Minimum similarity to accept a WhoScored<->FBref team pairing. Below this the
# pair is left unmapped (its matches get a regression mask of 0) rather than risk
# a silent, wrong target attribution.
_NAME_MATCH_THRESHOLD: float = 0.40

# Curated WhoScored -> FBref aliases for unambiguous abbreviations that share too
# few characters for fuzzy matching to resolve (e.g. "PSG", "RBL"). Seeded before
# the fuzzy pass so these high-value clubs are never masked.
_TEAM_ALIASES: Dict[str, str] = {
    "PSG": "Paris Saint-Germain",
    "RBL": "RB Leipzig",
}

# Per-team event aggregates: (shots_on_target, corners, cards).
_TeamTargets = Tuple[int, int, int]
# Lookup key: (league, season, date_str, frozenset{fbref_team_a, fbref_team_b}).
_GameKey = Tuple[str, str, str, FrozenSet[str]]


def _normalize_team(name: str) -> Tuple[Set[str], str]:
    """Return ``(token_set, joined_string)`` for fuzzy team-name comparison.

    Lower-cases, strips accents and punctuation, and removes club-form stopwords.
    The joined string backs character-level similarity for abbreviations that
    share no whole tokens (e.g. "RBL" vs "RB Leipzig").
    """
    decomposed = unicodedata.normalize("NFKD", name.lower())
    ascii_name = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_name)
    tokens = {t for t in cleaned.split() if t and t not in _TEAM_STOPWORDS}
    return tokens, "".join(sorted(tokens))


def _team_similarity(a: str, b: str) -> float:
    """Similarity in ``[0, 1]`` combining token-Jaccard and character ratio.

    Taking the maximum of the two measures handles both naming styles: Jaccard
    rewards shared whole words ("Bayern" vs "Bayern Munich"); the difflib
    character ratio rewards shared substrings when tokens diverge ("RBL" vs
    "RB Leipzig").
    """
    a_tokens, a_join = _normalize_team(a)
    b_tokens, b_join = _normalize_team(b)
    if not a_tokens or not b_tokens:
        return 0.0
    union = len(a_tokens | b_tokens)
    jaccard = len(a_tokens & b_tokens) / union if union else 0.0
    ratio = difflib.SequenceMatcher(None, a_join, b_join).ratio()
    return max(jaccard, ratio)


def _greedy_bijection(ws_teams: Iterable[str], fbref_teams: Iterable[str]) -> Dict[str, str]:
    """Greedily assign a 1:1 WhoScored->FBref team map by descending similarity.

    Both sets describe the same clubs, so the optimal assignment is a bijection.
    Curated aliases (:data:`_TEAM_ALIASES`) are seeded first; the remaining teams
    are matched by assigning the globally highest-similarity free pair first,
    which is exact here because correct pairs dominate incorrect ones by a wide
    margin (verified against the divergent German and English naming). Pairs below
    :data:`_NAME_MATCH_THRESHOLD` are left unmapped.
    """
    ws_set = set(ws_teams)
    fbref_set = set(fbref_teams)

    mapping: Dict[str, str] = {}
    used_fbref: Set[str] = set()
    for ws_name, fbref_name in _TEAM_ALIASES.items():
        if ws_name in ws_set and fbref_name in fbref_set:
            mapping[ws_name] = fbref_name
            used_fbref.add(fbref_name)

    scored: List[Tuple[float, str, str]] = []
    for ws in ws_set:
        for fb in fbref_set:
            scored.append((_team_similarity(ws, fb), ws, fb))
    scored.sort(key=lambda item: item[0], reverse=True)

    for score, ws, fb in scored:
        if score < _NAME_MATCH_THRESHOLD:
            break
        if ws in mapping or fb in used_fbref:
            continue
        mapping[ws] = fb
        used_fbref.add(fb)
    return mapping


def _schedule_teams(client: MongoClient, mongo: MongoConfig) -> Dict[Tuple[str, str], Set[str]]:
    """Collect the FBref team set for each (league, season) from the schedule."""
    collection = client[mongo.db][mongo.source_collection]
    pipeline = [
        {
            "$group": {
                "_id": {"league": "$league", "season": "$season"},
                "home": {"$addToSet": "$home_team"},
                "away": {"$addToSet": "$away_team"},
            }
        }
    ]
    result: Dict[Tuple[str, str], Set[str]] = {}
    for doc in collection.aggregate(pipeline):
        key = (doc["_id"]["league"], doc["_id"]["season"])
        teams = set(doc.get("home", [])) | set(doc.get("away", []))
        result[key] = {t for t in teams if isinstance(t, str)}
    return result


def build_event_target_index(
    client: MongoClient, mongo: MongoConfig
) -> Tuple[Dict[_GameKey, Dict[str, _TeamTargets]], Dict[str, int]]:
    """Aggregate per-match WhoScored targets, keyed for schedule joining.

    A single server-side aggregation over ``whoscored_events`` groups by
    ``(game_id, team)`` and counts shots on target, corners and cards. The match
    date is parsed from the ``game`` string prefix (events carry no date column).
    WhoScored team names are mapped to FBref names per (league, season) so the
    schedule can join on exact names; home/away orientation is resolved later by
    the schedule row, not here, which sidesteps own-goal / score ambiguities.

    Args:
        client: Open pymongo client.
        mongo: Mongo configuration.

    Returns:
        ``(index, stats)`` where ``index`` maps a game key to
        ``{fbref_team: (sot, corners, cards)}`` and ``stats`` carries coverage
        counters for logging.
    """
    schedule_teams = _schedule_teams(client, mongo)
    events = client[mongo.db][mongo.events_collection]
    pipeline = [
        {
            "$group": {
                "_id": {"game_id": "$game_id", "team": "$team"},
                "game": {"$first": "$game"},
                "league": {"$first": "$_league"},
                "season": {"$first": "$_season"},
                "sot": {"$sum": {"$cond": [{"$in": ["$type", list(_SOT_TYPES)]}, 1, 0]}},
                "corners": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$eq": ["$type", _CORNER_TYPE]},
                                    {"$eq": ["$outcome_type", _CORNER_OUTCOME]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "cards": {"$sum": {"$cond": [{"$eq": ["$type", _CARD_TYPE]}, 1, 0]}},
            }
        }
    ]

    raw: Dict[Tuple[str, str, object], Dict[str, object]] = {}
    ws_teams: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for doc in events.aggregate(pipeline, allowDiskUse=True):
        league = doc.get("league")
        season = doc.get("season")
        team = doc["_id"].get("team")
        game_id = doc["_id"].get("game_id")
        if not (isinstance(league, str) and isinstance(season, str) and isinstance(team, str)):
            continue
        ws_teams[(league, season)].add(team)
        game_record = raw.setdefault((league, season, game_id), {"game": doc.get("game"), "teams": {}})
        game_record["teams"][team] = (  # type: ignore[index]
            int(doc.get("sot", 0)),
            int(doc.get("corners", 0)),
            int(doc.get("cards", 0)),
        )

    name_maps: Dict[Tuple[str, str], Dict[str, str]] = {}
    unmapped: List[str] = []
    for key, ws_set in ws_teams.items():
        mapping = _greedy_bijection(ws_set, schedule_teams.get(key, set()))
        name_maps[key] = mapping
        unmapped.extend(f"{key[0]}/{key[1]}:{t}" for t in ws_set if t not in mapping)

    index: Dict[_GameKey, Dict[str, _TeamTargets]] = {}
    games_indexed = 0
    games_skipped = 0
    for (league, season, _game_id), payload in raw.items():
        game_str = payload.get("game")
        teams_map = payload.get("teams", {})
        if not isinstance(game_str, str) or len(teams_map) != 2:  # type: ignore[arg-type]
            games_skipped += 1
            continue
        name_map = name_maps.get((league, season), {})
        mapped: Dict[str, _TeamTargets] = {}
        for ws_team, targets in teams_map.items():  # type: ignore[union-attr]
            fbref_team = name_map.get(ws_team)
            if fbref_team is not None:
                mapped[fbref_team] = targets
        if len(mapped) != 2:
            games_skipped += 1
            continue
        game_key: _GameKey = (league, season, game_str[:10], frozenset(mapped.keys()))
        index[game_key] = mapped
        games_indexed += 1

    if unmapped:
        logger.warning("Unmapped WhoScored teams (their matches are masked): %s", ", ".join(sorted(unmapped)))
    stats = {
        "games_indexed": games_indexed,
        "games_skipped": games_skipped,
        "unmapped_teams": len(unmapped),
    }
    return index, stats


def build_feature_table(
    mongo: Optional[MongoConfig] = None,
    feature_cfg: Optional[FeatureConfig] = None,
) -> Tuple[Vocabulary, List[str], int]:
    """Build the processed feature collection from the raw schedule.

    Pipeline:
        1. Build deterministic categorical vocabularies (one cheap pass).
        2. Stream fixtures in chronological order; for each match emit leak-free
           rolling features, encode categoricals, attach the label and a global
           chronological ``row_idx``.
        3. Replace the processed collection and index it for temporal slicing.

    Multi-task targets: per-match shots on target, corners and cards are
    aggregated from ``whoscored_events`` (see :func:`build_event_target_index`)
    and joined onto each schedule row. A match with no WhoScored counterpart
    receives ``targets_present = 0`` so the training loop can mask its regression
    losses while still using it to train the outcome head.

    Args:
        mongo: Mongo settings (defaults to :class:`MongoConfig`).
        feature_cfg: Feature settings (defaults to :class:`FeatureConfig`).

    Returns:
        ``(vocabulary, numeric_feature_names, n_rows)``.
    """
    mongo = mongo or MongoConfig()
    feature_cfg = feature_cfg or FeatureConfig()
    feature_names = numeric_feature_names()

    client: MongoClient = MongoClient(mongo.uri)
    try:
        logger.info("Building categorical vocabularies.")
        vocab = _build_vocabulary(_iter_schedule(client, mongo))
        logger.info(
            "Vocabulary: %d teams, %d leagues, %d seasons.",
            vocab.num_teams, vocab.num_leagues, vocab.num_seasons,
        )

        spans = _season_match_spans(client, mongo)
        builder = FeatureBuilder(feature_cfg)

        logger.info("Aggregating WhoScored regression targets.")
        target_index, target_stats = build_event_target_index(client, mongo)
        logger.info(
            "WhoScored targets: %d games indexed, %d skipped, %d unmapped teams.",
            target_stats["games_indexed"], target_stats["games_skipped"],
            target_stats["unmapped_teams"],
        )

        target = client[mongo.db][mongo.feature_collection]
        target.drop()
        targets_present_total = 0

        batch: List[Dict[str, object]] = []
        row_idx = 0
        skipped = 0
        for doc in _iter_schedule(client, mongo):
            parsed = parse_score(doc.get("score"))  # type: ignore[arg-type]
            match_date = _coerce_date(doc.get("date"))
            home_team = doc.get("home_team")
            away_team = doc.get("away_team")
            league = doc.get("league")
            season = doc.get("season")
            if (
                parsed is None
                or match_date is None
                or not isinstance(home_team, str)
                or not isinstance(away_team, str)
                or not isinstance(league, str)
                or not isinstance(season, str)
            ):
                skipped += 1
                continue

            home_goals, away_goals = parsed
            week = doc.get("week")
            week_value = float(week) if isinstance(week, (int, float)) else None
            span = spans.get((league, season), 0.0)

            # Join the WhoScored regression targets by (league, season, date,
            # team pair). Orientation (home/away) is taken from the schedule row.
            # This is resolved before ``process`` so the realised shots on target
            # can feed the post-emit SOT EMA; the written target values are
            # unchanged, and the SOT is consumed only after features are emitted,
            # so the causality (leak-free) guarantee holds.
            date_str = match_date.strftime("%Y-%m-%d")
            game_key: _GameKey = (league, season, date_str, frozenset({home_team, away_team}))
            game_targets = target_index.get(game_key)
            if (
                game_targets is not None
                and home_team in game_targets
                and away_team in game_targets
            ):
                h_sot, h_corners, h_cards = game_targets[home_team]
                a_sot, a_corners, a_cards = game_targets[away_team]
                targets_present = 1
                targets_present_total += 1
                home_sot_obs: Optional[float] = float(h_sot)
                away_sot_obs: Optional[float] = float(a_sot)
            else:
                h_sot = h_corners = h_cards = 0
                a_sot = a_corners = a_cards = 0
                targets_present = 0
                home_sot_obs = None
                away_sot_obs = None

            features = builder.process(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                home_sot=home_sot_obs,
                away_sot=away_sot_obs,
                match_date=match_date,
                week=week_value,
                season_match_span=span,
            )

            record: Dict[str, object] = {
                "row_idx": row_idx,
                "game": doc.get("game"),
                "game_id": doc.get("game_id"),
                "date": match_date,
                "league": league,
                "season": season,
                "home_team_idx": vocab.team_index(home_team),
                "away_team_idx": vocab.team_index(away_team),
                "league_idx": vocab.league_index(league),
                "season_idx": vocab.season_index(season),
                "label": outcome_label(home_goals, away_goals),
                # Poisson goal targets (always present; source of the derived 1X2).
                "home_goals": float(home_goals),
                "away_goals": float(away_goals),
                # Multi-task regression targets (order matches REGRESSION_TARGETS).
                "home_sot": float(h_sot),
                "away_sot": float(a_sot),
                "home_corners": float(h_corners),
                "away_corners": float(a_corners),
                "home_cards": float(h_cards),
                "away_cards": float(a_cards),
                "targets_present": targets_present,
            }
            for name in feature_names:
                record[name] = float(features[name])

            batch.append(record)
            row_idx += 1
            if len(batch) >= 1000:
                target.insert_many(batch)
                batch = []

        if batch:
            target.insert_many(batch)

        target.create_index([("row_idx", ASCENDING)], unique=True)
        target.create_index([("label", ASCENDING)])

        coverage = (targets_present_total / row_idx * 100.0) if row_idx else 0.0
        logger.info(
            "Feature table built: %d rows written, %d skipped (unparseable). "
            "Regression targets present on %d/%d rows (%.1f%%).",
            row_idx, skipped, targets_present_total, row_idx, coverage,
        )
        return vocab, feature_names, row_idx
    finally:
        client.close()


def build_team_states(
    mongo: Optional[MongoConfig] = None,
    feature_cfg: Optional[FeatureConfig] = None,
) -> Tuple["FeatureBuilder", Dict[Tuple[str, str], float]]:
    """Replay the whole chronological schedule to recover every team's live state.

    This is the inference counterpart to :func:`build_feature_table`. It runs the
    identical leak-free pass -- same date ordering, same WhoScored shots-on-target
    join feeding the SOT EMA -- but writes nothing. It returns the populated
    :class:`FeatureBuilder`, whose internal per-team state now holds each club's
    post-history ELO and exponential moving averages, together with the per-(league,
    season) matchweek spans. An upcoming, unplayed fixture is then featurised with
    :meth:`FeatureBuilder.peek_match_features`, guaranteeing the prediction sees the
    exact same feature construction the model was trained on.

    Args:
        mongo: Mongo settings (defaults to :class:`MongoConfig`).
        feature_cfg: Feature settings (defaults to :class:`FeatureConfig`).

    Returns:
        ``(builder, spans)`` where ``builder`` carries current team state and
        ``spans`` maps ``(league, season)`` to its maximum matchweek.
    """
    mongo = mongo or MongoConfig()
    feature_cfg = feature_cfg or FeatureConfig()

    client: MongoClient = MongoClient(mongo.uri)
    try:
        spans = _season_match_spans(client, mongo)
        target_index, _ = build_event_target_index(client, mongo)
        builder = FeatureBuilder(feature_cfg)

        for doc in _iter_schedule(client, mongo):
            parsed = parse_score(doc.get("score"))  # type: ignore[arg-type]
            match_date = _coerce_date(doc.get("date"))
            home_team = doc.get("home_team")
            away_team = doc.get("away_team")
            league = doc.get("league")
            season = doc.get("season")
            if (
                parsed is None
                or match_date is None
                or not isinstance(home_team, str)
                or not isinstance(away_team, str)
                or not isinstance(league, str)
                or not isinstance(season, str)
            ):
                continue

            home_goals, away_goals = parsed
            date_str = match_date.strftime("%Y-%m-%d")
            game_key: _GameKey = (league, season, date_str, frozenset({home_team, away_team}))
            game_targets = target_index.get(game_key)
            if (
                game_targets is not None
                and home_team in game_targets
                and away_team in game_targets
            ):
                home_sot_obs: Optional[float] = float(game_targets[home_team][0])
                away_sot_obs: Optional[float] = float(game_targets[away_team][0])
            else:
                home_sot_obs = None
                away_sot_obs = None

            week = doc.get("week")
            week_value = float(week) if isinstance(week, (int, float)) else None
            span = spans.get((league, season), 0.0)
            builder.process(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                home_sot=home_sot_obs,
                away_sot=away_sot_obs,
                match_date=match_date,
                week=week_value,
                season_match_span=span,
            )
        return builder, spans
    finally:
        client.close()


def has_team_history(builder: "FeatureBuilder", team: str) -> bool:
    """Return whether ``team`` appears in the replayed state (has prior matches)."""
    state = builder._states.get(team)  # noqa: SLF001 - inference helper in same module
    return state is not None and state.played > 0
