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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

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
    # Venue-conditioned form: a club's home form and away form are distinct
    # processes (crowd, travel, tactical posture), so each side's block reads the
    # EMA of the venue it will actually play at (home side -> home-only history,
    # away side -> away-only history).
    "venue_gf_ema",    # EMA of goals scored at the relevant venue
    "venue_ga_ema",    # EMA of goals conceded at the relevant venue
    "venue_pts_ema",   # EMA of points earned at the relevant venue
    "venue_played",    # prior matches at that venue (EMA maturity signal)
    "congestion14",    # matches played in the trailing 14 days (fixture pile-up)
    "new_to_season",   # 1.0 if the team did not appear in the previous season
    # WhoScored event-volume EMAs (Big-5 only; matches without coverage leave
    # the trackers untouched, exactly like the existing sot_ema convention).
    "shots_ema",          # EMA of total shot attempts (attacking volume)
    "shots_against_ema",  # EMA of opponents' shot attempts (defensive suppression)
    "box_shots_ema",      # EMA of penalty-area shot attempts (chance quality)
    "deep_comp_ema",      # EMA of completed final-quarter passes (territory)
    "ws_cov",             # 1.0 if the team has any WhoScored-covered prior match
)
# NOTE: previous-season squad priors (build_squad_priors) were evaluated as ten
# additional per-side features and REMOVED: the 3-seed validation gate showed no
# gain over ELO + team embeddings (val log-loss 1.0036 -> 1.0059) and a test
# accuracy drop. Keep build_squad_priors for future revisits, but do not re-add
# the features without a fresh gate.

# Absolute pre-match ELO ratings of each side. Named explicitly (not h_/a_
# prefixed) because ELO is a relational rating maintained on a shared scale rather
# than an independent per-side rolling statistic.
_ELO_FEATURES: Tuple[str, ...] = (
    "home_elo",
    "away_elo",
)

_DIFF_FEATURES: Tuple[str, ...] = (
    "gf_ema_diff",        # home gf_ema - away gf_ema
    "ga_ema_diff",        # home ga_ema - away ga_ema
    "pts_ema_diff",       # home pts_ema - away pts_ema
    "elo_diff",           # home_elo - away_elo (signed strength gap)
    "venue_pts_ema_diff",  # home venue_pts_ema - away venue_pts_ema
)

# Head-to-head block between the two clubs of the fixture. The goal-difference
# EMA is sign-oriented to the current home team (positive = the home club has
# historically beaten this opponent).
_H2H_FEATURES: Tuple[str, ...] = (
    "h2h_played",   # prior meetings observed (capped; maturity signal)
    "h2h_gd_ema",   # EMA of (home-club goals - away-club goals) in prior meetings
)

_GLOBAL_FEATURES: Tuple[str, ...] = (
    "season_progress",  # matchweek normalised to [0, 1]; captures fatigue / table pressure
)

# Cap on the emitted head-to-head meeting count; beyond ~10 meetings the EMA is
# mature and a larger raw count only adds scale noise.
_H2H_PLAYED_CAP: int = 10

# Per-team rolling date memory used for the congestion feature. A 14-day window
# can physically hold at most ~7 fixtures; 15 entries is a safe upper bound.
_RECENT_DATES_MAXLEN: int = 15


def numeric_feature_names() -> List[str]:
    """Return the ordered list of numeric feature column names.

    The model's numeric input width equals ``len(numeric_feature_names())``.
    """
    names: List[str] = [f"h_{f}" for f in _PER_SIDE_FEATURES]
    names += [f"a_{f}" for f in _PER_SIDE_FEATURES]
    names += list(_ELO_FEATURES)
    names += list(_DIFF_FEATURES)
    names += list(_H2H_FEATURES)
    names += list(_GLOBAL_FEATURES)
    return names


def _prev_season(season: str) -> Optional[str]:
    """Return the season code immediately before ``season`` (``"2526" -> "2425"``).

    Season codes are two two-digit years (start, end). Returns ``None`` when the
    code does not follow that format.
    """
    if len(season) != 4 or not season.isdigit():
        return None
    start_year = int(season[:2])
    return f"{start_year - 1:02d}{start_year:02d}"


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
        home_gf_ema: EMA of goals scored in home matches only.
        home_ga_ema: EMA of goals conceded in home matches only.
        home_pts_ema: EMA of points earned in home matches only.
        home_played: Count of prior home matches.
        away_gf_ema: EMA of goals scored in away matches only.
        away_ga_ema: EMA of goals conceded in away matches only.
        away_pts_ema: EMA of points earned in away matches only.
        away_played: Count of prior away matches.
        recent_dates: Bounded memory of recent match dates for the congestion
            feature (matches in the trailing window).
        shots_ema: EMA of total shot attempts (WhoScored coverage only).
        shots_against_ema: EMA of opponents' shot attempts.
        box_shots_ema: EMA of penalty-area shot attempts.
        deep_comp_ema: EMA of completed final-quarter passes.
        ws_played: Count of prior matches with WhoScored coverage.
    """

    elo: float
    gf_ema: Optional[float] = None
    ga_ema: Optional[float] = None
    sot_ema: Optional[float] = None
    pts_ema: Optional[float] = None
    played: int = 0
    last_date: Optional[datetime] = None
    home_gf_ema: Optional[float] = None
    home_ga_ema: Optional[float] = None
    home_pts_ema: Optional[float] = None
    home_played: int = 0
    away_gf_ema: Optional[float] = None
    away_ga_ema: Optional[float] = None
    away_pts_ema: Optional[float] = None
    away_played: int = 0
    recent_dates: Deque[datetime] = field(
        default_factory=lambda: deque(maxlen=_RECENT_DATES_MAXLEN)
    )
    shots_ema: Optional[float] = None
    shots_against_ema: Optional[float] = None
    box_shots_ema: Optional[float] = None
    deep_comp_ema: Optional[float] = None
    ws_played: int = 0

    @classmethod
    def empty(cls, elo_base: float) -> "_TeamState":
        return cls(elo=elo_base)


@dataclass
class _H2HState:
    """Head-to-head memory for one unordered club pair.

    The goal-difference EMA is stored oriented to the lexicographically smaller
    club name of the pair; emit/update sites flip the sign when the current home
    team is the other club, so a single tracker serves both orientations.

    Attributes:
        gd_ema: EMA of the oriented goal difference, ``None`` before any meeting.
        played: Count of prior meetings folded in.
    """

    gd_ema: Optional[float] = None
    played: int = 0


class FeatureBuilder:
    """Builds leak-free per-match feature rows from chronological fixtures.

    The builder is stateful: feed it matches in ascending date order via
    :meth:`process`. It is *not* safe to feed out-of-order matches, as that would
    corrupt the rolling state and violate the causality guarantee.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self._cfg = config
        self._states: Dict[str, _TeamState] = {}
        self._h2h: Dict[FrozenSet[str], _H2HState] = {}
        # Seasons each team has actually played in, for the promoted/newly-seen
        # flag. Membership is recorded at update time (post-emit), so the flag a
        # match observes never includes that match's own season appearance.
        self._team_seasons: Dict[str, Set[str]] = defaultdict(set)

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

    def _new_to_season(self, team: str, season: str) -> float:
        """Promoted/newly-seen flag: 1.0 if the team has no prior-season matches.

        Uses only strictly prior appearances (membership is recorded post-emit),
        so the flag is leak-free. The earliest ingested season flags every team,
        which the season embedding can absorb.
        """
        previous = _prev_season(season)
        if previous is None:
            return 0.0
        return 0.0 if previous in self._team_seasons[team] else 1.0

    def _congestion(self, state: _TeamState, match_date: datetime) -> float:
        """Count prior matches inside the trailing congestion window."""
        window = self._cfg.congestion_window_days
        return float(
            sum(1 for played_on in state.recent_dates if 0 <= (match_date - played_on).days <= window)
        )

    def _side_features(
        self, team: str, match_date: datetime, is_home: bool, season: str
    ) -> Dict[str, float]:
        """Compute the pre-match recency-weighted feature block for one team.

        The venue block reads the trackers of the venue the team will actually
        play at: the home side sees its home-only EMAs, the away side its
        away-only EMAs.
        """
        state = self._state(team)
        new_to_season = self._new_to_season(team, season)
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
                "venue_gf_ema": 0.0,
                "venue_ga_ema": 0.0,
                "venue_pts_ema": 0.0,
                "venue_played": 0.0,
                "congestion14": 0.0,
                "new_to_season": new_to_season,
                "shots_ema": 0.0,
                "shots_against_ema": 0.0,
                "box_shots_ema": 0.0,
                "deep_comp_ema": 0.0,
                "ws_cov": 0.0,
            }

        if state.last_date is not None:
            rest = (match_date - state.last_date).days
            rest = max(0.0, min(float(rest), self._cfg.max_rest_days))
        else:
            rest = self._cfg.default_rest_days

        if is_home:
            venue_gf, venue_ga = state.home_gf_ema, state.home_ga_ema
            venue_pts, venue_played = state.home_pts_ema, state.home_played
        else:
            venue_gf, venue_ga = state.away_gf_ema, state.away_ga_ema
            venue_pts, venue_played = state.away_pts_ema, state.away_played

        return {
            "gf_ema": self._emit(state.gf_ema),
            "ga_ema": self._emit(state.ga_ema),
            "sot_ema": self._emit(state.sot_ema),
            "pts_ema": self._emit(state.pts_ema),
            "rest_days": rest,
            "played": float(state.played),
            "venue_gf_ema": self._emit(venue_gf),
            "venue_ga_ema": self._emit(venue_ga),
            "venue_pts_ema": self._emit(venue_pts),
            "venue_played": float(venue_played),
            "congestion14": self._congestion(state, match_date),
            "new_to_season": new_to_season,
            "shots_ema": self._emit(state.shots_ema),
            "shots_against_ema": self._emit(state.shots_against_ema),
            "box_shots_ema": self._emit(state.box_shots_ema),
            "deep_comp_ema": self._emit(state.deep_comp_ema),
            "ws_cov": 1.0 if state.ws_played > 0 else 0.0,
        }

    def _update_team_state(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_ws: Optional["_WsTeamStats"],
        away_ws: Optional["_WsTeamStats"],
        match_date: datetime,
        season: str,
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
        home_sot = float(home_ws.sot) if home_ws is not None else None
        away_sot = float(away_ws.sot) if away_ws is not None else None

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

        # WhoScored volume EMAs: both sides' stats come from the same coverage,
        # so home/away presence is joint. A team's shots_against tracks the
        # opponent's shot volume (defensive suppression signal).
        if home_ws is not None and away_ws is not None:
            home_state.shots_ema = self._ema_step(home_state.shots_ema, float(home_ws.shots), alpha)
            home_state.shots_against_ema = self._ema_step(
                home_state.shots_against_ema, float(away_ws.shots), alpha
            )
            home_state.box_shots_ema = self._ema_step(
                home_state.box_shots_ema, float(home_ws.box_shots), alpha
            )
            home_state.deep_comp_ema = self._ema_step(
                home_state.deep_comp_ema, float(home_ws.deep_comp), alpha
            )
            home_state.ws_played += 1
            away_state.shots_ema = self._ema_step(away_state.shots_ema, float(away_ws.shots), alpha)
            away_state.shots_against_ema = self._ema_step(
                away_state.shots_against_ema, float(home_ws.shots), alpha
            )
            away_state.box_shots_ema = self._ema_step(
                away_state.box_shots_ema, float(away_ws.box_shots), alpha
            )
            away_state.deep_comp_ema = self._ema_step(
                away_state.deep_comp_ema, float(away_ws.deep_comp), alpha
            )
            away_state.ws_played += 1

        # Venue-conditioned trackers: this match contributes to the home team's
        # home-only history and the away team's away-only history.
        home_state.home_gf_ema = self._ema_step(home_state.home_gf_ema, float(home_goals), alpha)
        home_state.home_ga_ema = self._ema_step(home_state.home_ga_ema, float(away_goals), alpha)
        home_state.home_pts_ema = self._ema_step(home_state.home_pts_ema, home_points, alpha)
        home_state.home_played += 1
        away_state.away_gf_ema = self._ema_step(away_state.away_gf_ema, float(away_goals), alpha)
        away_state.away_ga_ema = self._ema_step(away_state.away_ga_ema, float(home_goals), alpha)
        away_state.away_pts_ema = self._ema_step(away_state.away_pts_ema, away_points, alpha)
        away_state.away_played += 1

        home_state.recent_dates.append(match_date)
        away_state.recent_dates.append(match_date)
        self._team_seasons[home_team].add(season)
        self._team_seasons[away_team].add(season)

        # Head-to-head tracker, oriented to the lexicographically smaller club so
        # one EMA serves both fixture orientations.
        pair = frozenset({home_team, away_team})
        h2h = self._h2h.get(pair)
        if h2h is None:
            h2h = _H2HState()
            self._h2h[pair] = h2h
        oriented_gd = float(home_goals - away_goals)
        if home_team > away_team:
            oriented_gd = -oriented_gd
        h2h.gd_ema = self._ema_step(h2h.gd_ema, oriented_gd, alpha)
        h2h.played += 1

    def peek_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        week: Optional[float],
        season_match_span: float,
        league: str,
        season: str,
    ) -> Dict[str, float]:
        """Compute the pre-match feature vector WITHOUT mutating any team state.

        This is the read-only half of :meth:`process`: it produces the identical
        numeric features a training row receives, but leaves both teams' rolling
        state untouched. It is the inference entry point, where the upcoming
        fixture has no result to fold in. Calling it is therefore idempotent and
        order-free.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            match_date: Kickoff date (for the rest-days delta).
            week: Matchweek number, or ``None`` if absent.
            season_match_span: Matchweeks in the season, for ``season_progress``.
            league: League identifier (reserved for league-conditioned features).
            season: Season code (drives the promoted/newly-seen flag).

        Returns:
            Ordered numeric feature dict keyed by :func:`numeric_feature_names`.
        """
        del league  # Reserved: no league-conditioned numeric feature yet.

        # Pre-match ratings are read straight from current state (no mutation).
        home_elo = self._state(home_team).elo
        away_elo = self._state(away_team).elo

        home = self._side_features(home_team, match_date, is_home=True, season=season)
        away = self._side_features(away_team, match_date, is_home=False, season=season)

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
        features["venue_pts_ema_diff"] = home["venue_pts_ema"] - away["venue_pts_ema"]

        h2h = self._h2h.get(frozenset({home_team, away_team}))
        if h2h is None or h2h.played == 0:
            features["h2h_played"] = 0.0
            features["h2h_gd_ema"] = 0.0
        else:
            gd_ema = h2h.gd_ema if h2h.gd_ema is not None else 0.0
            if home_team > away_team:
                gd_ema = -gd_ema
            features["h2h_played"] = float(min(h2h.played, _H2H_PLAYED_CAP))
            features["h2h_gd_ema"] = gd_ema

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
        home_ws: Optional["_WsTeamStats"],
        away_ws: Optional["_WsTeamStats"],
        match_date: datetime,
        week: Optional[float],
        season_match_span: float,
        league: str,
        season: str,
    ) -> Dict[str, float]:
        """Emit the pre-match feature vector, then update both teams' state.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            home_goals: Realised home goals (used only for the post-emit update).
            away_goals: Realised away goals (post-emit update only).
            home_ws: Realised home-team WhoScored aggregates, or ``None`` when the
                match has no WhoScored coverage (post-emit EMA update only).
            away_ws: Realised away-team WhoScored aggregates, or ``None``.
            match_date: Kickoff date.
            week: Matchweek number, or ``None`` if absent.
            season_match_span: Number of matchweeks in the season, for
                normalising ``season_progress``.
            league: League identifier of the fixture.
            season: Season code of the fixture.

        Returns:
            Ordered numeric feature dict keyed by :func:`numeric_feature_names`.
        """
        features = self.peek_match_features(
            home_team, away_team, match_date, week, season_match_span, league, season
        )

        # Post-emit state update: strictly after features are read.
        self._update_team_state(
            home_team, away_team, home_goals, away_goals, home_ws, away_ws,
            match_date, season,
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
# Total shot volume for the attacking-pressure feature EMAs: every shot attempt,
# on target or not (woodwork included).
_SHOT_TYPES: FrozenSet[str] = frozenset({"Goal", "SavedShot", "MissedShots", "ShotOnPost"})
_PASS_TYPE: str = "Pass"
_SUCCESSFUL: str = "Successful"
# Opta pitch coordinates run 0-100 toward the attacked goal. x >= 83 is the
# penalty-area depth (shot-quality proxy); a completed pass ending at
# end_x >= 75 is a final-quarter entry (territory/creation proxy). Verified on
# the live collection: ~64% of shots fall in the box band, ~12% of completed
# passes in the deep band.
_BOX_X_MIN: float = 83.0
_DEEP_END_X_MIN: float = 75.0
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

@dataclass(frozen=True)
class _WsTeamStats:
    """Per-team, per-match WhoScored event aggregates.

    ``sot``/``corners``/``cards`` are the written regression targets (their
    values are byte-identical to the original tuple payload). The remaining
    counters never become targets: they feed the post-emit feature EMAs only.

    Attributes:
        sot: Shots on target (Goal + SavedShot).
        corners: Corners won (CornerAwarded, successful side only).
        cards: Cards received.
        shots: Total shot attempts (on target + missed + woodwork).
        box_shots: Shot attempts from the penalty-area band (``x >= 83``).
        deep_comp: Completed passes ending in the final quarter (``end_x >= 75``).
        passes: Pass attempts.
        passes_ok: Completed passes.
    """

    sot: int
    corners: int
    cards: int
    shots: int
    box_shots: int
    deep_comp: int
    passes: int
    passes_ok: int


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
) -> Tuple[Dict[_GameKey, Dict[str, _WsTeamStats]], Dict[str, int]]:
    """Aggregate per-match WhoScored stats, keyed for schedule joining.

    A single server-side aggregation over ``whoscored_events`` groups by
    ``(game_id, team)`` and counts the regression targets (shots on target,
    corners, cards) plus the feature-only volumes (total/box shots, deep
    completions, passes). The match date is parsed from the ``game`` string
    prefix (events carry no date column). WhoScored team names are mapped to
    FBref names per (league, season) so the schedule can join on exact names;
    home/away orientation is resolved later by the schedule row, not here, which
    sidesteps own-goal / score ambiguities.

    Args:
        client: Open pymongo client.
        mongo: Mongo configuration.

    Returns:
        ``(index, stats)`` where ``index`` maps a game key to
        ``{fbref_team: _WsTeamStats}`` and ``stats`` carries coverage counters
        for logging.
    """
    schedule_teams = _schedule_teams(client, mongo)
    events = client[mongo.db][mongo.events_collection]
    # Coordinate fields are occasionally absent; $ifNull maps them to -1 so the
    # band predicates stay well-defined (absent -> outside every band).
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
                "shots": {"$sum": {"$cond": [{"$in": ["$type", list(_SHOT_TYPES)]}, 1, 0]}},
                "box_shots": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$type", list(_SHOT_TYPES)]},
                                    {"$gte": [{"$ifNull": ["$x", -1.0]}, _BOX_X_MIN]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "deep_comp": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$eq": ["$type", _PASS_TYPE]},
                                    {"$eq": ["$outcome_type", _SUCCESSFUL]},
                                    {"$gte": [{"$ifNull": ["$end_x", -1.0]}, _DEEP_END_X_MIN]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "passes": {"$sum": {"$cond": [{"$eq": ["$type", _PASS_TYPE]}, 1, 0]}},
                "passes_ok": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$eq": ["$type", _PASS_TYPE]},
                                    {"$eq": ["$outcome_type", _SUCCESSFUL]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
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
        game_record["teams"][team] = _WsTeamStats(  # type: ignore[index]
            sot=int(doc.get("sot", 0)),
            corners=int(doc.get("corners", 0)),
            cards=int(doc.get("cards", 0)),
            shots=int(doc.get("shots", 0)),
            box_shots=int(doc.get("box_shots", 0)),
            deep_comp=int(doc.get("deep_comp", 0)),
            passes=int(doc.get("passes", 0)),
            passes_ok=int(doc.get("passes_ok", 0)),
        )

    name_maps: Dict[Tuple[str, str], Dict[str, str]] = {}
    unmapped: List[str] = []
    for key, ws_set in ws_teams.items():
        mapping = _greedy_bijection(ws_set, schedule_teams.get(key, set()))
        name_maps[key] = mapping
        unmapped.extend(f"{key[0]}/{key[1]}:{t}" for t in ws_set if t not in mapping)

    index: Dict[_GameKey, Dict[str, _WsTeamStats]] = {}
    games_indexed = 0
    games_skipped = 0
    for (league, season, _game_id), payload in raw.items():
        game_str = payload.get("game")
        teams_map = payload.get("teams", {})
        if not isinstance(game_str, str) or len(teams_map) != 2:  # type: ignore[arg-type]
            games_skipped += 1
            continue
        name_map = name_maps.get((league, season), {})
        mapped: Dict[str, _WsTeamStats] = {}
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

            # Join the WhoScored per-team aggregates by (league, season, date,
            # team pair). Orientation (home/away) is taken from the schedule row.
            # This is resolved before ``process`` so the realised event stats can
            # feed the post-emit feature EMAs; the written target values are
            # unchanged, and the stats are consumed only after features are
            # emitted, so the causality (leak-free) guarantee holds.
            date_str = match_date.strftime("%Y-%m-%d")
            game_key: _GameKey = (league, season, date_str, frozenset({home_team, away_team}))
            game_targets = target_index.get(game_key)
            if (
                game_targets is not None
                and home_team in game_targets
                and away_team in game_targets
            ):
                home_ws: Optional[_WsTeamStats] = game_targets[home_team]
                away_ws: Optional[_WsTeamStats] = game_targets[away_team]
                h_sot, h_corners, h_cards = home_ws.sot, home_ws.corners, home_ws.cards
                a_sot, a_corners, a_cards = away_ws.sot, away_ws.corners, away_ws.cards
                targets_present = 1
                targets_present_total += 1
            else:
                home_ws = None
                away_ws = None
                h_sot = h_corners = h_cards = 0
                a_sot = a_corners = a_cards = 0
                targets_present = 0

            features = builder.process(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                home_ws=home_ws,
                away_ws=away_ws,
                match_date=match_date,
                week=week_value,
                season_match_span=span,
                league=league,
                season=season,
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


@dataclass(frozen=True)
class SquadPriors:
    """Previous-season squad aggregates attached to one ``(team, season)`` key.

    Attributes:
        gls90: Squad goals per 90 (total goals / total 90s played).
        gk_save: Minutes-weighted goalkeeper save percentage (0-100 scale).
        min_hhi: Herfindahl index of squad minute shares; high = a short,
            settled rotation, low = heavy squad churn.
        age: Minutes-weighted squad age in years.
    """

    gls90: float
    gk_save: float
    min_hhi: float
    age: float


def _next_season(season: str) -> Optional[str]:
    """Return the season code immediately after ``season`` (``"2425" -> "2526"``)."""
    if len(season) != 4 or not season.isdigit():
        return None
    return f"{int(season[:2]) + 1:02d}{int(season[2:]) + 1:02d}"


def build_squad_priors(
    client: MongoClient, mongo: MongoConfig
) -> Dict[Tuple[str, str], SquadPriors]:
    """Aggregate squad-quality priors, keyed by the season they may be USED in.

    Currently UNUSED by the feature pipeline: wired in as ten per-side features,
    these priors failed their 3-seed validation gate (no val gain over ELO +
    team embeddings, lower test accuracy) and were removed. Kept as a verified
    utility for future experiments (e.g. cold-start-only usage).

    Leakage rule: FBref player docs are whole-season aggregates, so a match in
    season ``S`` may only ever observe season ``S-1`` values. The returned map is
    therefore keyed ``(team, S)`` while every value is computed from the
    ``(team, S-1)`` player documents. The join deliberately ignores league:
    promoted/relegated clubs change league but keep their FBref spelling, and a
    within-league join would drop exactly those clubs.

    Args:
        client: Open pymongo client.
        mongo: Mongo configuration.

    Returns:
        ``{(team, season): SquadPriors}`` for every team-season with a prior.
    """
    standard = client[mongo.db][mongo.player_standard_collection]
    keeper = client[mongo.db][mongo.player_keeper_collection]

    def _num(path: str) -> Dict[str, object]:
        """Defensive numeric cast: FBref occasionally stores numbers as strings
        (and ``age`` sometimes as ``"YY-DDD"``); take the leading integer part
        and fall back to 0 rather than failing the aggregation."""
        leading = {
            "$convert": {
                "input": {
                    "$arrayElemAt": [{"$split": [{"$toString": path}, "-"]}, 0]
                },
                "to": "double",
                "onError": 0,
                "onNull": 0,
            }
        }
        return {
            "$convert": {"input": path, "to": "double", "onError": leading, "onNull": 0}
        }

    # Squad outfield aggregates. HHI of minute shares is computed from
    # sum(min_i^2) / (sum(min_i))^2, both available server-side.
    standard_pipeline = [
        {
            "$group": {
                "_id": {"team": "$team", "season": "$season"},
                "gls": {"$sum": _num("$Performance_Gls")},
                "nineties": {"$sum": _num("$Playing Time_90s")},
                "minutes": {"$sum": _num("$Playing Time_Min")},
                "minutes_sq": {"$sum": {"$pow": [_num("$Playing Time_Min"), 2]}},
                "age_minutes": {
                    "$sum": {"$multiply": [_num("$age"), _num("$Playing Time_Min")]}
                },
            }
        }
    ]
    # Minutes-weighted GK save percentage.
    keeper_pipeline = [
        {
            "$group": {
                "_id": {"team": "$team", "season": "$season"},
                "save_minutes": {
                    "$sum": {
                        "$multiply": [_num("$Performance_Save%"), _num("$Playing Time_Min")]
                    }
                },
                "minutes": {"$sum": _num("$Playing Time_Min")},
            }
        }
    ]

    gk_save: Dict[Tuple[str, str], float] = {}
    for doc in keeper.aggregate(keeper_pipeline):
        team = doc["_id"].get("team")
        season = doc["_id"].get("season")
        minutes = float(doc.get("minutes", 0.0) or 0.0)
        if isinstance(team, str) and isinstance(season, str) and minutes > 0:
            gk_save[(team, season)] = float(doc.get("save_minutes", 0.0)) / minutes

    priors: Dict[Tuple[str, str], SquadPriors] = {}
    for doc in standard.aggregate(standard_pipeline):
        team = doc["_id"].get("team")
        season = doc["_id"].get("season")
        if not (isinstance(team, str) and isinstance(season, str)):
            continue
        use_season = _next_season(season)
        if use_season is None:
            continue
        nineties = float(doc.get("nineties", 0.0) or 0.0)
        minutes = float(doc.get("minutes", 0.0) or 0.0)
        if nineties <= 0 or minutes <= 0:
            continue
        priors[(team, use_season)] = SquadPriors(
            gls90=float(doc.get("gls", 0.0)) / nineties,
            gk_save=gk_save.get((team, season), 0.0),
            min_hhi=float(doc.get("minutes_sq", 0.0)) / (minutes * minutes),
            age=float(doc.get("age_minutes", 0.0)) / minutes,
        )
    return priors


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
                home_ws: Optional[_WsTeamStats] = game_targets[home_team]
                away_ws: Optional[_WsTeamStats] = game_targets[away_team]
            else:
                home_ws = None
                away_ws = None

            week = doc.get("week")
            week_value = float(week) if isinstance(week, (int, float)) else None
            span = spans.get((league, season), 0.0)
            builder.process(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                home_ws=home_ws,
                away_ws=away_ws,
                match_date=match_date,
                week=week_value,
                season_match_span=span,
                league=league,
                season=season,
            )
        return builder, spans
    finally:
        client.close()


def has_team_history(builder: "FeatureBuilder", team: str) -> bool:
    """Return whether ``team`` appears in the replayed state (has prior matches)."""
    state = builder._states.get(team)  # noqa: SLF001 - inference helper in same module
    return state is not None and state.played > 0
