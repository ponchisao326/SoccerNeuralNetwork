"""PyTorch multi-task model for joint match-statistics prediction.

Architecture rationale
-----------------------
The input is heterogeneous: a handful of high-cardinality categorical identifiers
(home team, away team, league, season) plus a block of continuous rolling-form
features. The network handles each appropriately and then branches into several
task-specific heads (multi-task learning, MTL).

Shared representation
~~~~~~~~~~~~~~~~~~~~~~~
* **Categorical -> embeddings.** Teams are represented by a *shared*
  ``nn.Embedding`` table indexed by both the home and away slots. One-hot
  encoding would produce a sparse ~100-wide vector per team with no notion of
  similarity; a dense embedding instead lets the model learn a low-dimensional
  latent "strength" vector per team and place similar teams near each other.
  Sharing the table between slots ties a team's identity to a single vector
  regardless of venue. League and season get their own small embeddings.

* **Continuous -> standardised dense block.** The numeric features feed a
  multilayer perceptron whose blocks are ``Linear -> BatchNorm1d -> ReLU ->
  Dropout`` (BatchNorm stabilises/accelerates optimisation; Dropout regularises).

The MLP output is a shared latent match representation. MTL is motivated by the
fact that match outcome, shot volume, corners and cards are correlated facets of
the same latent quantities (relative team strength, game state, tempo). Forcing
one backbone to serve all tasks acts as an inductive bias / regulariser: the
auxiliary regression tasks inject gradient signal that improves the shared
features and combats overfitting on the harder 1X2 classification task.

Task heads (branch off the shared representation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``head_outcome``: ``Linear -> 3`` raw logits. No softmax: ``CrossEntropyLoss``
  fuses ``log_softmax`` + NLL internally for numerical stability.
* ``head_sot`` / ``head_corners`` / ``head_cards``: ``Linear -> 2`` (home, away)
  followed by ``ReLU``. The targets are non-negative event counts, so a ReLU
  output activation constrains predictions to the valid ``[0, inf)`` range,
  removing the impossible negative region from the hypothesis space and easing
  optimisation of the regression losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ml.config import MAX_GOALS, NUM_CLASSES


@dataclass(frozen=True)
class ModelDimensions:
    """Vocabulary sizes and input width needed to instantiate the model.

    Attributes:
        num_teams: Team vocabulary size (including ``<unk>``).
        num_leagues: League vocabulary size.
        num_seasons: Season vocabulary size.
        num_numeric_features: Width of the continuous feature block.
    """

    num_teams: int
    num_leagues: int
    num_seasons: int
    num_numeric_features: int


class MatchStatsMultiTaskPredictor(nn.Module):
    """Shared-backbone MTL model predicting outcome, shots on target, corners, cards.

    Args:
        dims: Vocabulary sizes and numeric input width.
        team_embedding_dim: Latent width of the shared team embedding.
        league_embedding_dim: Latent width of the league embedding.
        season_embedding_dim: Latent width of the season embedding.
        hidden_dims: Hidden-layer widths of the shared dense trunk.
        dropout: Dropout probability per hidden block.
    """

    def __init__(
        self,
        dims: ModelDimensions,
        team_embedding_dim: int,
        league_embedding_dim: int,
        season_embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()

        # Shared team table (index 0 == <unk>). Used for both home and away.
        self.team_embedding = nn.Embedding(dims.num_teams, team_embedding_dim)
        self.league_embedding = nn.Embedding(dims.num_leagues, league_embedding_dim)
        self.season_embedding = nn.Embedding(dims.num_seasons, season_embedding_dim)

        embedding_width = (
            2 * team_embedding_dim  # home + away from the shared table
            + league_embedding_dim
            + season_embedding_dim
        )
        input_dim = embedding_width + dims.num_numeric_features

        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = width
        self.backbone = nn.Sequential(*layers)

        # Task-specific heads branching off the shared representation of width
        # ``prev``. ``head_goals`` predicts the two Poisson rates (home, away
        # expected goals); the 1X2 outcome is derived from them analytically (see
        # :func:`poisson_outcome_probs`) rather than predicted by a separate
        # softmax. The remaining heads regress non-negative event counts.
        self.head_goals = nn.Linear(prev, 2)
        self.head_sot = nn.Linear(prev, 2)
        self.head_corners = nn.Linear(prev, 2)
        self.head_cards = nn.Linear(prev, 2)

        # Dixon-Coles dependence parameter: a single global scalar (not match
        # specific) shared across the dataset, learned jointly with the network.
        # Initialised to 0 so training starts from the independent-Poisson model
        # and only departs from it if the data supports extra draw mass.
        self.dixon_coles_rho = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise embeddings (small normal) and linear layers (Kaiming)."""
        for embedding in (self.team_embedding, self.league_embedding, self.season_embedding):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.05)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all task outputs from one shared forward pass.

        Args:
            numeric: ``(B, num_numeric_features)`` standardised continuous block.
            categorical: ``(B, 4)`` long tensor of
                ``[home_team, away_team, league, season]`` indices.

        Returns:
            Dict with keys:
                ``goals``: ``(B, 2)`` strictly positive home/away expected goals
                    (Poisson rates). ``softplus`` keeps them in ``(0, inf)`` while
                    staying smooth near zero, unlike ``exp`` which can explode.
                ``sot``: ``(B, 2)`` non-negative home/away shots on target.
                ``corners``: ``(B, 2)`` non-negative home/away corners.
                ``cards``: ``(B, 2)`` non-negative home/away cards.
        """
        home = self.team_embedding(categorical[:, 0])
        away = self.team_embedding(categorical[:, 1])
        league = self.league_embedding(categorical[:, 2])
        season = self.season_embedding(categorical[:, 3])

        features = torch.cat([home, away, league, season, numeric], dim=1)
        latent = self.backbone(features)

        return {
            "goals": F.softplus(self.head_goals(latent)),
            "rho": self.dixon_coles_rho,
            "sot": F.relu(self.head_sot(latent)),
            "corners": F.relu(self.head_corners(latent)),
            "cards": F.relu(self.head_cards(latent)),
        }


def poisson_outcome_probs(
    goals: torch.Tensor,
    rho: Optional[torch.Tensor] = None,
    max_goals: int = MAX_GOALS,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Derive the 1X2 distribution from predicted home/away goal rates.

    Treating the two scorelines as Poisson variables ``H ~ Pois(l_h)`` and
    ``A ~ Pois(l_a)``, the outcome probabilities are the mass of the joint score
    grid above, on and below the diagonal::

        P(home win) = sum_{i>j} P(H=i, A=j)
        P(draw)     = sum_{i=j} P(H=i, A=j)
        P(away win) = sum_{i<j} P(H=i, A=j)

    Under independence ``P(H=i, A=j) = P(H=i) P(A=j)``. The marginal PMF is
    evaluated in log space for stability
    (``log P(X=k) = -l + k*log(l) - lgamma(k+1)``) over ``k in [0, max_goals]``.

    **Dixon-Coles low-score correction.** Independent Poisson systematically
    under-predicts draws because it cannot represent the positive dependence
    between the two low scorelines. Following Dixon and Coles (1997) the four
    lowest cells are multiplied by

        tau(0,0) = 1 - l_h*l_a*rho,   tau(0,1) = 1 + l_h*rho,
        tau(1,0) = 1 + l_a*rho,       tau(1,1) = 1 - rho,

    (``tau = 1`` elsewhere). A negative ``rho`` inflates the exact draws 0-0 and
    1-1 and deflates the 1-0 / 0-1 wins, lifting the draw probability for tight,
    low-scoring fixtures -- precisely where real draws cluster. ``rho`` is a single
    learnable parameter; at ``rho = 0`` the model collapses back to independent
    Poisson. Corrected cells are floored at zero to keep the simplex valid for the
    (negligible) high-rate edge cases. The result is differentiable in both
    ``(l_h, l_a)`` and ``rho`` and is renormalised. Column order matches
    :data:`ml.config.CLASS_NAMES` (``home_win, draw, away_win``).

    Args:
        goals: ``(B, 2)`` strictly positive expected goals ``[l_home, l_away]``.
        rho: Optional scalar Dixon-Coles dependence parameter. ``None`` or ``0``
            yields the independent-Poisson derivation.
        max_goals: Inclusive upper bound of the goal grid.
        eps: Floor added before logarithms and the final normalisation.

    Returns:
        ``(B, 3)`` outcome probability simplex.
    """
    lam_home = goals[:, 0].clamp_min(eps)
    lam_away = goals[:, 1].clamp_min(eps)

    k = torch.arange(max_goals + 1, device=goals.device, dtype=goals.dtype)
    log_factorial = torch.lgamma(k + 1.0)  # (G+1,)

    def _log_pmf(lam: torch.Tensor) -> torch.Tensor:
        # (B, G+1): log Poisson PMF for each goal count.
        return (
            -lam.unsqueeze(1)
            + k.unsqueeze(0) * torch.log(lam.unsqueeze(1))
            - log_factorial.unsqueeze(0)
        )

    p_home = torch.exp(_log_pmf(lam_home))  # (B, G+1)
    p_away = torch.exp(_log_pmf(lam_away))  # (B, G+1)
    joint = p_home.unsqueeze(2) * p_away.unsqueeze(1)  # (B, i, j)

    if rho is not None:
        # Bound the dependence so all tau stay positive for realistic rates.
        r = 0.15 * torch.tanh(rho).reshape(())
        tau = torch.ones_like(joint)
        tau[:, 0, 0] = 1.0 - lam_home * lam_away * r
        tau[:, 0, 1] = 1.0 + lam_home * r
        tau[:, 1, 0] = 1.0 + lam_away * r
        tau[:, 1, 1] = 1.0 - r
        joint = (joint * tau).clamp_min(0.0)

    i_idx = k.unsqueeze(1)  # (G+1, 1)
    j_idx = k.unsqueeze(0)  # (1, G+1)
    home_mask = (i_idx > j_idx).to(joint.dtype)
    draw_mask = (i_idx == j_idx).to(joint.dtype)
    away_mask = (i_idx < j_idx).to(joint.dtype)

    p_home_win = (joint * home_mask).sum(dim=(1, 2))
    p_draw = (joint * draw_mask).sum(dim=(1, 2))
    p_away_win = (joint * away_mask).sum(dim=(1, 2))

    probs = torch.stack([p_home_win, p_draw, p_away_win], dim=1)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)
