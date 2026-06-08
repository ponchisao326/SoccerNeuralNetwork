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
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from ml.config import NUM_CLASSES


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
        # ``prev``. Each is a single linear projection; the regression heads apply
        # a ReLU in ``forward`` to enforce non-negativity.
        self.head_outcome = nn.Linear(prev, NUM_CLASSES)
        self.head_sot = nn.Linear(prev, 2)
        self.head_corners = nn.Linear(prev, 2)
        self.head_cards = nn.Linear(prev, 2)

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
                ``outcome``: ``(B, 3)`` raw class logits (no softmax).
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
            "outcome": self.head_outcome(latent),
            "sot": F.relu(self.head_sot(latent)),
            "corners": F.relu(self.head_corners(latent)),
            "cards": F.relu(self.head_cards(latent)),
        }
