"""Pure domain models — zero external dependencies."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExtractionTarget:
    """Identifies a single (league, season) extraction job.

    Attributes:
        league: soccerdata league identifier (e.g. "ENG-Premier League").
        season: soccerdata season string (e.g. "2425" for 2024-25).
    """

    league: str
    season: str


@dataclass
class ExtractedBatch:
    """A named batch of flat records destined for a single MongoDB collection.

    Attributes:
        collection: Target MongoDB collection name.
        records: Flat Python dicts, each representing one document.
    """

    collection: str
    records: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)
