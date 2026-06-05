"""Pure domain models — zero external dependencies."""

from dataclasses import dataclass, field
from typing import Any, Optional


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
    """A named batch of records destined for a single MongoDB collection.

    A batch carries its records in one of two mutually exclusive ways:

    - **In-memory** (``records``): the default, used by small batches such as
      FBref tables. The records cross any process boundary as a pickled list.
    - **Spilled to disk** (``jsonl_path``): used by very large batches (a full
      WhoScored season is ~580k events). The producing process streams the
      records to a line-delimited JSON file and only the path crosses the
      process boundary, so neither the worker nor the parent ever holds the
      whole list in memory. ``record_count`` reports the spilled row count.

    Attributes:
        collection: Target MongoDB collection name.
        records: Flat Python dicts, each representing one document. Empty when
            the batch is spilled to disk.
        jsonl_path: Path to a line-delimited JSON spill file, or ``None`` for an
            in-memory batch.
        record_count: Number of records in the batch. For spilled batches this
            is set explicitly; for in-memory batches it tracks ``len(records)``.
    """

    collection: str
    records: list[dict[str, Any]] = field(default_factory=list)
    jsonl_path: Optional[str] = None
    record_count: int = 0

    def __len__(self) -> int:
        return self.record_count or len(self.records)
