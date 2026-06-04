"""Abstract port for all data-source adapters."""

from abc import ABC, abstractmethod

from etl.domain.models import ExtractedBatch, ExtractionTarget


class ExtractorPort(ABC):
    """Contract every data-source adapter must satisfy.

    Implementations are responsible for:
    - Wrapping blocking I/O in ``asyncio.to_thread`` or a ``ThreadPoolExecutor``.
    - Flattening pandas DataFrames (including MultiIndex) to plain Python dicts.
    - Tagging every record with ``_source``, ``_league``, and ``_season``.
    """

    @abstractmethod
    async def extract(self, target: ExtractionTarget) -> list[ExtractedBatch]:
        """Extract all relevant data for the given target.

        Args:
            target: League and season to extract.

        Returns:
            List of ``ExtractedBatch`` objects, one per logical collection.
            An empty list or batches with zero records are valid return values
            and must not be treated as errors by callers.
        """
        ...
