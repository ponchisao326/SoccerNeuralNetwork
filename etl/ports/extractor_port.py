"""Abstract port for all data-source adapters."""

from abc import ABC, abstractmethod

from etl.domain.models import ExtractedBatch, ExtractionTarget


class ExtractorPort(ABC):
    """Contract every data-source adapter must satisfy.

    Implementations are responsible for:
    - Wrapping blocking I/O in ``asyncio.to_thread`` or a ``ThreadPoolExecutor``.
    - Flattening pandas DataFrames (including MultiIndex) to plain Python dicts.
    - Tagging every record with ``_source``, ``_league``, and ``_season``.
    - Declaring their own safe concurrency via :attr:`max_concurrency`.
    """

    @property
    @abstractmethod
    def max_concurrency(self) -> int:
        """Maximum number of concurrent ``extract`` calls this adapter tolerates.

        The orchestrating service enforces a dedicated semaphore per adapter
        from this value, so each data source is rate-limited independently and
        a slow source never starves the concurrency budget of another.
        """
        ...

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
