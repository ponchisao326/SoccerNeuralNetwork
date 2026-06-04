"""Ingest use-case: orchestrates extraction and persistence for all targets."""

import asyncio
import logging
from typing import Sequence

from etl.domain.models import ExtractionTarget
from etl.ports.extractor_port import ExtractorPort
from etl.ports.repository_port import RepositoryPort

logger = logging.getLogger(__name__)


class IngestService:
    """Orchestrates the full ETL pipeline for a set of targets.

    Receives all dependencies by injection (Dependency Inversion Principle).
    The semaphore is the sole rate-limiting mechanism; it is held only while
    a scraper is performing HTTP I/O, then released before the insert so the
    slot is freed for the next scraper as quickly as possible.

    Args:
        extractors: One or more adapters to run per target.
        repository: Persistence adapter for all batches.
        semaphore_limit: Maximum number of concurrent HTTP scrapers.
            FBref returns HTTP 429 under heavier load; keep this at 3 or below.
    """

    def __init__(
        self,
        extractors: Sequence[ExtractorPort],
        repository: RepositoryPort,
        semaphore_limit: int = 3,
    ) -> None:
        self._extractors = extractors
        self._repository = repository
        self._semaphore = asyncio.Semaphore(semaphore_limit)

    async def run(self, targets: Sequence[ExtractionTarget]) -> None:
        """Schedule and execute all (extractor, target) combinations.

        Args:
            targets: All (league, season) pairs to process.
        """
        tasks = [
            asyncio.create_task(self._process(extractor, target))
            for extractor in self._extractors
            for target in targets
        ]

        logger.info(
            "IngestService: dispatching %d tasks (%d extractors x %d targets).",
            len(tasks), len(self._extractors), len(targets),
        )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed = [r for r in results if isinstance(r, Exception)]
        if failed:
            for exc in failed:
                logger.error("Unhandled task exception: %s: %s", type(exc).__name__, exc)

        logger.info(
            "IngestService: complete. %d/%d tasks succeeded.",
            len(tasks) - len(failed), len(tasks),
        )

    async def _process(
        self, extractor: ExtractorPort, target: ExtractionTarget
    ) -> None:
        """Acquire the semaphore, extract, release, then persist.

        The semaphore wraps only the extract call so HTTP concurrency is capped
        without blocking concurrent MongoDB inserts.

        Args:
            extractor: The adapter instance to use for this task.
            target: The (league, season) to extract.
        """
        async with self._semaphore:
            batches = await extractor.extract(target)

        for batch in batches:
            if batch.records:
                await self._repository.save_many(batch.collection, batch.records)
            else:
                logger.debug(
                    "Skipping empty batch '%s' for [%s/%s].",
                    batch.collection, target.league, target.season,
                )
