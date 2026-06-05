"""Ingest use-case: orchestrates extraction and persistence for all targets."""

import asyncio
import contextlib
import logging
from typing import Optional, Sequence

from etl.domain.models import ExtractionTarget
from etl.memory_guard import MemoryGuard
from etl.ports.extractor_port import ExtractorPort
from etl.ports.repository_port import RepositoryPort

logger = logging.getLogger(__name__)


class IngestService:
    """Orchestrates the full ETL pipeline for a set of targets.

    Receives all dependencies by injection (Dependency Inversion Principle).
    Each extractor gets its own semaphore sized from ``extractor.max_concurrency``,
    so the data sources are rate-limited independently: a slow source (WhoScored,
    bounded by its Chrome process pool) can never occupy the concurrency budget
    of a fast one (FBref, bounded by its HTTP 429 ceiling). A semaphore is held
    only while a scraper performs I/O, then released before the insert so the
    slot frees up as quickly as possible.

    Args:
        extractors: One or more adapters to run per target.
        repository: Persistence adapter for all batches.
        memory_guard: Optional host-memory backpressure guard. When supplied,
            each task waits for memory headroom before its heavy work, and the
            run is aborted (remaining tasks cancelled) if memory crosses the
            guard's critical threshold.
    """

    def __init__(
        self,
        extractors: Sequence[ExtractorPort],
        repository: RepositoryPort,
        memory_guard: Optional[MemoryGuard] = None,
    ) -> None:
        self._extractors = extractors
        self._repository = repository
        self._memory_guard = memory_guard
        self._semaphores: dict[int, asyncio.Semaphore] = {
            id(extractor): asyncio.Semaphore(extractor.max_concurrency)
            for extractor in extractors
        }

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

        watcher = (
            asyncio.create_task(self._abort_on_memory_pressure(tasks))
            if self._memory_guard is not None
            else None
        )

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if watcher is not None:
                watcher.cancel()

        failed = [r for r in results if isinstance(r, Exception)]
        cancelled = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
        for exc in failed:
            if not isinstance(exc, asyncio.CancelledError):
                logger.error("Unhandled task exception: %s: %s", type(exc).__name__, exc)

        if cancelled:
            logger.critical(
                "IngestService: %d task(s) cancelled by the memory guard. "
                "Re-run after the host recovers (lower WHOSCORED_PROCESSES).",
                cancelled,
            )

        logger.info(
            "IngestService: complete. %d/%d tasks succeeded (%d cancelled).",
            len(tasks) - len(failed), len(tasks), cancelled,
        )

    async def _abort_on_memory_pressure(
        self, tasks: Sequence["asyncio.Task[None]"]
    ) -> None:
        """Cancel all unfinished tasks if the memory guard turns critical.

        Backpressure (see :meth:`MemoryGuard.wait_for_headroom`) keeps memory
        below the abort tier in normal operation; this watcher is the last
        resort if in-flight work alone exceeds it. Cancelling stops any further
        heavy work from being scheduled while in-flight work drains.

        Args:
            tasks: All processing tasks to cancel on a critical reading.
        """
        assert self._memory_guard is not None
        await asyncio.gather(self._memory_guard.watch())
        if self._memory_guard.critical.is_set():
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _process(
        self, extractor: ExtractorPort, target: ExtractionTarget
    ) -> None:
        """Acquire the extractor's semaphore, extract, release, then persist.

        The semaphore wraps only the extract call so source concurrency is
        capped without blocking concurrent MongoDB inserts.

        Args:
            extractor: The adapter instance to use for this task.
            target: The (league, season) to extract.
        """
        label = f"{target.league}/{target.season}"

        # Backpressure: wait for host-memory headroom before starting the heavy
        # work, then mark this task as in-flight for the duration so the guard
        # knows memory is actively draining. ``track`` is a no-op stand-in when
        # no guard is configured.
        if self._memory_guard is not None:
            await self._memory_guard.wait_for_headroom(label)
            track = self._memory_guard.track()
        else:
            track = contextlib.nullcontext()

        async with track:
            async with self._semaphores[id(extractor)]:
                batches = await extractor.extract(target)

            # Insert one batch at a time and drop its records immediately after,
            # so a large batch (e.g. a full WhoScored season) is not held in
            # memory while the remaining batches are persisted. Spilled batches
            # are streamed straight from disk so the records never enter memory
            # in this process at all.
            while batches:
                batch = batches.pop()
                if batch.jsonl_path:
                    await self._repository.save_many_from_jsonl(
                        batch.collection, batch.jsonl_path
                    )
                elif batch.records:
                    await self._repository.save_many(batch.collection, batch.records)
                    batch.records = []
                else:
                    logger.debug(
                        "Skipping empty batch '%s' for [%s/%s].",
                        batch.collection, target.league, target.season,
                    )
