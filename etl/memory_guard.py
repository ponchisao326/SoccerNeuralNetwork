"""System-memory backpressure guard for the ingest pipeline.

The WhoScored extractors are memory-heavy: each ``(league, season)`` worker
builds ~580k event dicts and ships a pickled copy back to the parent, so a few
concurrent targets can push a 32 GB host into swap and freeze it. This guard
adds two safety tiers based on system-wide memory use:

1. **Backpressure (pause tier).** Before a task starts its heavy work it waits
   until memory drops below ``pause_bytes``. Already-running tasks keep
   draining (their inserts free memory), so the gate reopens on its own. This
   is the primary protection — it stops the pipeline from ever over-committing.
2. **Abort (critical tier).** A background watcher trips an event when memory
   crosses ``abort_bytes``; the orchestrator then cancels all not-yet-finished
   tasks so no new heavy work is scheduled, letting in-flight work drain.

Memory is measured as ``total - available`` (the portion that is genuinely in
use and not reclaimable), which is the meaningful figure for OOM proximity on
Linux.
"""

import asyncio
import contextlib
import logging
from typing import AsyncIterator

import psutil

logger = logging.getLogger(__name__)

_GIB = 2 ** 30


class MemoryGuard:
    """Throttles and, if necessary, aborts work based on host memory pressure.

    Args:
        pause_bytes: Hold new heavy work while ``used >= pause_bytes``.
        abort_bytes: Trip the critical event while ``used >= abort_bytes``.
        poll_interval: Seconds between memory re-checks.
    """

    def __init__(
        self,
        pause_bytes: int,
        abort_bytes: int,
        poll_interval: float = 2.0,
    ) -> None:
        self._pause = pause_bytes
        self._abort = abort_bytes
        self._poll = poll_interval
        self._inflight = 0
        self._critical = asyncio.Event()

    @staticmethod
    def used_bytes() -> int:
        """Return system memory in use (``total - available``) in bytes."""
        vm = psutil.virtual_memory()
        return vm.total - vm.available

    @property
    def critical(self) -> asyncio.Event:
        """Event set once memory has crossed the abort threshold."""
        return self._critical

    async def wait_for_headroom(self, label: str) -> None:
        """Block until memory use is below the pause threshold.

        To guarantee forward progress, the gate never blocks when nothing is
        in flight: if no other task is running, waiting cannot free memory, so
        the caller is allowed through regardless of the reading.

        Args:
            label: Identifier for the waiting task, used in log messages.
        """
        while True:
            used = self.used_bytes()
            if used < self._pause or self._inflight == 0:
                if used >= self._pause:
                    logger.warning(
                        "Memory at %.1f GiB (>= pause %.1f GiB) but nothing "
                        "in flight; proceeding with [%s] to ensure progress.",
                        used / _GIB, self._pause / _GIB, label,
                    )
                return
            logger.warning(
                "Memory at %.1f GiB (>= pause %.1f GiB); holding new work "
                "[%s] (in-flight=%d).",
                used / _GIB, self._pause / _GIB, label, self._inflight,
            )
            await asyncio.sleep(self._poll)

    @contextlib.asynccontextmanager
    async def track(self) -> AsyncIterator[None]:
        """Count the wrapped block as in-flight heavy work.

        The in-flight count lets :meth:`wait_for_headroom` distinguish "memory
        is high but draining" from "memory is high and nothing can free it".
        """
        self._inflight += 1
        try:
            yield
        finally:
            self._inflight -= 1

    async def watch(self) -> None:
        """Poll memory and set the critical event if the abort tier is crossed.

        Runs until cancelled. Setting the event is the signal for the
        orchestrator to cancel remaining tasks.
        """
        while not self._critical.is_set():
            used = self.used_bytes()
            if used >= self._abort:
                logger.critical(
                    "Memory at %.1f GiB crossed abort threshold %.1f GiB. "
                    "Signalling shutdown to protect the host.",
                    used / _GIB, self._abort / _GIB,
                )
                self._critical.set()
                return
            await asyncio.sleep(self._poll)
