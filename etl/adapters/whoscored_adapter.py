"""WhoScored data-source adapter.

Requires Chrome/Chromium and ``undetected-chromedriver`` on the host.

Concurrency model
-----------------
soccerdata's WhoScored driver keeps per-instance Selenium state and is not
thread-safe, so several Chrome sessions cannot coexist inside one process (the
root cause of the historical ``SessionNotCreatedException`` crashes). This
adapter therefore parallelises *across* ``(league, season)`` targets with a
``ProcessPoolExecutor``: every extraction runs in its own OS process with its
own headless Chrome, giving full isolation instead of the previous
single-threaded ``threading.Semaphore`` serialisation. The pool's
``max_workers`` — mirrored by :attr:`WhoScoredAdapter.max_concurrency` — caps
how many browsers run at once.

HTML-wrapper auto-repair
------------------------
Chrome wraps JSON API responses in ``<html><head></head><body>{…}</body></html>``
when saving to disk. soccerdata caches that page source verbatim, then tries to
``json.loads()`` it on the next read — failing at the first ``<``. The worker
detects the ``JSONDecodeError``, strips the wrapper from every matching cache
file, and retries up to ``_MAX_STRIP_RETRIES`` times (once per HTML-wrapped
fixture page).

Disk spill (memory safety)
--------------------------
A WhoScored season is ~580k events. Returning that as a pickled list made the
parent hold several full-season lists at once (one per concurrent worker, plus
pickle transients), which exhausted host RAM. Instead, each worker now
*streams* its flattened events to a JSON spill file under ``<cache>/_spill/``
and returns only the path; the repository reads that file back in bounded
chunks. Peak memory is therefore one DataFrame per worker plus one insert chunk
in the parent, flat regardless of season size.
"""

import asyncio
import json as _json
import logging
import re as _re
import uuid as _uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import soccerdata as sd

from etl.adapters._flatten import spill_dataframe_to_jsonl
from etl.domain.models import ExtractedBatch, ExtractionTarget
from etl.ports.extractor_port import ExtractorPort

logger = logging.getLogger(__name__)

_HTML_BODY_RE = _re.compile(r"<body>(.*?)(?:</body>|$)", _re.DOTALL)
_MAX_STRIP_RETRIES = 12
_SPILL_SUBDIR = "_spill"


def _strip_html_from_cache(data_dir: Path, league: str, season: str) -> int:
    """Strip ``<html>…<body>`` wrappers from Selenium-cached JSON files.

    Only rewrites files whose ``<body>`` content starts with ``{`` or ``[``
    (i.e. is actually JSON). Genuine HTML pages are left untouched.

    Args:
        data_dir: soccerdata cache root for this run.
        league: Used to filter cache file names by prefix.
        season: Used to filter cache file names by prefix.

    Returns:
        Number of files that were stripped and rewritten.
    """
    matches_dir = data_dir / "matches"
    if not matches_dir.exists():
        return 0

    prefix = f"{league}_{season}"
    stripped = 0
    for f in matches_dir.iterdir():
        if not f.is_file() or not f.name.startswith(prefix):
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            if not content.lstrip().startswith("<"):
                continue
            match = _HTML_BODY_RE.search(content)
            if not match:
                continue
            body = match.group(1).strip()
            if not body or body[0] not in ("{", "["):
                continue
            f.write_text(body, encoding="utf-8")
            stripped += 1
        except OSError as exc:
            logger.warning("Could not process cache file %s: %s", f, exc)
    return stripped


def _fetch_events_worker(
    league: str, season: str, data_dir: str, headless: bool
) -> tuple[Optional[str], int]:
    """Process-pool worker: scrape WhoScored events for one ``(league, season)``.

    Runs in a dedicated child process with its own headless Chrome session, so
    multiple targets are scraped in parallel without sharing soccerdata's
    non-thread-safe Selenium state. The flattened events (a full season is
    ~580k records) are **streamed to a JSON spill file** instead of being
    returned: only the file path crosses the process boundary, so neither this
    worker nor the parent ever holds the whole list in memory or pays a large
    pickle. ``read_events(on_error="skip")`` keeps a single broken match from
    aborting the whole season.

    Args:
        league: soccerdata league identifier.
        season: soccerdata season string.
        data_dir: soccerdata cache root (passed as ``str`` for pickling).
        headless: Run Chrome without a visible window.

    Returns:
        ``(spill_path, record_count)``. ``spill_path`` is ``None`` (and the
        count ``0``) when there is nothing to persist or on unrecoverable
        failure.
    """
    cache_dir = Path(data_dir)
    spill_path = cache_dir / _SPILL_SUBDIR / f"whoscored_{league}_{season}_{_uuid.uuid4().hex}.jsonl"
    for attempt in range(_MAX_STRIP_RETRIES + 1):
        try:
            ws = sd.WhoScored(
                leagues=league,
                seasons=season,
                data_dir=cache_dir,
                headless=headless,
            )
            df = ws.read_events(on_error="skip")
            if df is None or df.empty:
                return None, 0
            count = spill_dataframe_to_jsonl(
                df, str(spill_path), "whoscored", league, season
            )
            # Drop the DataFrame as soon as it is spilled so its memory is
            # reclaimed in this child process immediately.
            del df
            return str(spill_path), count
        except _json.JSONDecodeError:
            if attempt == _MAX_STRIP_RETRIES:
                logger.error(
                    "WhoScored max retries (%d) reached [%s/%s]. Giving up.",
                    _MAX_STRIP_RETRIES, league, season,
                )
                return None, 0
            stripped = _strip_html_from_cache(cache_dir, league, season)
            if stripped == 0:
                logger.error(
                    "WhoScored JSONDecodeError but no HTML-wrapped files found "
                    "[%s/%s]. Giving up.",
                    league, season,
                )
                return None, 0
            logger.warning(
                "WhoScored attempt %d: stripped %d HTML-wrapped file(s), "
                "retrying [%s/%s].",
                attempt + 1, stripped, league, season,
            )
        except Exception as exc:
            logger.error(
                "WhoScored worker [%s/%s]: %s: %s",
                league, season, type(exc).__name__, exc,
            )
            return None, 0
    return None, 0


class WhoScoredAdapter(ExtractorPort):
    """Extracts match-event data from WhoScored via isolated Chrome processes.

    Args:
        data_dir: Local directory used by soccerdata as the HTTP cache.
        executor: Shared ``ProcessPoolExecutor`` whose ``max_workers`` bounds
            the number of concurrent Chrome sessions.
        max_concurrency: Concurrency hint surfaced to the orchestrating service;
            should equal the pool's ``max_workers``.
        headless: Run Chrome without a visible window (faster and more stable).
    """

    def __init__(
        self,
        data_dir: Path,
        executor: ProcessPoolExecutor,
        max_concurrency: int,
        headless: bool = True,
    ) -> None:
        self._data_dir = data_dir
        self._executor = executor
        self._max_concurrency = max_concurrency
        self._headless = headless

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    async def extract(self, target: ExtractionTarget) -> list[ExtractedBatch]:
        """Extract match events for the given target.

        Args:
            target: League and season to extract.

        Returns:
            A single-element list containing a ``whoscored_events`` batch. The
            batch points at a JSON spill file on disk; it is empty (no path,
            zero count) if the request fails after all retries or yields no
            events.
        """
        loop = asyncio.get_running_loop()
        spill_path, count = await loop.run_in_executor(
            self._executor,
            _fetch_events_worker,
            target.league,
            target.season,
            str(self._data_dir),
            self._headless,
        )
        logger.info(
            "WhoScored events OK [%s/%s]: %d records spilled to disk",
            target.league, target.season, count,
        )
        return [
            ExtractedBatch(
                collection="whoscored_events",
                jsonl_path=spill_path,
                record_count=count,
            )
        ]
