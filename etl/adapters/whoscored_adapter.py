"""WhoScored data-source adapter.

Requires Chrome/Chromium and ``undetected-chromedriver`` on the host.

Concurrency note
----------------
soccerdata's WhoScored driver is not thread-safe and cannot sustain multiple
concurrent Chrome sessions. ``_CHROME_LOCK`` (a ``threading.Semaphore(1)``)
serialises all ``_fetch_events`` calls across the entire process so only one
Chrome window is ever open at a time, regardless of the asyncio Semaphore used
by ``IngestService``.

HTML-wrapper auto-repair
------------------------
Chrome wraps JSON API responses in ``<html><head></head><body>{…}</body></html>``
when saving to disk. soccerdata caches that page source verbatim, then tries to
``json.loads()`` it on the next read — failing at the first ``<``. The adapter
detects the ``JSONDecodeError``, strips the wrapper from every matching cache
file, and retries. Because a season has up to 10 fixture pages (each a separate
cache file), the strip+retry loop runs up to ``_MAX_STRIP_RETRIES`` times, once
per newly-created HTML-wrapped file.
"""

import asyncio
import json as _json
import logging
import re as _re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
import soccerdata as sd

from etl.adapters._flatten import flatten_dataframe
from etl.domain.models import ExtractedBatch, ExtractionTarget
from etl.ports.extractor_port import ExtractorPort

logger = logging.getLogger(__name__)

_CHROME_LOCK = threading.Semaphore(1)
_HTML_BODY_RE = _re.compile(r"<body>(.*?)(?:</body>|$)", _re.DOTALL)
_MAX_STRIP_RETRIES = 12


class WhoScoredAdapter(ExtractorPort):
    """Extracts match-event data from WhoScored.

    Args:
        data_dir: Local directory used by soccerdata as the HTTP cache.
        executor: Shared thread-pool for all sync scraping workers.
    """

    def __init__(self, data_dir: Path, executor: ThreadPoolExecutor) -> None:
        self._data_dir = data_dir
        self._executor = executor

    async def extract(self, target: ExtractionTarget) -> list[ExtractedBatch]:
        """Extract match events for the given target.

        Args:
            target: League and season to extract.

        Returns:
            A single-element list containing a ``whoscored_events`` batch.
            The batch is empty if the request fails after all retries.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._fetch_events,
            target.league,
            target.season,
        )

        if result is None:
            return [ExtractedBatch(collection="whoscored_events")]

        records = flatten_dataframe(result, "whoscored", target.league, target.season)
        logger.info(
            "WhoScored events OK [%s/%s]: %d records",
            target.league, target.season, len(records),
        )
        return [ExtractedBatch(collection="whoscored_events", records=records)]

    def _fetch_events(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """Blocking worker: pull match events from WhoScored.

        Acquires ``_CHROME_LOCK`` so only one Chrome session runs at a time.
        On each ``JSONDecodeError`` the HTML-wrapped cache files are stripped
        and the call retries, up to ``_MAX_STRIP_RETRIES`` times.

        Args:
            league: soccerdata league identifier.
            season: soccerdata season string.

        Returns:
            Raw DataFrame, or ``None`` on unrecoverable failure.
        """
        with _CHROME_LOCK:
            for attempt in range(_MAX_STRIP_RETRIES + 1):
                try:
                    ws = sd.WhoScored(
                        leagues=league, seasons=season, data_dir=self._data_dir
                    )
                    return ws.read_events()
                except _json.JSONDecodeError:
                    if attempt == _MAX_STRIP_RETRIES:
                        logger.error(
                            "WhoScored max retries (%d) reached [%s/%s]. Giving up.",
                            _MAX_STRIP_RETRIES, league, season,
                        )
                        return None
                    stripped = self._strip_html_from_cache(league, season)
                    if stripped == 0:
                        logger.error(
                            "WhoScored JSONDecodeError but no HTML-wrapped files found "
                            "[%s/%s]. Giving up.",
                            league, season,
                        )
                        return None
                    logger.warning(
                        "WhoScored attempt %d: stripped %d HTML-wrapped file(s), "
                        "retrying [%s/%s].",
                        attempt + 1, stripped, league, season,
                    )
                except Exception as exc:
                    logger.error(
                        "WhoScored _fetch_events [%s/%s]: %s: %s",
                        league, season, type(exc).__name__, exc,
                    )
                    return None
        return None  # unreachable, satisfies type checker

    def _strip_html_from_cache(self, league: str, season: str) -> int:
        """Strip ``<html><head></head><body>`` wrappers from Selenium-cached files.

        Only rewrites files whose ``<body>`` content starts with ``{`` or ``[``
        (i.e. is actually JSON). Genuine HTML pages are left untouched.

        Args:
            league: Used to filter cache file names by prefix.
            season: Used to filter cache file names by prefix.

        Returns:
            Number of files that were stripped and rewritten.
        """
        matches_dir = self._data_dir / "matches"
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
