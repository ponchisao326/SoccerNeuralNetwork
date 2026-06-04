"""FBref data-source adapter."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
import soccerdata as sd

from etl.adapters._flatten import flatten_dataframe
from etl.domain.models import ExtractedBatch, ExtractionTarget
from etl.ports.extractor_port import ExtractorPort

logger = logging.getLogger(__name__)

FBREF_STAT_TYPES: tuple[str, ...] = (
    "standard",
    "keeper",
    "shooting",
    "playing_time",
    "misc",
)


class FBrefAdapter(ExtractorPort):
    """Extracts player-season stats and match schedules from FBref.

    All blocking ``soccerdata`` calls are offloaded to a ``ThreadPoolExecutor``
    via ``asyncio.to_thread`` so they never stall the event loop.

    Args:
        data_dir: Local directory used by soccerdata as the HTTP cache.
        executor: Shared thread-pool for all sync scraping workers.
        stat_types: FBref stat categories to pull per (league, season) pair.
    """

    def __init__(
        self,
        data_dir: Path,
        executor: ThreadPoolExecutor,
        stat_types: tuple[str, ...] = FBREF_STAT_TYPES,
    ) -> None:
        self._data_dir = data_dir
        self._executor = executor
        self._stat_types = stat_types

    async def extract(self, target: ExtractionTarget) -> list[ExtractedBatch]:
        """Extract schedule and player-season stats for the given target.

        Args:
            target: League and season to extract.

        Returns:
            One ``ExtractedBatch`` per stat type plus one for the schedule.
            Batches that failed to fetch are returned empty (no exception raised).
        """
        loop = asyncio.get_running_loop()
        futures = []

        futures.append(
            loop.run_in_executor(
                self._executor,
                self._fetch_schedule,
                target.league,
                target.season,
            )
        )

        for stat_type in self._stat_types:
            futures.append(
                loop.run_in_executor(
                    self._executor,
                    self._fetch_player_stats,
                    target.league,
                    target.season,
                    stat_type,
                )
            )

        results = await asyncio.gather(*futures, return_exceptions=True)

        batches: list[ExtractedBatch] = []

        schedule_result = results[0]
        if isinstance(schedule_result, Exception):
            logger.error(
                "FBref schedule FAILED [%s/%s]: %s",
                target.league, target.season, schedule_result,
            )
            batches.append(ExtractedBatch(collection="fbref_schedule"))
        else:
            records = flatten_dataframe(
                schedule_result, "fbref", target.league, target.season
            )
            batches.append(ExtractedBatch(collection="fbref_schedule", records=records))
            logger.info(
                "FBref schedule OK [%s/%s]: %d records",
                target.league, target.season, len(records),
            )

        for idx, stat_type in enumerate(self._stat_types):
            stat_result = results[idx + 1]
            collection = f"fbref_player_{stat_type}"
            if isinstance(stat_result, Exception):
                logger.error(
                    "FBref %s FAILED [%s/%s]: %s",
                    stat_type, target.league, target.season, stat_result,
                )
                batches.append(ExtractedBatch(collection=collection))
            else:
                records = flatten_dataframe(
                    stat_result, "fbref", target.league, target.season
                )
                batches.append(ExtractedBatch(collection=collection, records=records))
                logger.info(
                    "FBref %s OK [%s/%s]: %d records",
                    stat_type, target.league, target.season, len(records),
                )

        return batches

    def _fetch_schedule(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """Blocking worker: pull match schedule from FBref.

        Args:
            league: soccerdata league identifier.
            season: soccerdata season string.

        Returns:
            Raw DataFrame or ``None`` on failure.
        """
        fbref = sd.FBref(leagues=league, seasons=season, data_dir=self._data_dir)
        return fbref.read_schedule()

    def _fetch_player_stats(
        self, league: str, season: str, stat_type: str
    ) -> Optional[pd.DataFrame]:
        """Blocking worker: pull player-season stats from FBref.

        Args:
            league: soccerdata league identifier.
            season: soccerdata season string.
            stat_type: FBref stat category (e.g. ``"standard"``).

        Returns:
            Raw DataFrame or ``None`` on failure.
        """
        fbref = sd.FBref(leagues=league, seasons=season, data_dir=self._data_dir)
        return fbref.read_player_season_stats(stat_type=stat_type)
