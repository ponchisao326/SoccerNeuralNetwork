"""Serially pre-warm the soccerdata FBref HTTP cache for one or more leagues.

Why this exists
---------------
soccerdata 1.9.0 fetches FBref through an undetected-Chrome (UC) driver to bypass
Cloudflare, exactly like the WhoScored reader. The main ETL pipeline runs FBref in
a thread pool (concurrency 3) concurrently with the WhoScored process pool, so a
brand-new (uncached) league spawns several UC-Chrome instances at once, which
collide and crash ("NoSuchWindowException: target window already closed"). Leagues
already in the cache never hit this because ``soccerdata`` only launches the driver
on a cache miss.

This script fetches every FBref endpoint the pipeline needs (schedule + the five
player-season stat types in :data:`etl.adapters.fbref_adapter.FBREF_STAT_TYPES`)
strictly one request at a time, so a single stable Chrome instance does all the
work. Once the cache is warm, the normal pipeline run reads FBref from disk (no
Chrome) and only WhoScored uses the browser.

Usage
-----
    uv run python -m scripts.prewarm_fbref_cache "ESP-Segunda Division"
    uv run python -m scripts.prewarm_fbref_cache "ESP-Segunda Division" --seasons 2324
    uv run python -m scripts.prewarm_fbref_cache "ESP-Segunda Division" "INT-..."

The league IDs must be registered in soccerdata's league dict (default leagues or
the project's ``soccerdata_config/config/league_dict.json``). ``SOCCERDATA_DIR`` is
pointed at the project config dir so custom leagues resolve, and the cache is
written to ``SOCCERDATA_CACHE_DIR`` (default ``./soccerdata_cache``) -- the same
directory the pipeline reads.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Mirror main.py: load .env and register the project soccerdata config dir BEFORE
# importing soccerdata so custom leagues (e.g. the Segunda) resolve at import time.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
os.environ.setdefault("SOCCERDATA_DIR", str(_PROJECT_ROOT / "soccerdata_config"))

import soccerdata as sd  # noqa: E402

from etl.adapters.fbref_adapter import FBREF_STAT_TYPES  # noqa: E402

logger = logging.getLogger(__name__)

# Same seasons the ETL pipeline targets (kept in sync with main.py).
_DEFAULT_SEASONS: tuple[str, ...] = ("2425", "2324", "2223")


def prewarm(league: str, season: str, cache_dir: Path) -> None:
    """Fetch and cache every FBref endpoint the pipeline needs for one target.

    Each call is issued on its own reader instance and resolved sequentially, so
    only one Chrome session is ever live. Failures are logged and swallowed so a
    single missing stat type does not abort the remaining warm-up work.
    """
    logger.info("Pre-warming FBref cache: %s / %s", league, season)
    try:
        schedule = sd.FBref(leagues=league, seasons=season, data_dir=cache_dir).read_schedule()
        logger.info("  schedule: %d matches", len(schedule))
    except Exception as exc:  # noqa: BLE001 - warm-up must be resilient
        logger.error("  schedule FAILED: %s: %s", type(exc).__name__, exc)

    for stat_type in FBREF_STAT_TYPES:
        try:
            reader = sd.FBref(leagues=league, seasons=season, data_dir=cache_dir)
            stats = reader.read_player_season_stats(stat_type=stat_type)
            logger.info("  %s: %d player rows", stat_type, len(stats))
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s FAILED: %s: %s", stat_type, type(exc).__name__, exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Serially pre-warm the FBref cache.")
    parser.add_argument("leagues", nargs="+", help="soccerdata league IDs to warm.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(_DEFAULT_SEASONS),
        help=f"Seasons to warm (default: {' '.join(_DEFAULT_SEASONS)}).",
    )
    args = parser.parse_args()

    cache_dir = Path(os.getenv("SOCCERDATA_CACHE_DIR", "./soccerdata_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # A stale lock makes seleniumbase hang waiting on a previous (dead) run.
    lock = _PROJECT_ROOT / "downloaded_files" / "driver_fixing.lock"
    if lock.exists():
        lock.unlink()
        logger.info("Removed stale driver lock: %s", lock)

    for league in args.leagues:
        for season in args.seasons:
            prewarm(league, season, cache_dir)

    logger.info("Done. Now run the pipeline; FBref will read from cache (no Chrome).")


if __name__ == "__main__":
    main()
