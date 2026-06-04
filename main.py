"""Entry point: wires dependencies and launches the ETL pipeline."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import motor.motor_asyncio

from etl.adapters.fbref_adapter import FBrefAdapter
from etl.adapters.mongo_repository import MongoMatchRepository
from etl.adapters.whoscored_adapter import WhoScoredAdapter
from etl.domain.models import ExtractionTarget
from etl.logging_config import setup_logging
from etl.services.ingest_service import IngestService

LEAGUES: list[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

SEASONS: list[str] = ["2425", "2324", "2223"]

MONGODB_URI: str = os.getenv(
    "MONGODB_URI",
    "mongodb://user:password@0.0.0.0:00000/",
)
DB_NAME: str = os.getenv("MONGODB_DB", "soccer_nn")
CACHE_DIR: Path = Path(os.getenv("SOCCERDATA_CACHE_DIR", "./soccerdata_cache"))

# 2x semaphore ceiling: absorbs the gap between a release and next acquisition
# so pre-warmed threads are ready without pool-allocation latency.
THREAD_POOL_SIZE: int = 6
SEMAPHORE_LIMIT: int = 3

_log_file = setup_logging(logs_dir=Path("logs"))
logger = logging.getLogger(__name__)
logger.info("Log file: %s", _log_file)


async def main() -> None:
    """Build the dependency graph, construct all targets, and run the pipeline."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    client: motor.motor_asyncio.AsyncIOMotorClient = (
        motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
    )
    db: motor.motor_asyncio.AsyncIOMotorDatabase = client[DB_NAME]

    targets: list[ExtractionTarget] = [
        ExtractionTarget(league=league, season=season)
        for league in LEAGUES
        for season in SEASONS
    ]

    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        fbref_adapter = FBrefAdapter(data_dir=CACHE_DIR, executor=executor)
        whoscored_adapter = WhoScoredAdapter(data_dir=CACHE_DIR, executor=executor)
        repository = MongoMatchRepository(db=db)

        service = IngestService(
            extractors=[fbref_adapter, whoscored_adapter],
            repository=repository,
            semaphore_limit=SEMAPHORE_LIMIT,
        )

        logger.info(
            "Starting ETL: %d leagues x %d seasons = %d targets.",
            len(LEAGUES), len(SEASONS), len(targets),
        )
        await service.run(targets)

    client.close()
    logger.info("MongoDB client closed. Pipeline finished.")


if __name__ == "__main__":
    asyncio.run(main())
