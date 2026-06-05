"""Entry point: wires dependencies and launches the ETL pipeline."""

import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import motor.motor_asyncio
from dotenv import load_dotenv

from etl.adapters.fbref_adapter import FBrefAdapter
from etl.adapters.mongo_repository import MongoMatchRepository
from etl.adapters.whoscored_adapter import WhoScoredAdapter
from etl.domain.models import ExtractionTarget
from etl.logging_config import setup_logging
from etl.memory_guard import MemoryGuard
from etl.services.ingest_service import IngestService

# Load .env from the project root explicitly so it is found regardless of the
# current working directory. Keep the file at the repository root (not in etl/).
load_dotenv(Path(__file__).resolve().parent / ".env")

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

# FBref: stateless HTTP. 2x semaphore ceiling absorbs the gap between a release
# and the next acquisition so pre-warmed threads avoid pool-allocation latency.
FBREF_THREAD_POOL_SIZE: int = 6
FBREF_MAX_CONCURRENCY: int = 3

# WhoScored: each target runs in its own process with an isolated headless
# Chrome. Each worker also holds one full season of events in memory (~580k
# dicts) plus the pickled copy sent back to the parent, so raising this scales
# RAM steeply. The default of 2 is the RAM-safe ceiling on a 32 GB host; bump it
# via the env var only if memory headroom allows. Tunable without code changes.
WHOSCORED_PROCESSES: int = int(os.getenv("WHOSCORED_PROCESSES", "2"))
WHOSCORED_HEADLESS: bool = os.getenv("WHOSCORED_HEADLESS", "1") != "0"

# Host-memory safety valve (see etl/memory_guard.py). New heavy extractions are
# held while system memory use exceeds MEM_PAUSE_GB (backpressure); the run is
# aborted if it ever crosses MEM_ABORT_GB. Defaults target a ~32 GB host: pause
# with ~6 GB headroom, abort before swap thrashing freezes the machine. Set
# MEM_PAUSE_GB=0 to disable the guard entirely.
_GIB = 2 ** 30
MEM_PAUSE_GB: float = float(os.getenv("MEM_PAUSE_GB", "25"))
MEM_ABORT_GB: float = float(os.getenv("MEM_ABORT_GB", "28"))

# NOTE: logging is configured inside the ``__main__`` guard, not at module
# level. The WhoScored ProcessPoolExecutor uses the "spawn" start method, which
# re-imports this module in every child process. Module-level setup_logging()
# would therefore open a redundant (empty) log file per child and race on the
# logs/ directory. Keeping it under the guard runs it once, in the parent only.
logger = logging.getLogger(__name__)


async def main() -> None:
    """Build the dependency graph, construct all targets, and run the pipeline."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Clear orphan WhoScored spill files left by a previously interrupted run.
    # The repository deletes each file after consuming it, so anything here is
    # stale and safe to remove before a fresh run.
    spill_dir = CACHE_DIR / "_spill"
    if spill_dir.exists():
        for stale in spill_dir.glob("*.jsonl"):
            stale.unlink(missing_ok=True)

    client: motor.motor_asyncio.AsyncIOMotorClient = (
        motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
    )
    db: motor.motor_asyncio.AsyncIOMotorDatabase = client[DB_NAME]

    targets: list[ExtractionTarget] = [
        ExtractionTarget(league=league, season=season)
        for league in LEAGUES
        for season in SEASONS
    ]

    # "spawn" gives each Chrome worker a clean interpreter, avoiding the locked
    # state and Selenium instability that "fork" inherits from this process.
    spawn_ctx = multiprocessing.get_context("spawn")

    with ThreadPoolExecutor(max_workers=FBREF_THREAD_POOL_SIZE) as thread_executor, \
            ProcessPoolExecutor(
                max_workers=WHOSCORED_PROCESSES, mp_context=spawn_ctx
            ) as process_executor:
        fbref_adapter = FBrefAdapter(
            data_dir=CACHE_DIR,
            executor=thread_executor,
            max_concurrency=FBREF_MAX_CONCURRENCY,
        )
        whoscored_adapter = WhoScoredAdapter(
            data_dir=CACHE_DIR,
            executor=process_executor,
            max_concurrency=WHOSCORED_PROCESSES,
            headless=WHOSCORED_HEADLESS,
        )
        repository = MongoMatchRepository(db=db)

        memory_guard = (
            MemoryGuard(
                pause_bytes=int(MEM_PAUSE_GB * _GIB),
                abort_bytes=int(MEM_ABORT_GB * _GIB),
            )
            if MEM_PAUSE_GB > 0
            else None
        )

        service = IngestService(
            extractors=[fbref_adapter, whoscored_adapter],
            repository=repository,
            memory_guard=memory_guard,
        )

        logger.info(
            "Starting ETL: %d leagues x %d seasons = %d targets "
            "(WhoScored processes=%d, headless=%s, mem pause=%.0fG abort=%.0fG).",
            len(LEAGUES), len(SEASONS), len(targets),
            WHOSCORED_PROCESSES, WHOSCORED_HEADLESS,
            MEM_PAUSE_GB, MEM_ABORT_GB,
        )
        await service.run(targets)

    client.close()
    logger.info("MongoDB client closed. Pipeline finished.")


if __name__ == "__main__":
    _log_file = setup_logging(logs_dir=Path("logs"))
    logger.info("Log file: %s", _log_file)
    asyncio.run(main())
