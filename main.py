"""Entry point: wires dependencies and launches the ETL pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

# The .env load and SOCCERDATA_DIR registration MUST happen before any soccerdata
# import (the adapters below import it transitively). soccerdata reads
# SOCCERDATA_DIR at import time to locate its config directory, and that is where
# the custom ``league_dict.json`` registering non-default leagues (e.g. the
# Spanish Segunda Division / LaLiga Hypermotion) lives. ``setdefault`` lets an
# explicit .env value win, falling back to the version-controlled project config.
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")
os.environ.setdefault("SOCCERDATA_DIR", str(_PROJECT_ROOT / "soccerdata_config"))

import asyncio  # noqa: E402
import logging  # noqa: E402
import multiprocessing  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

import motor.motor_asyncio  # noqa: E402

from etl.adapters.fbref_adapter import FBrefAdapter  # noqa: E402
from etl.adapters.mongo_repository import MongoMatchRepository  # noqa: E402
from etl.adapters.whoscored_adapter import WhoScoredAdapter  # noqa: E402
from etl.domain.models import ExtractionTarget  # noqa: E402
from etl.logging_config import setup_logging  # noqa: E402
from etl.memory_guard import MemoryGuard  # noqa: E402
from etl.services.ingest_service import IngestService  # noqa: E402

_LEAGUES_ALL: list[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    # Spanish second division, sponsor name "LaLiga Hypermotion". Not part of
    # soccerdata's default league dict; registered via soccerdata_config/config/
    # league_dict.json (FBref name "Spanish Segunda División"). FBref coverage is
    # verified; the WhoScored tournament name is best-effort, so its event-derived
    # regression targets may be masked if it does not resolve.
    "ESP-Segunda Division",
]

_SEASONS_ALL: list[str] = ["2526", "2425", "2324", "2223"]


def _select(env_var: str, available: list[str]) -> list[str]:
    """Return an optional comma-separated subset of ``available`` from the env.

    The pipeline has no upsert, so re-running a league already in the database
    duplicates its documents. This override lets a single new league (or season)
    be ingested in isolation -- leaving existing collections untouched -- without
    editing this file, e.g. ``ETL_LEAGUES="ESP-Segunda Division" uv run python
    main.py``. Unknown entries fail fast so a typo never silently scrapes nothing.
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return available
    chosen = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in chosen if item not in available]
    if unknown:
        raise SystemExit(f"{env_var}: unknown entries {unknown}. Valid options: {available}")
    return chosen


LEAGUES: list[str] = _select("ETL_LEAGUES", _LEAGUES_ALL)
SEASONS: list[str] = _select("ETL_SEASONS", _SEASONS_ALL)

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
