"""MongoDB persistence adapter backed by motor."""

import json
import logging
from pathlib import Path
from typing import Any

import motor.motor_asyncio
from pymongo.errors import BulkWriteError

from etl.ports.repository_port import RepositoryPort

logger = logging.getLogger(__name__)

# Insert in bounded chunks instead of one giant call. A WhoScored season yields
# ~580k event dicts; a single insert_many keeps the whole list plus its BSON
# encoding resident at once, which (multiplied by concurrent workers) is what
# exhausted host RAM. Chunking caps the per-call transient and lets each chunk's
# references drop out of scope between awaits.
_INSERT_CHUNK_SIZE = 5_000


class MongoMatchRepository(RepositoryPort):
    """Asynchronous MongoDB repository using the motor driver.

    Args:
        db: Motor async database handle. The caller is responsible for
            managing the client lifecycle (connect / close).
        chunk_size: Maximum documents per ``insert_many`` call. Bounds peak
            memory during large inserts.
    """

    def __init__(
        self,
        db: motor.motor_asyncio.AsyncIOMotorDatabase,
        chunk_size: int = _INSERT_CHUNK_SIZE,
    ) -> None:
        self._db = db
        self._chunk_size = chunk_size

    async def save_many(self, collection: str, records: list[dict[str, Any]]) -> int:
        """Bulk-insert records into the named collection, in bounded chunks.

        Records are split into ``chunk_size`` slices and inserted sequentially.
        Each chunk uses ``ordered=False`` so MongoDB continues inserting after a
        duplicate-key error, maximising partial success during incremental
        re-runs. Chunking keeps peak memory flat regardless of batch size.

        Args:
            collection: Target collection name within the configured database.
            records: Flat Python dicts to insert. Must be BSON-safe.

        Returns:
            Count of successfully inserted documents (>= 0).
        """
        if not records:
            logger.warning("save_many called with empty record list for '%s'.", collection)
            return 0

        total = len(records)
        inserted_total = 0
        for start in range(0, total, self._chunk_size):
            chunk = records[start:start + self._chunk_size]
            inserted_total += await self._insert_chunk(collection, chunk)

        logger.info(
            "Inserted %d/%d documents into '%s' (%d-doc chunks).",
            inserted_total, total, collection, self._chunk_size,
        )
        return inserted_total

    async def save_many_from_jsonl(
        self, collection: str, path: str, delete_after: bool = True
    ) -> int:
        """Stream a line-delimited JSON spill file into the collection.

        Reads at most ``chunk_size`` lines into memory at a time, inserts that
        chunk, then moves on — so a multi-hundred-thousand-row spill file is
        persisted with flat memory use. Tolerates duplicate-key errors per
        chunk like :meth:`save_many`.

        Args:
            collection: Target collection name within the configured database.
            path: Path to a line-delimited JSON file, one document per line.
            delete_after: Remove the spill file once fully consumed.

        Returns:
            Count of successfully inserted documents (>= 0).
        """
        spill = Path(path)
        if not spill.exists():
            logger.warning("Spill file '%s' for '%s' does not exist.", path, collection)
            return 0

        inserted_total = 0
        total = 0
        chunk: list[dict[str, Any]] = []
        try:
            with spill.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    chunk.append(json.loads(line))
                    total += 1
                    if len(chunk) >= self._chunk_size:
                        inserted_total += await self._insert_chunk(collection, chunk)
                        chunk = []
            if chunk:
                inserted_total += await self._insert_chunk(collection, chunk)
        finally:
            if delete_after:
                spill.unlink(missing_ok=True)

        logger.info(
            "Inserted %d/%d documents into '%s' from spill (%d-doc chunks).",
            inserted_total, total, collection, self._chunk_size,
        )
        return inserted_total

    async def _insert_chunk(
        self, collection: str, chunk: list[dict[str, Any]]
    ) -> int:
        """Insert a single chunk, tolerating duplicate-key errors.

        Args:
            collection: Target collection name.
            chunk: Slice of records to insert in one ``insert_many`` call.

        Returns:
            Number of documents successfully inserted from this chunk.
        """
        try:
            result = await self._db[collection].insert_many(chunk, ordered=False)
            return len(result.inserted_ids)
        except BulkWriteError as exc:
            inserted = exc.details.get("nInserted", 0)
            n_errors = len(exc.details.get("writeErrors", []))
            logger.error(
                "BulkWriteError on '%s': %d inserted, %d errors (duplicate keys tolerated).",
                collection, inserted, n_errors,
            )
            return inserted
        except Exception as exc:
            logger.error(
                "Unexpected error saving chunk to '%s': %s: %s",
                collection, type(exc).__name__, exc,
            )
            return 0
