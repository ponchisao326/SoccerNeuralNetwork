"""MongoDB persistence adapter backed by motor."""

import logging
from typing import Any

import motor.motor_asyncio
from pymongo.errors import BulkWriteError

from etl.ports.repository_port import RepositoryPort

logger = logging.getLogger(__name__)


class MongoMatchRepository(RepositoryPort):
    """Asynchronous MongoDB repository using the motor driver.

    Args:
        db: Motor async database handle. The caller is responsible for
            managing the client lifecycle (connect / close).
    """

    def __init__(self, db: motor.motor_asyncio.AsyncIOMotorDatabase) -> None:
        self._db = db

    async def save_many(self, collection: str, records: list[dict[str, Any]]) -> int:
        """Bulk-insert records into the named collection.

        Uses ``ordered=False`` so MongoDB continues inserting after a duplicate-key
        error, maximising partial success during incremental re-runs.

        Args:
            collection: Target collection name within the configured database.
            records: Flat Python dicts to insert. Must be BSON-safe.

        Returns:
            Count of successfully inserted documents (>= 0).
        """
        if not records:
            logger.warning("save_many called with empty record list for '%s'.", collection)
            return 0

        try:
            result = await self._db[collection].insert_many(records, ordered=False)
            count = len(result.inserted_ids)
            logger.info("Inserted %d documents into '%s'.", count, collection)
            return count
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
                "Unexpected error saving to '%s': %s: %s",
                collection, type(exc).__name__, exc,
            )
            return 0
