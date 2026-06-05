"""Abstract port for all persistence adapters."""

from abc import ABC, abstractmethod
from typing import Any


class RepositoryPort(ABC):
    """Contract every storage adapter must satisfy.

    Implementations must use an async driver (e.g. ``motor``) and must tolerate
    duplicate-key errors gracefully to support idempotent pipeline re-runs.
    """

    @abstractmethod
    async def save_many(self, collection: str, records: list[dict[str, Any]]) -> int:
        """Persist a list of flat records to the named collection.

        Args:
            collection: Target collection name within the configured database.
            records: Flat Python dicts to insert. Each dict must be BSON-safe.

        Returns:
            Number of successfully inserted documents (>= 0).
        """
        ...

    @abstractmethod
    async def save_many_from_jsonl(
        self, collection: str, path: str, delete_after: bool = True
    ) -> int:
        """Persist records streamed from a line-delimited JSON spill file.

        Implementations must read the file incrementally and insert in bounded
        chunks so that neither the full file nor a large in-memory list is held
        at once. This is the persistence path for batches too large to keep in
        memory (see ``ExtractedBatch.jsonl_path``).

        Args:
            collection: Target collection name within the configured database.
            path: Path to a line-delimited JSON file, one document per line.
            delete_after: Remove the spill file once consumed.

        Returns:
            Number of successfully inserted documents (>= 0).
        """
        ...
