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
