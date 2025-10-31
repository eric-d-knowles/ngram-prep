"""Write buffer for batch database operations."""

from __future__ import annotations

__all__ = ["WriteBuffer"]


class WriteBuffer:
    """Buffer for batching database writes."""

    def __init__(self):
        """Initialize write buffer."""
        self.items: list[tuple[bytes, bytes]] = []
        self.current_bytes = 0

    def add(self, key: bytes, value: bytes) -> None:
        """
        Add an item to the buffer.

        Args:
            key: Database key
            value: Database value
        """
        self.items.append((key, value))
        self.current_bytes += len(key) + len(value)

    def flush_and_count(self, db, disable_wal: bool = True) -> int:
        """
        Flush buffer to database and return count of items written.

        Args:
            db: Database handle
            disable_wal: Whether to disable WAL for the batch

        Returns:
            Number of items written
        """
        if not self.items:
            return 0

        count = len(self.items)

        with db.write_batch(disable_wal=disable_wal, sync=False) as batch:
            for key, value in self.items:
                batch.put(key, value)

        self.items.clear()
        self.current_bytes = 0

        return count

    def __len__(self) -> int:
        """Return number of items in buffer."""
        return len(self.items)
