"""Write buffer for batching database writes."""

from __future__ import annotations

from typing import Optional

from .progress import Counters, increment_counter

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
            key: Key bytes
            value: Value bytes
        """
        self.items.append((key, value))
        self.current_bytes += len(key) + len(value)

    def flush(
            self,
            dst_db,
            disable_wal: bool,
            counters: Optional[Counters] = None
    ) -> None:
        """
        Flush all buffered items to the database.

        Args:
            dst_db: Destination database
            disable_wal: Whether to disable WAL for this write
            counters: Optional shared counters for progress tracking
        """
        if not self.items:
            return

        with dst_db.write_batch(disable_wal=disable_wal, sync=False) as wb:
            for key, value in self.items:
                wb.merge(key, value)

        if counters:
            increment_counter(counters.items_written, len(self.items))
            increment_counter(counters.bytes_flushed, self.current_bytes)

        self._clear()

    def flush_and_count(self, dst_db, disable_wal: bool) -> int:
        """
        Flush all buffered items to the database and return count.

        Args:
            dst_db: Destination database
            disable_wal: Whether to disable WAL for this write

        Returns:
            Number of items written
        """
        if not self.items:
            return 0

        count = len(self.items)

        with dst_db.write_batch(disable_wal=disable_wal, sync=False) as wb:
            for key, value in self.items:
                wb.merge(key, value)

        self._clear()
        return count

    def _clear(self) -> None:
        """Clear the buffer."""
        self.items.clear()
        self.current_bytes = 0
