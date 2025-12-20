"""Batch writer for accumulating and flushing database writes."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import rocks_shim as rs

from ngramprep.ngram_acquire.db.write import write_batch_to_db
from ngramprep.ngram_acquire.db.metadata import processed_key

logger = logging.getLogger(__name__)

__all__ = ["BatchWriter"]


class BatchWriter:
    """
    Accumulates database writes and flushes when thresholds are exceeded.

    Batches writes to improve RocksDB performance by reducing write amplification.
    Tracks both entry count and total bytes to prevent memory exhaustion.
    """

    def __init__(
        self,
        db: rs.DB,
        max_entries: int = 50_000,
        max_bytes: int = 64 * (1 << 20),  # 64 MiB
    ):
        """
        Initialize batch writer.

        Args:
            db: RocksDB database handle
            max_entries: Maximum entries before auto-flush
            max_bytes: Maximum bytes (keys + values) before auto-flush
        """
        self.db = db
        self.max_entries = max_entries
        self.max_bytes = max_bytes

        self.pending_data: Dict[str, bytes] = {}
        self.pending_files: List[str] = []
        self.pending_bytes = 0

        self.total_entries_written = 0
        self.write_batches = 0

    def add(self, filename: str, data: Dict[str, bytes]) -> bool:
        """
        Add data to the batch.

        Args:
            filename: Name of source file (for metadata tracking)
            data: Parsed data to write (ngram -> packed bytes)

        Returns:
            True if batch should be flushed after this addition
        """
        self.pending_data.update(data)
        self.pending_files.append(filename)
        self.pending_bytes += self._approx_kv_bytes(data)

        return (
            len(self.pending_data) >= self.max_entries
            or self.pending_bytes >= self.max_bytes
        )

    def flush(self) -> None:
        """
        Flush pending batch to database and mark files as processed.

        Raises:
            Exception: If database write fails (caller should handle)
        """
        if not self.pending_data:
            return

        try:
            # Prepare metadata keys for files to mark as processed
            metadata_keys = [processed_key(fname) for fname in self.pending_files]

            # Write data and metadata in a single batch with WAL disabled
            entries_written = write_batch_to_db(
                self.db,
                self.pending_data,
                disable_wal=True,
                metadata_keys=metadata_keys
            )
            self.total_entries_written += entries_written
            self.write_batches += 1

            logger.info(
                "Flushed batch: %d entries, %d files",
                entries_written, len(self.pending_files)
            )
        except Exception:
            logger.error(
                "DB write error for %d entries from %d files; aborting to prevent data loss",
                len(self.pending_data), len(self.pending_files)
            )
            raise
        finally:
            self._clear()

    def _clear(self) -> None:
        """Clear the batch state."""
        self.pending_data.clear()
        self.pending_files.clear()
        self.pending_bytes = 0

    def _approx_kv_bytes(self, d: Dict[str, bytes]) -> int:
        """Estimate total bytes for key-value pairs."""
        return sum(len(k.encode("utf-8")) + len(v) for k, v in d.items())

    def get_stats(self) -> tuple[int, int]:
        """
        Get write statistics.

        Returns:
            Tuple of (total_entries_written, write_batches)
        """
        return self.total_entries_written, self.write_batches
