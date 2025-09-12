# ngram_filter/pipeline/ingest.py
"""Ingest filtered RocksDB shards into a final database."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from common_db.api import open_db, scan_all

# Type aliases for clarity
KeyValue = tuple[bytes, bytes]
KeyValueOp = tuple[bytes, bytes, str]


def batch_for_merge(
        pairs: Iterable[KeyValue],
        target_bytes: int = 64 * 1024 * 1024,  # 64 MB
        max_items: int = 100_000,
) -> Iterable[list[KeyValueOp]]:
    """
    Yield batches of (key, value, 'merge') operations.

    Args:
        pairs: Iterable of (key, value) pairs
        target_bytes: Target size per batch in bytes
        max_items: Maximum items per batch

    Yields:
        Batches of merge operations ready for database writes
    """
    batch: list[KeyValueOp] = []
    current_size = 0

    for key, value in pairs:
        value = value or b""
        batch.append((key, value, "merge"))
        current_size += len(key) + len(value) + 16  # +16 for overhead

        if current_size >= target_bytes or len(batch) >= max_items:
            yield batch
            batch, current_size = [], 0

    if batch:
        yield batch


def get_db_property(db, name: str) -> str:
    """Get a database property, returning '?' if unavailable."""
    try:
        return db.get_property(name) or "?"
    except Exception:
        return "?"


def print_db_stats(db, prefix: str = "[ingest] dst") -> None:
    """Print diagnostic information about the database."""
    estimated_keys = get_db_property(db, "rocksdb.estimate-num-keys")
    memory_usage = get_db_property(db, "rocksdb.cur-size-all-mem-tables")
    flush_pending = get_db_property(db, "rocksdb.mem-table-flush-pending")
    background_errors = get_db_property(db, "rocksdb.background-errors")

    print(
        f"{prefix}: keys={estimated_keys} mem={memory_usage}B "
        f"flush_pending={flush_pending} bg_errs={background_errors}",
        flush=True
    )


def print_phase_banner() -> None:
    """Print the phase 2 banner."""
    print("\nPhase 2: Filtered RocksDB Shards → Final DB")
    print("=" * 89)


def ingest_shards_streaming(
        dst_db_path: Path,
        shards_root: Path,
        *,
        read_profile: str = "read:packed24",
        write_profile: str = "write:packed24",
        batch_bytes: int = 64 * 1024 * 1024,  # 64 MB
        batch_items: int = 100_000,
        disable_wal: bool = False,
        diag_every_batches: int = 50,
        diag_every_seconds: float = 5.0
) -> None:
    """
    Merge per-shard RocksDBs into a single destination database.

    Args:
        dst_db_path: Path to the destination database
        shards_root: Root directory containing shard databases
        read_profile: Profile for reading source databases
        write_profile: Profile for writing to destination database
        batch_bytes: Target bytes per batch
        batch_items: Maximum items per batch
        disable_wal: Whether to disable write-ahead logging
        diag_every_batches: Print diagnostics every N batches
        diag_every_seconds: Print diagnostics every N seconds
    """
    print_phase_banner()

    # Find all shard directories
    shard_dirs = sorted(
        path for path in shards_root.iterdir()
        if path.is_dir() and path.name.startswith("unit_")
    )

    print(f"Folding {len(shard_dirs)} shard(s)...", flush=True)

    # Initialize counters
    total_items = 0
    total_bytes = 0
    total_batches = 0

    with open_db(dst_db_path, mode="rw", profile=write_profile) as dst:
        last_diag_time = time.perf_counter()

        for shard_dir in shard_dirs:
            shard_items = 0
            shard_bytes = 0
            shard_batches = 0

            # Process each shard
            with open_db(shard_dir, mode="ro", profile=read_profile) as src:
                for batch in batch_for_merge(
                        scan_all(src),
                        target_bytes=batch_bytes,
                        max_items=batch_items,
                ):
                    try:
                        # Write batch to destination database
                        with dst.write_batch(disable_wal=disable_wal, sync=False) as wb:
                            for key, value, _ in batch:
                                wb.merge(key, value)
                    except Exception as e:
                        print(f"[ingest][ERROR] Commit failed on {shard_dir.name}: {e}", flush=True)
                        raise

                    # Update counters
                    shard_batches += 1
                    total_batches += 1
                    batch_items_count = len(batch)
                    shard_items += batch_items_count
                    batch_bytes_count = sum(len(k) + len(v) for k, v, _ in batch)
                    shard_bytes += batch_bytes_count

                    # Optional: Print periodic diagnostics
                    # current_time = time.perf_counter()
                    # if (shard_batches % diag_every_batches == 0 or
                    #     current_time - last_diag_time >= diag_every_seconds):
                    #     print_db_stats(dst)
                    #     last_diag_time = current_time

            # Update totals and report shard completion
            total_items += shard_items
            total_bytes += shard_bytes
            print(
                f"{shard_dir.name}: merged {shard_items:,} items "
                f"({shard_bytes / 1_000_000:.1f} MB)",
                flush=True
            )

        # Finalize the database
        _finalize_database(dst)

    # Print final summary
    print(
        f"PROCESSING COMPLETE: Final DB contains {total_items:,} items, "
        f"{total_bytes / 1_000_000:,.1f} MB",
        flush=True,
    )


def _finalize_database(db) -> None:
    """Finalize the database by flushing and compacting."""
    try:
        print("Finalizing (flush)...", flush=True)
        db.finalize_bulk()
    except Exception as e:
        print(f"[ingest] finalize_bulk not available or failed: {e}", flush=True)

    try:
        print("Compacting...", flush=True)
        db.compact_all()
    except Exception:
        pass  # Compaction is optional

    print("=" * 89)