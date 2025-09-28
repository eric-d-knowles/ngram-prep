# ngram_filter/pipeline/ingest.py
"""Ingest filtered RocksDB shards into a final database."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterable
import multiprocessing as mp
from queue import Empty

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
    # print("\nPhase 2: Filtered RocksDB Shards → Final DB")
    # print("=" * 89)


def shard_reader_worker(shard_paths: list[Path], queue: mp.Queue, read_profile: str, batch_size: int = 10000):
    """Worker process that reads shards and batches items before sending to queue."""
    try:
        for shard_dir in shard_paths:
            batch = []
            batch_count = 0
            with open_db(shard_dir, mode="ro", profile=read_profile) as src:
                for key, value in scan_all(src):
                    batch.append((key, value or b""))

                    if len(batch) >= batch_size:
                        # Send batch with batch number for tracking
                        queue.put((shard_dir.name, "BATCH", batch_count, batch))
                        batch = []
                        batch_count += 1

                # Send remaining items
                if batch:
                    queue.put((shard_dir.name, "BATCH", batch_count, batch))
                    batch_count += 1

            # Signal this shard is complete with total batch count
            queue.put((shard_dir.name, "SHARD_COMPLETE", batch_count, None))

    except Exception as e:
        print(f"[reader] Error in worker: {e}", flush=True)
        raise


def delete_shard_safely(shard_path: Path, shard_name: str):
    """Safely delete a shard directory."""
    try:
        shutil.rmtree(shard_path)
        print(f"  Deleted processed shard: {shard_name}")
    except Exception as e:
        print(f"  Warning: Failed to delete {shard_name}: {e}")


def ingest_shards_streaming(
        dst_db_path: Path,
        shards_root: Path,
        *,
        read_profile: str = "read:packed24",
        write_profile: str = "write:packed24",
        batch_bytes: int = 128 * 1024 * 1024,  # 128 MB
        batch_items: int = 200_000,
        disable_wal: bool = False,
        diag_every_batches: int = 50,
        diag_every_seconds: float = 5.0,
        delete_after_ingest: bool = False,
        num_readers: int = 8,  # Number of parallel reader processes
        enable_compact: bool = True,  # Make compaction optional
) -> tuple[int, int]:
    """
    Merge per-shard RocksDBs into a single destination database with parallel reading.

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
        delete_after_ingest: Delete shards after successful ingestion
        num_readers: Number of parallel reader processes
        enable_compact: Whether to perform final compaction (can be slow on large DBs)
    """
    print_phase_banner()

    # Find all shard directories
    shard_dirs = sorted(
        path for path in shards_root.iterdir()
        if path.is_dir() and path.name.startswith("unit_")
    )

    print(f"  Folding {len(shard_dirs)} shard(s) with {num_readers} parallel readers...", flush=True)

    # Split shards across reader processes
    shard_chunks = [shard_dirs[i::num_readers] for i in range(num_readers)]

    # Create queue for (shard_name, batch_of_items) tuples
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=50)  # Much smaller queue since we're sending batches

    # Start reader processes
    readers = []
    for chunk in shard_chunks:
        if chunk:  # Only start process if it has shards to read
            p = ctx.Process(target=shard_reader_worker, args=(chunk, queue, read_profile, 10000))
            p.start()
            readers.append(p)

    # Track which shards are complete and ready for deletion
    completed_shards = set()
    total_shards = len(shard_dirs)

    # Track shard ingestion progress: shard_name -> {'total_batches': N, 'ingested_batches': N}
    shard_progress = {}
    shard_paths_map = {shard.name: shard for shard in shard_dirs}

    # Track stats per shard
    shard_stats = {shard.name: {'items': 0, 'bytes': 0} for shard in shard_dirs}

    # Initialize counters
    total_items = 0
    total_bytes = 0
    total_batches = 0

    # Batch accumulator
    batch = []
    batch_size = 0

    with open_db(dst_db_path, mode="rw", profile=write_profile) as dst:
        last_diag_time = time.perf_counter()

        # Process batches from queue until all shards complete
        while len(completed_shards) < total_shards:
            try:
                # Get message from queue with timeout
                message = queue.get(timeout=1.0)
                shard_name, msg_type = message[0], message[1]

                if msg_type == "SHARD_COMPLETE":
                    # Shard finished reading - track total batches
                    total_batches = message[2]
                    if shard_name not in shard_progress:
                        shard_progress[shard_name] = {'total_batches': 0, 'ingested_batches': 0}
                    shard_progress[shard_name]['total_batches'] = total_batches

                    # Check if this shard is now fully ingested
                    if shard_progress[shard_name]['ingested_batches'] >= total_batches:
                        completed_shards.add(shard_name)
                        stats = shard_stats[shard_name]
                        print(
                            f"  {shard_name}: {stats['items']:,} items "
                            f"({stats['bytes'] / 1_000_000:.1f} MB)",
                            flush=True
                        )

                        # Safe to delete now - fully ingested
                        if delete_after_ingest:
                            delete_shard_safely(shard_paths_map[shard_name], shard_name)
                    continue

                elif msg_type == "BATCH":
                    # Process a batch of data
                    batch_num, batch_data = message[2], message[3]

                    # Initialize tracking if needed
                    if shard_name not in shard_progress:
                        shard_progress[shard_name] = {'total_batches': 0, 'ingested_batches': 0}

                    # Process each item in the batch
                    for key, value in batch_data:
                        # Add to write batch
                        batch.append((key, value, "merge"))
                        batch_size += len(key) + len(value) + 16

                        # Update shard stats
                        shard_stats[shard_name]['items'] += 1
                        shard_stats[shard_name]['bytes'] += len(key) + len(value)

                        # Write batch if full
                        if batch_size >= batch_bytes or len(batch) >= batch_items:
                            try:
                                with dst.write_batch(disable_wal=disable_wal, sync=False) as wb:
                                    for k, v, _ in batch:
                                        wb.merge(k, v)
                            except Exception as e:
                                print(f"[ingest][ERROR] Commit failed: {e}", flush=True)
                                raise

                            # Update counters
                            total_batches += 1
                            total_items += len(batch)
                            total_bytes += batch_size

                            # Reset batch
                            batch = []
                            batch_size = 0

                    # Mark this batch as successfully ingested
                    shard_progress[shard_name]['ingested_batches'] += 1

                    # Check if shard is now complete (all batches ingested)
                    if (shard_progress[shard_name]['total_batches'] > 0 and
                            shard_progress[shard_name]['ingested_batches'] >= shard_progress[shard_name][
                                'total_batches']):

                        if shard_name not in completed_shards:
                            completed_shards.add(shard_name)
                            stats = shard_stats[shard_name]
                            print(
                                f"  {shard_name}: {stats['items']:,} items "
                                f"({stats['bytes'] / 1_000_000:.1f} MB)",
                                flush=True
                            )

                            # Safe to delete now - fully ingested
                            if delete_after_ingest:
                                delete_shard_safely(shard_paths_map[shard_name], shard_name)

            except Empty:
                # Queue timeout - check if we're done
                continue

        # Write final partial batch
        if batch:
            try:
                with dst.write_batch(disable_wal=disable_wal, sync=False) as wb:
                    for k, v, _ in batch:
                        wb.merge(k, v)
                total_items += len(batch)
                total_bytes += batch_size
            except Exception as e:
                print(f"[ingest][ERROR] Final batch commit failed: {e}", flush=True)
                raise

        # Wait for all reader processes to finish
        for p in readers:
            p.join()

        # Finalize the database
        _finalize_database(dst, enable_compact)

    # Note: Shards are deleted during processing if delete_after_ingest=True
    # No need for cleanup here since it's done immediately after reading each shard

    return total_items, total_bytes


def _finalize_database(db, enable_compact: bool = True) -> None:
    """Finalize the database by flushing and optionally compacting."""
    print(f"\nPhase 3: Finalizing...")
    print("═" * 100)

    try:
        print("  Performing final flush...", flush=True)
        db.finalize_bulk()
    except Exception as e:
        print(f"[ingest] finalize_bulk not available or failed: {e}", flush=True)

    if enable_compact:
        try:
            print("  Compacting...", flush=True)
            db.compact_all()
        except Exception:
            pass
    else:
        print("  Skipping compaction (disabled)", flush=True)