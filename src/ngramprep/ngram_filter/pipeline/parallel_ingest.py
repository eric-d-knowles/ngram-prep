"""Parallel shard reading with sequential deterministic writing."""

from __future__ import annotations

import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from ngramprep.common_db.api import open_db, scan_all
from ngramprep.tracking import WorkTracker
from ..config import PipelineConfig

__all__ = ["ingest_shards_parallel"]


def read_shard_worker(
    shard_path: Path,
    unit_id: str,
    read_profile: str,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process that reads a shard into memory.

    Args:
        shard_path: Path to shard database
        unit_id: Work unit ID
        read_profile: RocksDB read profile
        result_queue: Queue to put (unit_id, data) tuples
    """
    try:
        if not shard_path.exists():
            # Empty shard - send empty data
            result_queue.put((unit_id, []))
            return

        # Read entire shard into memory
        data = []
        with open_db(shard_path, mode="r", profile=read_profile) as shard_db:
            for key, value in scan_all(shard_db):
                data.append((key, value))

        result_queue.put((unit_id, data))

    except Exception as e:
        # On error, send empty data
        print(f"Error reading shard {unit_id}: {e}")
        result_queue.put((unit_id, []))


def ingest_shards_parallel(
    output_dir: Path,
    dst_db_path: Path,
    work_tracker_path: Path,
    pipeline_config: PipelineConfig,
    num_readers: int = 4,
    queue_size: int = 8,
) -> tuple[int, int]:
    """
    Ingest all completed shards with parallel reads and writes.

    Reads shards in parallel and writes them as they arrive (no ordering required).
    Each shard contains independent data, so write order doesn't affect correctness.

    Args:
        output_dir: Directory containing shard DBs
        dst_db_path: Path to destination database
        work_tracker_path: Path to work tracker database
        pipeline_config: Pipeline configuration
        num_readers: Number of parallel reader processes
        queue_size: Maximum number of shards buffered in queue (controls memory usage)

    Returns:
        Tuple of (total_items_merged, total_bytes_merged)
    """
    # Get configuration
    ingest_read_profile = getattr(pipeline_config, "ingest_read_profile", "read:packed24")
    ingest_write_profile = getattr(pipeline_config, "ingest_write_profile", "write:packed24")
    ingest_disable_wal = getattr(pipeline_config, "ingest_disable_wal", True)
    batch_size = getattr(pipeline_config, "ingest_batch_items", 2_000_000)

    # Get all completed work units
    work_tracker = WorkTracker(work_tracker_path, claim_order="sequential")

    import sqlite3
    with sqlite3.connect(str(work_tracker_path), timeout=10.0) as conn:
        cursor = conn.execute(
            "SELECT unit_id FROM work_units WHERE status = 'completed'"
        )
        completed_units = [row[0] for row in cursor.fetchall()]

    if not completed_units:
        print("No completed shards to ingest")
        return 0, 0

    # Statistics
    total_items = 0
    total_bytes = 0
    start_time = time.time()

    # Create result queue with limited size to control memory usage
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=queue_size)

    # Open destination DB once
    dst_db_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(
        total=len(completed_units),
        desc="Shards Ingested:",
        unit="shards",
        ncols=100,
        bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) as pbar:
        with open_db(dst_db_path, mode="rw", profile=ingest_write_profile, create_if_missing=True) as dst_db:
            # Open write batch
            wb = dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False)
            wb.__enter__()
            batch_item_count = 0

            try:
                # Process shards as they complete (no ordering required)
                active_processes = []

                # Start initial batch of readers
                for i in range(min(num_readers, len(completed_units))):
                    unit_id = completed_units[i]
                    shard_path = output_dir / f"{unit_id}.db"

                    p = ctx.Process(
                        target=read_shard_worker,
                        args=(shard_path, unit_id, ingest_read_profile, result_queue),
                    )
                    p.start()
                    active_processes.append((i, p))

                next_to_launch_idx = min(num_readers, len(completed_units))

                # Main loop: write results as they arrive
                shards_written = 0
                while shards_written < len(completed_units):
                    # Get next result from queue (write immediately, no buffering)
                    unit_id, data = result_queue.get()

                    # Write data immediately
                    if data:  # Not empty
                        for key, value in data:
                            wb.merge(key, value)
                            batch_item_count += 1
                            total_items += 1
                            total_bytes += len(key) + len(value)

                            # Commit batch periodically
                            if batch_item_count >= batch_size:
                                wb.__exit__(None, None, None)
                                wb = dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False)
                                wb.__enter__()
                                batch_item_count = 0

                    # Mark as ingested and delete shard
                    work_tracker.ingest_work_unit(unit_id)
                    shard_path = output_dir / f"{unit_id}.db"
                    if shard_path.exists():
                        try:
                            shutil.rmtree(shard_path)
                        except Exception:
                            pass

                    pbar.update(1)
                    shards_written += 1

                    # Launch next reader if available
                    if next_to_launch_idx < len(completed_units):
                        unit_id = completed_units[next_to_launch_idx]
                        shard_path = output_dir / f"{unit_id}.db"

                        p = ctx.Process(
                            target=read_shard_worker,
                            args=(shard_path, unit_id, ingest_read_profile, result_queue),
                        )
                        p.start()
                        active_processes.append((next_to_launch_idx, p))
                        next_to_launch_idx += 1

                # Wait for all reader processes to finish
                for idx, p in active_processes:
                    p.join(timeout=30)
                    if p.is_alive():
                        p.terminate()
                        p.join()

                # Commit final batch
                wb.__exit__(None, None, None)

            except Exception as e:
                # Clean up on error
                try:
                    wb.__exit__(None, None, None)
                except:
                    pass

                # Terminate all active processes
                for _, p in active_processes:
                    if p.is_alive():
                        p.terminate()
                        p.join()
                raise

    elapsed = time.time() - start_time
    print(f"\nIngestion complete: {len(completed_units)} shards, {total_items:,} items "
          f"in {elapsed:.1f}s ({total_items/elapsed:,.0f} items/s)")

    return total_items, total_bytes
