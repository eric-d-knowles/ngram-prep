"""Worker pool management for the ngram pivot pipeline."""

from __future__ import annotations

import multiprocessing as mp
import shutil
import threading
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle
from ngram_prep.common_db.api import open_db, scan_all
from ngram_prep.tracking import WorkTracker

from ..config import PipelineConfig
from .worker import worker_process, WorkerConfig
from .progress import Counters

__all__ = ["run_worker_pool", "ingest_coordinator_process"]


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
        setproctitle(f"ngp:ingest-reader[{unit_id}]")

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


def ingest_reader_process_DEPRECATED(
        reader_id: int,
        output_dir: Path,
        data_queue: mp.Queue,
        ingest_read_profile: str,
        work_tracker_path: Path,
        stop_event: Optional[mp.Event],
) -> None:
    """
    Reader process that polls work tracker for completed units and loads them.

    Uses disk-based "pre-queue": polls work tracker for completed-but-not-ingested units,
    loads shard from output_dir, sends data to writer.

    For separate-stage ingestion (stop_event=None), reader stops when no completed
    units are found after several checks. For concurrent ingestion, reader waits
    for stop_event to be set by workers.

    Args:
        reader_id: Unique ID for this reader process
        output_dir: Directory containing completed shard DBs
        data_queue: Queue to send (unit_id, shard_path) to writer
        ingest_read_profile: RocksDB profile for reading
        work_tracker_path: Path to work tracker
        stop_event: Optional event to signal shutdown (None for separate-stage mode)
    """
    import time
    from ngram_prep.tracking import WorkTracker
    from ngram_prep.common_db import open_db, scan_all

    try:
        setproctitle(f"ngp:ingest-reader-{reader_id}")
        work_tracker = WorkTracker(work_tracker_path, claim_order="sequential")

        consecutive_empty = 0
        max_consecutive_empty = 10  # Stop after 10 consecutive empty checks (1 second)

        while True:
            # Check stop event if provided (concurrent ingestion mode)
            if stop_event is not None and stop_event.is_set():
                break

            # Atomically claim next completed unit (disk-based pre-queue!)
            # This transitions 'completed' → 'ingesting' atomically, preventing duplicate processing
            unit_id = work_tracker.claim_completed_unit_for_ingest()

            if unit_id is None:
                # No work available
                consecutive_empty += 1

                # In separate-stage mode (stop_event=None), stop after consecutive empties
                if stop_event is None and consecutive_empty >= max_consecutive_empty:
                    break

                # Wait briefly and check again
                time.sleep(0.1)
                continue

            # Reset empty counter - we found work
            consecutive_empty = 0

            shard_path = output_dir / f"{unit_id}.db"

            # Send shard path to writer (no pickle overhead!)
            # Writer will read directly - avoids memory pressure and serialization bottleneck
            try:
                # Send (unit_id, shard_path) - writer will queue for deletion after reading
                data_queue.put((unit_id, shard_path))

            except Exception as e:
                print(f"[reader-{reader_id}] Error reading {unit_id}: {e}", flush=True)
                # Send empty to allow pipeline to continue
                data_queue.put((unit_id, None))

        # Send sentinel when done
        data_queue.put(None)

    except Exception as e:
        print(f"[reader-{reader_id}] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def ingest_coordinator_process(
        dst_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        pipeline_config: PipelineConfig,
        stop_event: Optional[mp.Event] = None,
) -> None:
    """
    Ingest all completed shards with parallel reads and writes.

    Spawns multiple reader processes that read shards in parallel, then writes
    results as they arrive (no ordering required - each shard contains disjoint
    key ranges, so write order doesn't affect correctness).

    This is more efficient than the filtering module which enforces deterministic
    order for reproducibility of merge operations.

    Args:
        dst_db_path: Path to destination database
        work_tracker_path: Path to work tracker
        output_dir: Directory containing completed shard DBs
        pipeline_config: Pipeline configuration
        stop_event: Unused (kept for API compatibility)
    """
    # Get configuration
    ingest_read_profile = getattr(pipeline_config, "ingest_read_profile", "read:packed24")
    ingest_write_profile = getattr(pipeline_config, "ingest_write_profile", "write:packed24")
    ingest_disable_wal = getattr(pipeline_config, "ingest_disable_wal", True)
    batch_size = getattr(pipeline_config, "ingest_batch_items", 10_000_000)
    num_readers = getattr(pipeline_config, "num_ingest_readers", 8)
    queue_size = getattr(pipeline_config, "ingest_queue_depth", 50)

    # Get all completed work units in deterministic order
    work_tracker = WorkTracker(work_tracker_path, claim_order="sequential")

    import sqlite3
    with sqlite3.connect(str(work_tracker_path), timeout=10.0) as conn:
        cursor = conn.execute(
            "SELECT unit_id FROM work_units WHERE status = 'completed' ORDER BY unit_id"
        )
        completed_units = [row[0] for row in cursor.fetchall()]

    if not completed_units:
        print("No completed shards to ingest")
        return

    # Create result queue with limited size to control memory usage
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=queue_size)

    # Open destination DB once
    dst_db_path.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
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
                # Launch initial batch of readers (controlled by num_readers)
                # Launch additional readers as data arrives from queue
                active_processes = []

                # Start initial batch
                for i in range(min(num_readers, len(completed_units))):
                    unit_id = completed_units[i]
                    shard_path = output_dir / f"{unit_id}.db"

                    p = ctx.Process(
                        target=read_shard_worker,
                        args=(shard_path, unit_id, ingest_read_profile, result_queue),
                    )
                    p.start()
                    active_processes.append(p)

                next_to_launch_idx = min(num_readers, len(completed_units))

                # Main loop: write results as they arrive (no ordering required)
                shards_written = 0

                while shards_written < len(completed_units):
                    # Get next result from queue (write immediately, no buffering)
                    unit_id, data = result_queue.get()

                    # Write data immediately
                    if data:  # Not empty
                        for key, value in data:
                            wb.put(key, value)
                            batch_item_count += 1

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

                    # Launch next reader immediately after receiving data (not after writing)
                    # This ensures readers keep running while writer is busy
                    if next_to_launch_idx < len(completed_units):
                        next_unit_id = completed_units[next_to_launch_idx]
                        next_shard_path = output_dir / f"{next_unit_id}.db"

                        p = ctx.Process(
                            target=read_shard_worker,
                            args=(next_shard_path, next_unit_id, ingest_read_profile, result_queue),
                        )
                        p.start()
                        active_processes.append(p)
                        next_to_launch_idx += 1

                # Wait for all reader processes to finish
                for p in active_processes:
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

    print()  # Newline after progress bar


def run_worker_pool(
        num_workers: int,
        src_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        pipeline_config: PipelineConfig,
        worker_config: WorkerConfig,
        counters: Optional[Counters] = None,
) -> None:
    """
    Run a pool of worker processes to handle all work units.

    Args:
        num_workers: Number of worker processes to spawn
        src_db_path: Path to source database
        work_tracker_path: Path to work tracker database
        output_dir: Directory for output databases
        pipeline_config: Pipeline configuration
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking

    Raises:
        RuntimeError: If any worker processes fail
    """
    ctx = mp.get_context("spawn")
    processes = []

    # Start all worker processes (no concurrent ingestion)
    for worker_id in range(num_workers):
        process = ctx.Process(
            target=worker_process,
            args=(
                worker_id,
                src_db_path,
                work_tracker_path,
                output_dir,
                pipeline_config,
                worker_config,
                counters,
            ),
            name=f"ngp:worker-{worker_id}"
        )
        process.start()
        processes.append(process)

    # Wait for all workers to complete
    for process in processes:
        process.join()

    # Check for any failures
    failed_processes = [p for p in processes if p.exitcode != 0]
    if failed_processes:
        raise RuntimeError(
            f"{len(failed_processes)} worker processes failed with non-zero exit codes"
        )
