"""Worker pool management for the ngram pivot pipeline."""

from __future__ import annotations

import multiprocessing as mp
import shutil
import threading
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle
from ngramprep.common_db.api import open_db, scan_all
from ngramprep.tracking import WorkTracker

from ..config import PipelineConfig
from .worker import worker_process, worker_process_sst, WorkerConfig
from .progress import Counters

__all__ = ["run_worker_pool", "ingest_coordinator_process", "ingest_coordinator_sst_direct"]


def read_and_write_batch_process(
    worker_id: int,
    unit_queue: mp.Queue,
    buffer_size: int,
    output_dir: Path,
    dst_db_path: Path,
    read_profile: str,
    write_profile: str,
    disable_wal: bool,
    work_tracker_path: Path,
    write_lock: mp.Lock,
) -> int:
    """
    Process that loops: reads buffer_size shards, writes them, repeats.

    Strategy:
    1. Pull buffer_size unit IDs from shared queue
    2. Read those shards into memory (PARALLEL across workers)
    3. Acquire write lock (SERIAL)
    4. Write batch
    5. Release lock and repeat

    This keeps workers continuously busy with bounded memory usage.

    Args:
        worker_id: Worker process ID
        unit_queue: Shared queue of unit IDs to process
        buffer_size: Number of shards to buffer before writing
        output_dir: Directory containing shard DBs
        dst_db_path: Path to destination database
        read_profile: RocksDB read profile
        write_profile: RocksDB write profile for writes
        disable_wal: Disable WAL for writes
        work_tracker_path: Path to work tracker database
        write_lock: Multiprocessing lock for serializing writes

    Returns:
        Number of shards processed
    """
    import time as time_module

    try:
        setproctitle(f"ngp:ingest-worker-{worker_id:03d}")
        work_tracker = WorkTracker(work_tracker_path, claim_order="sequential")
        shards_processed = 0

        while True:
            # PHASE 1: Read buffer_size shards into memory
            read_start = time_module.time()
            batch_data = []  # List of (unit_id, shard_data) tuples

            for _ in range(buffer_size):
                try:
                    unit_id = unit_queue.get(timeout=1.0)  # Wait up to 1 second for work
                except Exception:
                    # Queue empty after timeout - no more work available
                    break

                shard_path = output_dir / f"{unit_id}.db"

                if not shard_path.exists():
                    batch_data.append((unit_id, []))
                    continue

                # Read shard into memory
                data = []
                with open_db(shard_path, mode="r", profile=read_profile) as shard_db:
                    for key, value in scan_all(shard_db):
                        data.append((key, value))

                batch_data.append((unit_id, data))

            # If no work retrieved, we're done
            if not batch_data:
                break

            # read_time = time_module.time() - read_start
            # total_items = sum(len(data) for _, data in batch_data)
            # print(f"Worker {worker_id} read {len(batch_data)} shards ({total_items:,} items) in {read_time:.3f}s", flush=True)

            # PHASE 2 & 3: Write batch and update tracker (SERIAL - acquire lock)
            # lock_wait_start = time_module.time()
            with write_lock:
                # lock_wait_time = time_module.time() - lock_wait_start
                # if lock_wait_time > 0.1:
                #     print(f"Worker {worker_id} waited {lock_wait_time:.3f}s for write lock", flush=True)
                # Write to destination DB
                with open_db(dst_db_path, mode="rw", profile=write_profile, create_if_missing=True) as dst_db:
                    wb = dst_db.write_batch(disable_wal=disable_wal, sync=False)
                    wb.__enter__()

                    try:
                        for unit_id, data in batch_data:
                            # Use put_batch to avoid Python per-item overhead
                            # (~1.4x faster than individual put() calls)
                            # put_start = time_module.time()
                            wb.put_batch(data)
                            # put_time = time_module.time() - put_start
                            # print(f"Worker {worker_id} put_batch({len(data):,} items) in {put_time:.3f}s", flush=True)

                            # Mark as ingested immediately after writing this shard
                            # (still inside write lock to avoid SQLite contention)
                            work_tracker.ingest_work_unit(unit_id)
                            shard_path = output_dir / f"{unit_id}.db"
                            if shard_path.exists():
                                try:
                                    shutil.rmtree(shard_path)
                                except Exception:
                                    pass

                        # commit_start = time_module.time()
                        wb.__exit__(None, None, None)
                        # commit_time = time_module.time() - commit_start
                        # print(f"Worker {worker_id} committed batch in {commit_time:.3f}s", flush=True)

                    except Exception as e:
                        try:
                            wb.__exit__(None, None, None)
                        except:
                            pass
                        raise

            shards_processed += len(batch_data)

        return shards_processed

    except Exception as e:
        print(f"Worker {worker_id} error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return shards_processed


def ingest_coordinator_process(
        dst_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        pipeline_config: PipelineConfig,
        stop_event: Optional[mp.Event] = None,
) -> None:
    """
    Ingest all completed shards with N parallel worker processes.

    Strategy:
    - N worker processes (num_ingest_readers)
    - Each worker reads buffer_shards into memory before writing
    - Workers take turns acquiring write lock to dump their batch
    - True parallelism during reads (no GIL), serial writes (no contention)

    Memory Usage:
    - Each shard ~150MB average
    - Total RAM = num_readers * buffer_shards * 150MB
    - Example: 8 workers * 3 shards = 24 shards * 150MB = ~3.6GB RAM

    Benefits:
    - ✅ No queue serialization overhead (no pickle)
    - ✅ True parallel reads across N workers (no Python GIL)
    - ✅ Each worker buffers multiple shards for better I/O throughput
    - ✅ Serial writes eliminate RocksDB lock contention
    - ✅ Large batches minimize commit overhead

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
    num_workers = getattr(pipeline_config, "num_ingest_readers", 8)
    buffer_shards = getattr(pipeline_config, "ingest_buffer_shards", 3)

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

    # Create destination DB parent directory
    dst_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create multiprocessing context and shared queue
    ctx = mp.get_context("spawn")
    unit_queue = ctx.Queue()
    write_lock = ctx.Lock()

    # Populate queue with all unit IDs
    for unit_id in completed_units:
        unit_queue.put(unit_id)

    from tqdm import tqdm

    # Start all workers
    processes = []
    for worker_id in range(num_workers):
        process = ctx.Process(
            target=read_and_write_batch_process,
            args=(
                worker_id,
                unit_queue,
                buffer_shards,
                output_dir,
                dst_db_path,
                ingest_read_profile,
                ingest_write_profile,
                ingest_disable_wal,
                work_tracker_path,
                write_lock,
            ),
            name=f"ngp:ingest-worker-{worker_id}"
        )
        process.start()
        processes.append((worker_id, process))

    # Monitor progress
    completed_units_set = set(completed_units)
    with tqdm(
        total=len(completed_units),
        desc="Shards Ingested:",
        unit="shards",
        ncols=100,
        bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) as pbar:
        # Poll work tracker to update progress
        processed_count = 0
        import time
        while processed_count < len(completed_units):
            # Check how many of OUR units have been marked as ingested
            import sqlite3
            with sqlite3.connect(str(work_tracker_path), timeout=10.0, isolation_level=None) as conn:
                # isolation_level=None enables autocommit and ensures we see latest commits
                placeholders = ','.join('?' * len(completed_units))
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM work_units WHERE status = 'ingested' AND unit_id IN ({placeholders})",
                    completed_units
                )
                current_count = cursor.fetchone()[0]

            if current_count > processed_count:
                pbar.update(current_count - processed_count)
                processed_count = current_count

            # Check if all workers finished
            all_done = all(not p.is_alive() for _, p in processes)
            if all_done:
                break

            time.sleep(0.5)

        # Wait for all workers to complete
        for worker_id, process in processes:
            process.join()
            if process.exitcode not in (0, None):
                print(f"Warning: Worker {worker_id} exited with code {process.exitcode}", flush=True)

    print()  # Newline after progress bar


def ingest_coordinator_sst_direct(
        dst_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        pipeline_config: PipelineConfig,
) -> None:
    """
    Ingest completed SST files via direct file ingestion (fast path).

    This coordinator ingests SST files created by worker_process_sst() directly
    into the destination database using RocksDB's ingest() API. This is MUCH
    faster than write_batch mode since it bypasses memtable writes entirely.

    Strategy:
    - Collect all completed SST file paths
    - Batch SST files for ingestion
    - Use DB.ingest() to bulk-add files
    - Mark shards as ingested and clean up

    Benefits:
    - ✅ No serial write lock contention (ingest is atomic)
    - ✅ No memtable overhead
    - ✅ Direct file operations (move/link)
    - ✅ Orders of magnitude faster for large datasets

    Requirements:
    - Shards must have non-overlapping key ranges
    - SST files must be created with SstFileWriter

    Args:
        dst_db_path: Path to destination database
        work_tracker_path: Path to work tracker
        output_dir: Directory containing completed SST files
        pipeline_config: Pipeline configuration
    """
    from tqdm import tqdm

    # Get configuration
    ingest_write_profile = getattr(pipeline_config, "ingest_write_profile", "write:packed24")

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

    # Create destination DB parent directory
    dst_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure destination DB exists (create if needed)
    with open_db(dst_db_path, mode="rw", profile=ingest_write_profile, create_if_missing=True) as dst_db:
        pass  # Just create it, will reopen below

    with tqdm(
        total=len(completed_units),
        desc="SST Files Ingested:",
        unit="files",
        ncols=100,
        bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) as pbar:
        # Open destination DB once for all ingestions
        with open_db(dst_db_path, mode="rw", profile=ingest_write_profile) as dst_db:
            for unit_id in completed_units:
                sst_path = output_dir / f"{unit_id}.sst"

                if not sst_path.exists():
                    # Empty shard or already cleaned up - just mark as ingested
                    work_tracker.ingest_work_unit(unit_id)
                    pbar.update(1)
                    continue

                try:
                    # Ingest SST file directly into destination DB
                    # move=True: Move file instead of copying (faster, saves space)
                    dst_db.ingest([str(sst_path)], move=True, write_global_seqno=False)

                    # Mark as ingested in work tracker
                    work_tracker.ingest_work_unit(unit_id)

                    pbar.update(1)

                except Exception as e:
                    print(f"\nError ingesting SST file {unit_id}: {e}", flush=True)
                    # Don't mark as ingested if it failed
                    # This allows resume to retry
                    raise

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

    Routes to appropriate worker function based on ingest_mode:
    - "direct_sst": Uses worker_process_sst() to write SST files directly
    - "write_batch": Uses worker_process() to write RocksDB databases

    Args:
        num_workers: Number of worker processes to spawn
        src_db_path: Path to source database
        work_tracker_path: Path to work tracker database
        output_dir: Directory for output databases or SST files
        pipeline_config: Pipeline configuration
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking

    Raises:
        RuntimeError: If any worker processes fail
    """
    ctx = mp.get_context("spawn")
    processes = []

    # Determine which worker function to use based on ingest mode
    ingest_mode = getattr(pipeline_config, "ingest_mode", "write_batch")
    worker_func = worker_process_sst if ingest_mode == "direct_sst" else worker_process

    # Start all worker processes
    for worker_id in range(num_workers):
        process = ctx.Process(
            target=worker_func,
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
