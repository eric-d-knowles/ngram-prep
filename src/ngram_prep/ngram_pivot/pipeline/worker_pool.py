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

__all__ = ["run_worker_pool", "ingest_writer_process"]


def deletion_worker(deletion_queue: mp.Queue) -> None:
    """Background worker that deletes shard directories asynchronously."""
    setproctitle("ngp:shard-cleanup")
    while True:
        shard_path = deletion_queue.get()
        if shard_path is None:  # Sentinel to stop
            break
        try:
            shutil.rmtree(shard_path)
        except Exception:
            pass  # Ignore deletion errors


def ingest_reader_process(
        reader_id: int,
        output_dir: Path,
        data_queue: mp.Queue,
        ingest_read_profile: str,
        work_tracker_path: Path,
        stop_event: mp.Event,
) -> None:
    """
    Reader process that polls work tracker for completed units and loads them.

    Uses disk-based "pre-queue": polls work tracker for completed-but-not-ingested units,
    loads shard from output_dir, sends data to writer.

    No ingest_queue needed - reads directly from filesystem based on work tracker state!

    Args:
        reader_id: Unique ID for this reader process
        output_dir: Directory containing completed shard DBs
        data_queue: Queue to send (unit_id, shard_data_list) to writer
        ingest_read_profile: RocksDB profile for reading
        work_tracker_path: Path to work tracker
        deletion_queue: Queue for async shard deletion
        stop_event: Event to signal shutdown when all work is done
    """
    import time
    from ngram_prep.tracking import WorkTracker
    from ngram_prep.common_db import open_db, scan_all

    try:
        setproctitle(f"ngp:ingest-reader-{reader_id}")
        work_tracker = WorkTracker(work_tracker_path, claim_order="sequential")

        while not stop_event.is_set():
            # Atomically claim next completed unit (disk-based pre-queue!)
            # This transitions 'completed' → 'ingesting' atomically, preventing duplicate processing
            unit_id = work_tracker.claim_completed_unit_for_ingest()

            if unit_id is None:
                # No work available - wait briefly and check again
                time.sleep(0.1)
                continue

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
        num_filter_workers: int,
        stop_event: Optional[mp.Event] = None,
) -> None:
    """
    Coordinator that spawns readers (poll disk for completed shards) + single writer.

    Uses disk-based pre-queue: readers poll work tracker for completed units,
    eliminating the need for ingest_queue entirely. Workers write to disk,
    readers pick up from disk, writer ingests.

    Args:
        dst_db_path: Path to destination database
        work_tracker_path: Path to work tracker
        output_dir: Directory containing completed shard DBs
        pipeline_config: Pipeline configuration
        num_filter_workers: Number of filter workers (for tracking completion)
    """
    try:
        setproctitle("ngp:ingest-writer")

        ingest_read_profile = getattr(pipeline_config, "ingest_read_profile", "read:packed24")
        ingest_write_profile = getattr(pipeline_config, "ingest_write_profile", "write:packed24")
        ingest_disable_wal = getattr(pipeline_config, "ingest_disable_wal", True)
        batch_size = getattr(pipeline_config, "ingest_batch_items", 10_000_000)

        work_tracker = WorkTracker(
            work_tracker_path,
            claim_order=pipeline_config.work_unit_claim_order
        )

        # Start background deletion worker
        ctx = mp.get_context("spawn")
        deletion_queue = ctx.Queue()
        deletion_process = ctx.Process(target=deletion_worker, args=(deletion_queue,), daemon=True)
        deletion_process.start()

        # Use provided stop_event for coordinated shutdown
        if stop_event is None:
            stop_event = ctx.Event()

        # Start single reader process (polls work tracker, sends paths to writer)
        queue_depth = getattr(pipeline_config, 'ingest_queue_depth', 50)
        data_queue = ctx.Queue(maxsize=queue_depth)
        reader_process = ctx.Process(
            target=ingest_reader_process,
            args=(0, output_dir, data_queue, ingest_read_profile, work_tracker_path, stop_event),
            name=f"ngp:ingest-reader",
            daemon=True
        )
        reader_process.start()

        num_reader_threads = getattr(pipeline_config, 'num_ingest_reader_threads', 4)

        # Initialize work tracker to mark units as ingested after successful write
        work_tracker = WorkTracker(
            work_tracker_path,
            claim_order=pipeline_config.work_unit_claim_order
        )

        # Thread worker function to read shard and return items
        from concurrent.futures import ThreadPoolExecutor
        import queue as queue_module

        def read_shard_to_list(shard_path):
            """Read entire shard and return list of (key, value) tuples."""
            items = []
            try:
                with open_db(shard_path, mode="r", profile=ingest_read_profile) as shard_db:
                    for key, value in scan_all(shard_db):
                        items.append((key, value))
                return items
            except Exception as e:
                print(f"[shard-reader] Error reading {shard_path}: {e}", flush=True)
                return []

        # Single writer with parallel shard readers
        with open_db(dst_db_path, mode="rw", profile=ingest_write_profile, create_if_missing=True) as dst_db:
            # Thread pool for parallel shard reading
            executor = ThreadPoolExecutor(max_workers=num_reader_threads)

            # Track futures for units being read
            pending_units = {}  # unit_id -> (Future, shard_path)
            done_receiving = False

            # Main loop: keep thread pool saturated
            while not done_receiving or pending_units:
                # Step 1: Fill thread pool with work (up to num_reader_threads concurrent)
                while len(pending_units) < num_reader_threads and not done_receiving:
                    try:
                        item = data_queue.get(timeout=0.01)

                        if item is None:
                            done_receiving = True
                            break

                        unit_id, shard_path = item
                        if shard_path is None:
                            continue

                        # Submit to thread pool
                        future = executor.submit(read_shard_to_list, shard_path)
                        pending_units[unit_id] = (future, shard_path)

                    except queue_module.Empty:
                        break  # No more shards available right now

                # Step 2: Check for completed reads
                completed_units = []
                for check_unit_id, (check_future, check_path) in list(pending_units.items()):
                    if check_future.done():
                        completed_units.append((check_unit_id, check_future, check_path))
                        pending_units.pop(check_unit_id)

                # Step 3: Write completed units to database (one commit per shard)
                for comp_unit_id, comp_future, comp_path in completed_units:
                    items = comp_future.result()  # Get items from future

                    # Write all items from this shard in one batch
                    with dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False) as wb:
                        for key, value in items:
                            wb.put(key, value)

                    # Mark unit as ingested and queue for deletion
                    work_tracker.ingest_work_unit(comp_unit_id)
                    deletion_queue.put(comp_path)

                # Step 4: Small sleep to avoid busy-waiting when pending but nothing complete
                if pending_units and not completed_units:
                    import time
                    time.sleep(0.01)

            # Process any remaining pending units (one commit per shard)
            for unit_id, (future, shard_path) in pending_units.items():
                items = future.result()  # Wait for completion

                # Write all items from this shard in one batch
                with dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False) as wb:
                    for key, value in items:
                        wb.put(key, value)

                # Mark unit as ingested and queue for deletion
                work_tracker.ingest_work_unit(unit_id)
                deletion_queue.put(shard_path)

            # Shutdown thread pool
            executor.shutdown(wait=True)

        # Wait for reader to finish
        reader_process.join(timeout=5)
        if reader_process.is_alive():
            reader_process.terminate()

        # Stop deletion worker
        deletion_queue.put(None)
        deletion_process.join(timeout=30)
        if deletion_process.is_alive():
            deletion_process.terminate()

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            deletion_queue.put(None)
            deletion_process.join(timeout=5)
        except:
            pass


def run_worker_pool(
        num_workers: int,
        src_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        dst_db_path: Path,
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
        dst_db_path: Path to destination database
        pipeline_config: Pipeline configuration
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking

    Raises:
        RuntimeError: If any worker processes fail
    """
    ctx = mp.get_context("spawn")

    # Create stop event for coordinated shutdown
    stop_event = ctx.Event()

    # Start ingest coordinator (spawns multiple reader processes + writer)
    # No ingest_queue needed - readers poll work tracker for completed units (disk-based pre-queue!)
    writer_process = ctx.Process(
        target=ingest_coordinator_process,
        args=(
            dst_db_path,
            work_tracker_path,
            output_dir,
            pipeline_config,
            num_workers,
            stop_event,
        ),
        name="ngp:ingest-coord"
    )
    writer_process.start()

    processes = []

    # Start all worker processes
    for worker_id in range(num_workers):
        process = ctx.Process(
            target=worker_process,
            args=(
                worker_id,
                src_db_path,
                work_tracker_path,
                output_dir,
                None,  # No ingest_queue - workers write to output_dir and mark as "completed"
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

    # Signal ingest coordinator that all workers are done
    # Reader will finish processing remaining completed units, then stop
    stop_event.set()

    # Wait for writer to finish processing remaining units
    writer_process.join(timeout=120)
    if writer_process.is_alive():
        # Ingest writer did not stop gracefully, terminating
        writer_process.terminate()
        writer_process.join(timeout=10)

    # Check for any failures
    failed_processes = [p for p in processes if p.exitcode != 0]
    if failed_processes:
        raise RuntimeError(
            f"{len(failed_processes)} worker processes failed with non-zero exit codes"
        )
