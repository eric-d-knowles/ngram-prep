"""Worker pool management for the ngram filter pipeline."""

from __future__ import annotations

import multiprocessing as mp
import shutil
import threading
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle
from ngram_prep.common_db.api import open_db, scan_all
from ngram_prep.ngram_filter.tracking import WorkTracker

from ..config import FilterConfig, PipelineConfig
from .worker import worker_process, WorkerConfig
from .progress import Counters

__all__ = ["run_worker_pool", "ingest_writer_process"]


def deletion_worker(deletion_queue: mp.Queue) -> None:
    """Background worker that deletes shard directories asynchronously."""
    setproctitle("ngf:shard-cleanup")
    while True:
        shard_path = deletion_queue.get()
        if shard_path is None:  # Sentinel to stop
            break
        try:
            shutil.rmtree(shard_path)
        except Exception:
            pass  # Ignore deletion errors


def prefetch_worker(
        ingest_queue: mp.Queue,
        prefetch_queue: mp.Queue,
        ingest_read_profile: str,
        num_filter_workers: int,
) -> None:
    """
    Background thread that prefetches shard data ahead of the merger.

    This overlaps disk I/O (opening and reading shards) with merge operations,
    improving throughput by keeping the merger continuously fed with data.

    Args:
        ingest_queue: Queue receiving completed shards from workers
        prefetch_queue: Queue to send prefetched data to merger
        ingest_read_profile: RocksDB profile for reading shards
        num_filter_workers: Number of workers (for counting sentinels)
    """
    sentinels_received = 0

    while sentinels_received < num_filter_workers:
        item = ingest_queue.get()

        # Check for sentinel
        if item is None:
            sentinels_received += 1
            # Pass sentinel to merger
            prefetch_queue.put(None)
            continue

        unit_id, shard_path = item

        # Prefetch all data from this shard
        # Worker has already closed the shard, so safe to open read-only
        try:
            with open_db(shard_path, mode="r", profile=ingest_read_profile) as shard_db:
                # Read all data into memory
                prefetched_data = list(scan_all(shard_db))

            # Send prefetched data to merger
            prefetch_queue.put((unit_id, shard_path, prefetched_data))
            print(f"[PREFETCH] Successfully prefetched unit {unit_id} ({len(prefetched_data)} items)")
        except Exception as e:
            # On error, pass through without data (merger will handle gracefully)
            import traceback
            print(f"[PREFETCH] FAILED to prefetch unit {unit_id} from {shard_path}: {e}")
            traceback.print_exc()
            prefetch_queue.put((unit_id, shard_path, []))


def ingest_writer_process(
        dst_db_path: Path,
        work_tracker_path: Path,
        pipeline_config: PipelineConfig,
        ingest_queue: mp.Queue,
        num_filter_workers: int,
) -> None:
    """
    Single writer process that receives completed shards and ingests them.

    This process is the ONLY process that opens the destination database for writing,
    eliminating lock contention.

    Uses a prefetch thread to overlap shard reading with merge operations.

    Args:
        dst_db_path: Path to destination database
        work_tracker_path: Path to work tracker
        pipeline_config: Pipeline configuration
        ingest_queue: Queue to receive (unit_id, shard_path) tuples
        num_filter_workers: Number of filter workers (for counting sentinels)
    """
    try:
        setproctitle("ngf:ingest-writer")

        work_tracker = WorkTracker(work_tracker_path)
        ingest_read_profile = getattr(pipeline_config, "ingest_read_profile", "read:packed24")
        ingest_write_profile = getattr(pipeline_config, "ingest_write_profile", "write:packed24")
        ingest_disable_wal = getattr(pipeline_config, "ingest_disable_wal", True)
        # Increased from 200k to 2M to 5M for better write batching and throughput
        batch_size = getattr(pipeline_config, "ingest_batch_items", 5_000_000)
        # Enable/disable prefetching
        enable_prefetch = getattr(pipeline_config, "enable_prefetch", True)
        prefetch_queue_depth = getattr(pipeline_config, "prefetch_queue_depth", 3)

        sentinels_received = 0
        # Batch tracker updates to reduce SQLite overhead
        tracker_batch = []
        tracker_batch_size = 20

        # Performance monitoring
        import time as time_module
        shards_processed = 0
        total_items = 0
        last_report = time_module.time()

        # Prefetch tracking
        prefetch_hits = 0
        prefetch_misses = 0

        # Start background deletion worker
        ctx = mp.get_context("spawn")
        deletion_queue = ctx.Queue()
        deletion_process = ctx.Process(target=deletion_worker, args=(deletion_queue,), daemon=True)
        deletion_process.start()

        # Start prefetch thread if enabled
        prefetch_thread = None
        if enable_prefetch:
            import queue as queue_module
            # Use a bounded queue to limit memory usage
            prefetch_queue = queue_module.Queue(maxsize=prefetch_queue_depth)
            prefetch_thread = threading.Thread(
                target=prefetch_worker,
                args=(ingest_queue, prefetch_queue, ingest_read_profile, num_filter_workers),
                daemon=True,
                name="prefetch-thread"
            )
            prefetch_thread.start()
            print(f"[PREFETCH] Prefetch thread STARTED (queue depth={prefetch_queue_depth})")
        else:
            prefetch_queue = None
            print(f"[PREFETCH] Prefetch DISABLED")

        # Batch branch checking interval - check threshold every N items instead of every item
        CHECK_INTERVAL = 100_000
        MERGE_BATCH_SIZE = 5000  # Batch size for merge_batch API calls

        with open_db(dst_db_path, mode="rw", profile=ingest_write_profile, create_if_missing=True) as dst_db:
            # Open write batch once and keep it open across multiple shards
            # This allows batching writes from many small shards together
            wb = dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False)
            wb.__enter__()
            item_count = 0

            while sentinels_received < num_filter_workers:
                try:
                    # Get data from prefetch queue if enabled, otherwise from ingest queue
                    if enable_prefetch:
                        item = prefetch_queue.get(timeout=1.0)
                    else:
                        item = ingest_queue.get(timeout=1.0)

                    # Check for sentinel (None means a worker has finished)
                    if item is None:
                        sentinels_received += 1
                        continue

                    if enable_prefetch:
                        # Prefetched data format: (unit_id, shard_path, prefetched_data)
                        unit_id, shard_path, prefetched_data = item
                    else:
                        # Direct format: (unit_id, shard_path)
                        unit_id, shard_path = item
                        prefetched_data = None

                    # Stream shard data using batch merge API for reduced Python→C++ overhead
                    start_count = item_count
                    merge_batch = []

                    if enable_prefetch and prefetched_data is not None and len(prefetched_data) > 0:
                        print(f"[PREFETCH] Using prefetched data for unit {unit_id} ({len(prefetched_data)} items)")
                        prefetch_hits += 1
                        # Use prefetched data (already in memory)
                        for key, value in prefetched_data:
                            merge_batch.append((key, value))
                            item_count += 1

                            # Flush merge batch when it reaches threshold
                            if len(merge_batch) >= MERGE_BATCH_SIZE:
                                wb.merge_batch(merge_batch)
                                merge_batch = []

                            # Periodically check if write batch should be committed
                            if (item_count - start_count) % CHECK_INTERVAL == 0:
                                if item_count >= batch_size:
                                    # Flush any remaining merge batch before committing write batch
                                    if merge_batch:
                                        wb.merge_batch(merge_batch)
                                        merge_batch = []

                                    wb.__exit__(None, None, None)
                                    wb = dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False)
                                    wb.__enter__()
                                    item_count = 0
                                    start_count = 0
                    else:
                        # Fallback: read directly from shard (no prefetch or prefetch failed)
                        print(f"[PREFETCH] FALLBACK to direct read for unit {unit_id} (prefetch_enabled={enable_prefetch}, has_data={prefetched_data is not None and len(prefetched_data) > 0 if prefetched_data is not None else 'N/A'})")
                        prefetch_misses += 1
                        with open_db(shard_path, mode="r", profile=ingest_read_profile) as shard_db:
                            for key, value in scan_all(shard_db):
                                merge_batch.append((key, value))
                                item_count += 1

                                # Flush merge batch when it reaches threshold
                                if len(merge_batch) >= MERGE_BATCH_SIZE:
                                    wb.merge_batch(merge_batch)
                                    merge_batch = []

                                # Periodically check if write batch should be committed
                                if (item_count - start_count) % CHECK_INTERVAL == 0:
                                    if item_count >= batch_size:
                                        # Flush any remaining merge batch before committing write batch
                                        if merge_batch:
                                            wb.merge_batch(merge_batch)
                                            merge_batch = []

                                        wb.__exit__(None, None, None)
                                        wb = dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False)
                                        wb.__enter__()
                                        item_count = 0
                                        start_count = 0

                    # Flush any remaining items in merge batch
                    if merge_batch:
                        wb.merge_batch(merge_batch)

                    # Calculate shard items after loop completes
                    shard_items = item_count - start_count

                    # Performance monitoring
                    shards_processed += 1
                    total_items += shard_items
                    current_time = time_module.time()
                    if current_time - last_report >= 10.0:
                        elapsed = current_time - last_report
                        rate = total_items / elapsed
                        total_shards_so_far = prefetch_hits + prefetch_misses
                        hit_rate = (prefetch_hits / total_shards_so_far * 100) if total_shards_so_far > 0 else 0
                        print(f"[PREFETCH] Stats: {shards_processed} shards, {total_items:,} items in {elapsed:.1f}s ({rate:,.0f} items/s) | Prefetch: {prefetch_hits} hits, {prefetch_misses} misses ({hit_rate:.1f}% hit rate)")
                        shards_processed = 0
                        total_items = 0
                        last_report = current_time

                    # Add to tracker batch instead of updating immediately
                    tracker_batch.append(unit_id)

                    # Queue shard for async deletion (non-blocking)
                    deletion_queue.put(shard_path)

                    # Flush tracker batch when it reaches batch_size
                    if len(tracker_batch) >= tracker_batch_size:
                        for uid in tracker_batch:
                            work_tracker.ingest_work_unit(uid)
                        tracker_batch = []

                except Exception as e:
                    import queue as queue_module
                    if isinstance(e, queue_module.Empty):
                        continue  # Normal timeout, keep waiting
                    # Error occurred during ingestion
                    import traceback
                    traceback.print_exc()

            # Close the final write batch
            if wb is not None:
                wb.__exit__(None, None, None)

            # Flush any remaining tracker updates
            if tracker_batch:
                for uid in tracker_batch:
                    work_tracker.ingest_work_unit(uid)
                tracker_batch = []

        # Print final prefetch statistics
        total_shards = prefetch_hits + prefetch_misses
        if total_shards > 0:
            hit_rate = (prefetch_hits / total_shards * 100)
            print(f"\n[PREFETCH] FINAL STATS: {prefetch_hits} hits / {prefetch_misses} misses / {total_shards} total ({hit_rate:.1f}% hit rate)")
        else:
            print(f"\n[PREFETCH] FINAL STATS: No shards processed")

        # Wait for prefetch thread to finish
        if prefetch_thread is not None:
            prefetch_thread.join(timeout=5)

        # Signal deletion worker to stop and wait for it to finish
        deletion_queue.put(None)  # Sentinel
        deletion_process.join(timeout=30)
        if deletion_process.is_alive():
            deletion_process.terminate()

    except Exception as e:
        # Writer process crashed
        import traceback
        traceback.print_exc()
        # Try to stop deletion worker
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
        filter_config: FilterConfig,
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
        filter_config: Configuration for filtering
        pipeline_config: Pipeline configuration
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking

    Raises:
        RuntimeError: If any worker processes fail
    """
    ctx = mp.get_context("spawn")

    # Create ingest queue for coordinated writes
    # Increased from 2x to 10x to 20x to prevent workers from blocking when ingest writer is busy
    ingest_queue = ctx.Queue(maxsize=num_workers * 20)

    # Start single ingest writer process
    writer_process = ctx.Process(
        target=ingest_writer_process,
        args=(
            dst_db_path,
            work_tracker_path,
            pipeline_config,
            ingest_queue,
            num_workers,
        ),
        name="ngf:ingest-writer"
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
                ingest_queue,
                filter_config,
                pipeline_config,
                worker_config,
                counters,
            ),
            name=f"ngf:worker-{worker_id}"
        )
        process.start()
        processes.append(process)

    # Wait for all workers to complete
    for process in processes:
        process.join()

    # Send sentinels to writer so it knows all workers are done
    for _ in range(num_workers):
        ingest_queue.put(None)

    # Wait for writer to finish
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
