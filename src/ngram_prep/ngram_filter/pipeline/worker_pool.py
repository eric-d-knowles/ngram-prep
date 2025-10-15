"""Worker pool management for the ngram filter pipeline."""

from __future__ import annotations

import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle
from ngram_prep.common_db.api import open_db, scan_all
from ngram_prep.parallel import WorkTracker

from ..config import FilterConfig, PipelineConfig
from .worker import worker_process, WorkerConfig
from .progress import Counters

__all__ = ["run_worker_pool", "ingest_writer_process"]


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
        batch_size = getattr(pipeline_config, "ingest_batch_items", 200_000)

        sentinels_received = 0

        with open_db(dst_db_path, mode="rw", profile=ingest_write_profile, create_if_missing=True) as dst_db:
            while sentinels_received < num_filter_workers:
                try:
                    item = ingest_queue.get(timeout=1.0)

                    # Check for sentinel (None means a worker has finished)
                    if item is None:
                        sentinels_received += 1
                        continue

                    unit_id, shard_path = item

                    # Ingest the shard
                    batch = []
                    with open_db(shard_path, mode="r", profile=ingest_read_profile) as shard_db:
                        for key, value in scan_all(shard_db):
                            value = value or b""
                            batch.append((key, value))

                            if len(batch) >= batch_size:
                                with dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False) as wb:
                                    for k, v in batch:
                                        wb.merge(k, v)
                                batch = []

                        # Write final batch
                        if batch:
                            with dst_db.write_batch(disable_wal=ingest_disable_wal, sync=False) as wb:
                                for k, v in batch:
                                    wb.merge(k, v)

                    # Mark as ingested and delete shard
                    work_tracker.ingest_work_unit(unit_id)
                    try:
                        shutil.rmtree(shard_path)
                    except Exception:
                        pass

                except Exception as e:
                    import queue
                    if isinstance(e, queue.Empty):
                        continue  # Normal timeout, keep waiting
                    print(f"[ingest-writer] Error: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"[ingest-writer] Writer crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()


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
    ingest_queue = ctx.Queue(maxsize=num_workers * 2)

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
        print("Warning: Ingest writer did not stop, terminating...", flush=True)
        writer_process.terminate()
        writer_process.join(timeout=10)

    # Check for any failures
    failed_processes = [p for p in processes if p.exitcode != 0]
    if failed_processes:
        raise RuntimeError(
            f"{len(failed_processes)} worker processes failed with non-zero exit codes"
        )
