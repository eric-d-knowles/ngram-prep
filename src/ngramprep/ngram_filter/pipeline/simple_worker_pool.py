"""Simplified worker pool that just produces shards (no concurrent ingestion)."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Optional

from ..config import FilterConfig, PipelineConfig
from .worker import worker_process, WorkerConfig
from .progress import Counters

__all__ = ["run_worker_pool_simple"]


def run_worker_pool_simple(
        num_workers: int,
        src_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        filter_config: FilterConfig,
        pipeline_config: PipelineConfig,
        worker_config: WorkerConfig,
        counters: Optional[Counters] = None,
) -> None:
    """
    Run a pool of worker processes to filter and write shards.

    Workers produce shard DBs in output_dir. No concurrent ingestion happens here.
    Ingestion is handled separately after all workers finish.

    Args:
        num_workers: Number of worker processes to spawn
        src_db_path: Path to source database
        work_tracker_path: Path to work tracker database
        output_dir: Directory for output shard databases
        filter_config: Configuration for filtering
        pipeline_config: Pipeline configuration
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking

    Raises:
        RuntimeError: If any worker processes fail
    """
    ctx = mp.get_context("spawn")
    processes = []

    # Start all worker processes
    # Workers will write shards to output_dir and mark them as completed
    for worker_id in range(num_workers):
        process = ctx.Process(
            target=worker_process,
            args=(
                worker_id,
                src_db_path,
                work_tracker_path,
                output_dir,
                None,  # No ingest_queue - workers just write shards
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

    # Check for any failures
    failed_processes = [p for p in processes if p.exitcode != 0]
    if failed_processes:
        raise RuntimeError(
            f"{len(failed_processes)} worker processes failed with non-zero exit codes"
        )
