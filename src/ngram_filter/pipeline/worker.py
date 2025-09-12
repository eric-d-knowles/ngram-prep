# ngram_filter/pipeline/worker.py
"""Multi-process worker system for filtering ngram databases."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, replace

from setproctitle import setproctitle

from ..config import FilterConfig
from ..filters.builder import build_processor
from ..filters.core_cy import METADATA_PREFIX
from .work_tracker import WorkTracker, WorkUnit
from .progress import Counters, increment_counter
from common_db.api import open_db, range_scan


@dataclass
class WorkerConfig:
    """Configuration for individual worker processes."""

    buffer_size: int = 10_000  # Items to buffer before flushing
    buffer_bytes: int = 8 * 1024 * 1024  # 8MB buffer limit
    profile: str = "bulk_write:packed24"  # RocksDB profile
    disable_wal: bool = True  # Disable WAL for performance
    disable_compaction: bool = True  # Disable auto-compaction


def worker_process(
        worker_id: int,
        src_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        filter_config: FilterConfig,
        worker_config: WorkerConfig,
        counters: Optional[Counters] = None,
) -> None:
    """
    Main worker process that claims and processes work units.

    Args:
        worker_id: Unique identifier for this worker
        src_db_path: Path to source database
        work_tracker_path: Path to work tracker database
        output_dir: Directory for output databases
        filter_config: Configuration for filtering
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking
    """
    try:
        # Set process title for system monitoring
        setproctitle(f"ngf:worker[{worker_id}]")

        # Initialize work tracker and processor
        work_tracker = WorkTracker(work_tracker_path)
        processor = _initialize_processor(worker_id, filter_config)

        if processor is None:
            return

        processed_units = 0

        # Main work loop
        while True:
            work_unit = _claim_next_work_unit(worker_id, work_tracker)
            if work_unit is None:
                break

            try:
                _process_work_unit(
                    work_unit,
                    src_db_path,
                    output_dir,
                    processor,
                    worker_config,
                    counters
                )

                work_tracker.complete_work_unit(work_unit.unit_id)
                processed_units += 1

            except Exception as e:
                print(f"Worker {worker_id} failed on unit {work_unit.unit_id}: {e}")
                traceback.print_exc()
                work_tracker.fail_work_unit(work_unit.unit_id)

    except Exception as e:
        print(f"Worker {worker_id} crashed during startup: {e}")
        traceback.print_exc()


def _initialize_processor(worker_id: int, filter_config: FilterConfig):
    """Initialize the filter processor for this worker."""
    try:
        # Attach memory-mapped vocabulary if needed
        config = _attach_vocabulary_if_needed(worker_id, filter_config)
        processor = build_processor(config)
        return processor

    except Exception as e:
        print(f"Worker {worker_id} failed to build processor: {e}")
        return None


def _attach_vocabulary_if_needed(worker_id: int, filter_config: FilterConfig) -> FilterConfig:
    """Attach memory-mapped vocabulary to filter config if needed."""
    if not filter_config.vocab_path or filter_config.vocab_view:
        return filter_config

    try:
        from ngram_filter.filters.shared_vocab import MMapVocab
        vocab_view = MMapVocab(str(filter_config.vocab_path))
        return replace(filter_config, vocab_view=vocab_view)

    except Exception as e:
        print(f"Worker {worker_id} vocabulary error: {e}")
        return filter_config


def _claim_next_work_unit(worker_id: int, work_tracker: WorkTracker) -> Optional[WorkUnit]:
    """Attempt to claim the next available work unit."""
    try:
        return work_tracker.claim_work_unit(f"worker-{worker_id}")
    except Exception as e:
        print(f"Worker {worker_id} failed to claim work unit: {e}")
        return None


def _process_work_unit(
        work_unit: WorkUnit,
        src_db_path: Path,
        output_dir: Path,
        processor,
        config: WorkerConfig,
        counters: Optional[Counters] = None,
) -> None:
    """
    Process a single work unit by filtering its key range.

    Args:
        work_unit: The work unit to process
        src_db_path: Path to source database
        output_dir: Directory for output databases
        processor: Filter processor function
        config: Worker configuration
        counters: Optional shared counters for progress tracking
    """
    # Create output database for this work unit
    output_path = output_dir / f"{work_unit.unit_id}.db"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize write buffer
    buffer = WriteBuffer(config.buffer_size, config.buffer_bytes)

    # Process the work unit's key range
    with open_db(src_db_path, mode="ro") as src_db:
        with open_db(
                output_path,
                mode="rw",
                profile=config.profile,
                create_if_missing=True
        ) as dst_db:
            _process_key_range(
                work_unit,
                src_db,
                dst_db,
                processor,
                buffer,
                config,
                counters
            )

            # Final flush and finalization
            buffer.flush(dst_db, config.disable_wal, counters)
            _finalize_output_database(dst_db)


class WriteBuffer:
    """Buffer for batching database writes."""

    def __init__(self, max_items: int, max_bytes: int):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.items: list[tuple[bytes, bytes]] = []
        self.current_bytes = 0

    def add(self, key: bytes, value: bytes) -> bool:
        """
        Add an item to the buffer.

        Returns:
            True if buffer should be flushed after this addition
        """
        self.items.append((key, value))
        self.current_bytes += len(key) + len(value)

        return (len(self.items) >= self.max_items or
                self.current_bytes >= self.max_bytes)

    def flush(
            self,
            dst_db,
            disable_wal: bool,
            counters: Optional[Counters] = None
    ) -> None:
        """Flush all buffered items to the database."""
        if not self.items:
            return

        with dst_db.write_batch(disable_wal=disable_wal, sync=False) as wb:
            for key, value in self.items:
                wb.merge(key, value)

        if counters:
            increment_counter(counters.items_written, len(self.items))
            increment_counter(counters.bytes_flushed, self.current_bytes)

        self._clear()

    def _clear(self) -> None:
        """Clear the buffer."""
        self.items.clear()
        self.current_bytes = 0


def _process_key_range(
        work_unit: WorkUnit,
        src_db,
        dst_db,
        processor,
        buffer: WriteBuffer,
        config: WorkerConfig,
        counters: Optional[Counters],
) -> None:
    """Process all keys in the work unit's range."""
    start_key = work_unit.start_key if work_unit.start_key is not None else b""
    end_key = work_unit.end_key

    for key, value in range_scan(src_db, start_key, end_key):
        if counters:
            increment_counter(counters.items_scanned, 1)

        # Skip metadata keys
        if key.startswith(METADATA_PREFIX):
            if counters:
                increment_counter(counters.items_filtered, 1)
            continue

        # Apply filtering
        processed_key = processor(key)
        if not processed_key:
            if counters:
                increment_counter(counters.items_filtered, 1)
            continue

        # Prepare for buffering
        processed_key_bytes = _ensure_bytes(processed_key)
        value_bytes = _ensure_bytes(value)

        if counters:
            increment_counter(counters.items_enqueued, 1)

        # Add to buffer and flush if needed
        should_flush = buffer.add(processed_key_bytes, value_bytes)
        if should_flush:
            buffer.flush(dst_db, config.disable_wal, counters)


def _ensure_bytes(data) -> bytes:
    """Ensure data is in bytes format."""
    if isinstance(data, bytes):
        return data
    if data is None:
        return b""
    return bytes(data)


def _finalize_output_database(dst_db) -> None:
    """Finalize the output database."""
    try:
        dst_db.finalize_bulk()
    except Exception:
        pass  # Method may not be implemented


def run_worker_pool(
        num_workers: int,
        src_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        filter_config: FilterConfig,
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
        filter_config: Configuration for filtering
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking

    Raises:
        RuntimeError: If any worker processes fail
    """
    ctx = mp.get_context("spawn")
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
                filter_config,
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