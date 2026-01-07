"""Worker process for filtering ngram databases."""

from __future__ import annotations

import multiprocessing as mp
import random
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle

from ..config import FilterConfig, PipelineConfig
from ..filters.processor_factory import build_processor
from ..filters.core_cy import METADATA_PREFIX
from ..year_binning import bin_year_records
from .progress import Counters, increment_counter
from .write_buffer import WriteBuffer
from ngramprep.common_db.api import open_db, range_scan
from ngramprep.tracking import WorkTracker, WorkUnit, SimpleOutputManager

__all__ = ["WorkerConfig", "worker_process"]


@dataclass
class WorkerConfig:
    """Configuration for individual worker processes."""

    disable_wal: bool = True  # Disable WAL for performance
    disable_compaction: bool = True  # Disable auto-compaction


def worker_process(
        worker_id: int,
        src_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        ingest_queue: Optional[mp.Queue],
        filter_config: FilterConfig,
        pipeline_config: PipelineConfig,
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
        ingest_queue: Queue to send completed shards to ingest writer
        filter_config: Configuration for filtering
        pipeline_config: Configuration for DB operations
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking
    """
    try:
        # Set process title for system monitoring
        setproctitle(f"ngf:worker[{worker_id:03d}]")

        # Add startup jitter to avoid thundering herd on initial claims
        # Spread workers out over ~2 seconds at startup
        import random
        import time
        startup_delay = random.uniform(0, min(2.0, worker_id * 0.15))
        time.sleep(startup_delay)

        # Initialize work tracker and processor
        work_tracker = WorkTracker(
            work_tracker_path,
            claim_order=pipeline_config.work_unit_claim_order
        )
        processor = _initialize_processor(worker_id, filter_config)

        if processor is None:
            return

        # Initialize output manager
        output_manager = SimpleOutputManager(output_dir, extension=".db")

        processed_units = 0
        consecutive_no_work = 0  # Track consecutive failed claims for backoff

        # Main work loop
        while True:
            work_unit = _claim_next_work_unit(worker_id, work_tracker, output_manager)

            if work_unit is None:
                # No work available - use exponential backoff to reduce thundering herd
                consecutive_no_work += 1

                # Reduced backoff: 0.1s, 0.2s, 0.4s (max)
                # Exit quickly when no work remains
                base_delay = min(0.1 * (2 ** (consecutive_no_work - 1)), 0.4)
                jitter = random.uniform(0, base_delay * 0.3)
                delay = base_delay + jitter

                # Exit after 3 failed attempts - work tracker now detects completion faster
                if consecutive_no_work >= 3:
                    # No work for 3 attempts - likely all work is done
                    break

                time.sleep(delay)
                continue

            # Reset backoff counter on successful claim
            consecutive_no_work = 0

            try:
                # Check if range is empty first
                if _is_range_empty(work_unit, src_db_path, pipeline_config):
                    # Empty range - skip processing and splitting, just mark as completed
                    work_tracker.complete_work_unit(work_unit.unit_id)
                    continue

                # Track local counters for this work unit
                local_counters = {'scanned': 0, 'filtered': 0, 'enqueued': 0, 'written': 0}

                was_split, _ = _process_work_unit(
                    work_unit,
                    src_db_path,
                    output_dir,
                    ingest_queue,
                    processor,
                    worker_config,
                    pipeline_config,
                    filter_config,
                    local_counters,
                    work_tracker,
                    counters
                )

                # Counters are now updated at each checkpoint (every flush_interval_s)
                # No need to update again here - would cause double counting
                # Keep local_counters for per-shard statistics/debugging

                processed_units += 1

            except Exception as e:
                # Actual failure - mark unit as failed
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
        from ..filters.shared_vocab import MMapVocab
        vocab_view = MMapVocab(str(filter_config.vocab_path))
        return replace(filter_config, vocab_view=vocab_view)

    except Exception as e:
        print(f"Worker {worker_id} vocabulary error: {e}")
        return filter_config


def _claim_next_work_unit(
    worker_id: int, work_tracker: WorkTracker, output_manager: SimpleOutputManager
) -> Optional[WorkUnit]:
    """Attempt to claim the next available work unit and clean up partial outputs."""
    try:
        work_unit = work_tracker.claim_work_unit(f"worker-{worker_id}", max_retries=10)

        # Clean up partial output if it exists from a previous failed run
        if work_unit and output_manager.output_exists(work_unit.unit_id):
            output_manager.cleanup_partial_output(work_unit.unit_id)

        return work_unit
    except Exception as e:
        print(f"Worker {worker_id} failed to claim work unit: {e}")
        return None


def _is_range_empty(work_unit: WorkUnit, src_db_path: Path, pipeline_config: PipelineConfig) -> bool:
    """Check if a work unit's key range contains no data.

    This performs a quick check by looking for the first key in the range.
    If no key is found, the range is empty and doesn't need processing or splitting.

    Args:
        work_unit: The work unit to check
        src_db_path: Path to source database
        pipeline_config: Pipeline configuration

    Returns:
        True if the range contains no data, False otherwise
    """
    start_key = work_unit.start_key if work_unit.start_key is not None else b""
    end_key = work_unit.end_key

    try:
        with open_db(src_db_path, mode="r", profile=pipeline_config.reader_profile) as src_db:
            # Check if there's at least one key in the range (excluding metadata)
            for key, _ in range_scan(src_db, start_key, end_key):
                # Skip metadata keys
                if key.startswith(METADATA_PREFIX):
                    continue
                # Found at least one non-metadata key
                return False
            # No non-metadata keys found
            return True
    except Exception:
        # If we can't check, assume it's not empty to be safe
        return False


def _process_work_unit(
        work_unit: WorkUnit,
        src_db_path: Path,
        output_dir: Path,
        ingest_queue: Optional[mp.Queue],
        processor,
        config: WorkerConfig,
        pipeline_config: PipelineConfig,
        filter_config: FilterConfig,
        local_counters: Optional[dict] = None,
        work_tracker: Optional[WorkTracker] = None,
        counters: Optional[Counters] = None,
) -> tuple[bool, tuple[int, int, int]]:
    """
    Process a single work unit by filtering its key range.

    This function handles the core filtering work. It processes the entire range
    from start_key to end_key, flushing periodically and checkpointing progress.
    After filtering completes, the shard path is sent to the ingest writer queue.

    Worker-Driven Splitting:
        Workers may split their own units before processing starts if idle workers
        are detected. This shrinks the work_unit's end_key, causing the worker to
        naturally complete earlier without any special detection logic needed.

    Args:
        work_unit: The work unit to process
        src_db_path: Path to source database
        output_dir: Directory for output databases
        ingest_queue: Queue to send completed shard to ingest writer
        processor: Filter processor function
        config: Worker configuration
        pipeline_config: Pipeline configuration
        local_counters: Optional local counters for tracking work before commit
        work_tracker: Optional work tracker for checkpointing progress

    Returns:
        Tuple of (was_split, unused_tuple):
        - was_split: Always False (splits no longer detected during processing)
        - unused_tuple: Always (0, 0, 0) - kept for API compatibility
    """
    # Create output database for this work unit
    output_path = output_dir / f"{work_unit.unit_id}.db"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize write buffer
    buffer = WriteBuffer()

    # Process the work unit's key range
    was_split = False
    with open_db(
            src_db_path,
            mode="r",
            profile=pipeline_config.reader_profile
    ) as src_db:
        with open_db(
                output_path,
                mode="rw",
                profile=pipeline_config.writer_profile,
                create_if_missing=True
        ) as dst_db:
            was_split, _ = _process_key_range(
                work_unit,
                src_db,
                dst_db,
                src_db_path,
                processor,
                buffer,
                config,
                pipeline_config,
                filter_config,
                local_counters,
                work_tracker,
                counters
            )

            # Final flush and finalization
            # Always flush remaining buffer, whether split or not
            num_written = buffer.flush_and_count(dst_db, pipeline_config.writer_disable_wal)
            if local_counters:
                local_counters['written'] += num_written
            _finalize_output_database(dst_db)

    # Mark unit as completed
    if work_tracker:
        work_tracker.complete_work_unit(work_unit.unit_id)

    # Send completed shard to ingest writer queue
    # The writer will handle ingestion and mark as ingested
    if ingest_queue:
        ingest_queue.put((work_unit.unit_id, output_path))

    return was_split, (0, 0, 0)


def _process_key_range(
        work_unit: WorkUnit,
        src_db,
        dst_db,
        src_db_path: Path,
        processor,
        buffer: WriteBuffer,
        config: WorkerConfig,
        pipeline_config: PipelineConfig,
        filter_config: FilterConfig,
        local_counters: Optional[dict] = None,
        work_tracker: Optional[WorkTracker] = None,
        counters: Optional[Counters] = None,
) -> tuple[bool, tuple[int, int, int]]:
    """
    Process all keys in the work unit's range.

    This function scans the source database from start_key to end_key and writes
    filtered results to a buffer. It flushes periodically and checkpoints progress.

    Work units are split at claim-time (before processing starts) if idle workers exist.
    This function simply processes the given range without any mid-processing splits.

    Returns:
        Tuple of (was_split, unused_tuple):
        - was_split: Always False (splitting now happens at claim time)
        - unused_tuple: Always (0, 0, 0) - kept for API compatibility
    """
    import time

    start_key = work_unit.start_key if work_unit.start_key is not None else b""
    end_key = work_unit.end_key

    # Track counts since last flush (for batching counter updates)
    since_flush = {'scanned': 0, 'filtered': 0, 'enqueued': 0}

    # Track time for periodic flushes
    last_flush_time = time.time()
    flush_interval_s = getattr(pipeline_config, 'flush_interval_s', 5.0)

    for key, value in range_scan(src_db, start_key, end_key):
        # Skip metadata keys entirely (don't count them)
        if key.startswith(METADATA_PREFIX):
            continue

        since_flush['scanned'] += 1

        # Check if we should flush (time-based, checked EVERY iteration)
        # This ensures flush happens even when all keys are filtered out
        should_flush = False
        if flush_interval_s > 0:
            current_time = time.time()
            if current_time - last_flush_time >= flush_interval_s:
                should_flush = True

        # Apply filtering
        processed_key = processor(key)
        if not processed_key:
            since_flush['filtered'] += 1
            # If flush interval elapsed, flush even though key was filtered
            if should_flush:
                # Flush buffer (may be empty if all recent keys were filtered)
                num_written = buffer.flush_and_count(dst_db, pipeline_config.writer_disable_wal)
                if local_counters:
                    local_counters['written'] += num_written

                # Update counters at flush boundary
                if local_counters:
                    local_counters['scanned'] += since_flush['scanned']
                    local_counters['filtered'] += since_flush['filtered']
                    local_counters['enqueued'] += since_flush['enqueued']
                    local_counters['written'] += num_written

                    # Also update shared counters immediately for live progress
                    if counters:
                        increment_counter(counters.items_scanned, since_flush['scanned'])
                        increment_counter(counters.items_filtered, since_flush['filtered'])
                        increment_counter(counters.items_enqueued, since_flush['enqueued'])
                        increment_counter(counters.items_written, num_written)

                    since_flush = {'scanned': 0, 'filtered': 0, 'enqueued': 0}

                last_flush_time = time.time()

            continue

        # Prepare for buffering
        processed_key_bytes = _ensure_bytes(processed_key)
        value_bytes = _ensure_bytes(value)

        # Apply year binning if configured
        if filter_config.bin_size > 1:
            value_bytes = bin_year_records(value_bytes, filter_config.bin_size)

        since_flush['enqueued'] += 1

        # Add to buffer
        buffer.add(processed_key_bytes, value_bytes)

        if should_flush:
            # Flush buffer to disk
            num_written = buffer.flush_and_count(dst_db, pipeline_config.writer_disable_wal)
            if local_counters:
                local_counters['written'] += num_written

            # Update counters at flush boundary
            if local_counters:
                local_counters['scanned'] += since_flush['scanned']
                local_counters['filtered'] += since_flush['filtered']
                local_counters['enqueued'] += since_flush['enqueued']
                local_counters['written'] += num_written

                # Also update shared counters immediately for live progress
                if counters:
                    increment_counter(counters.items_scanned, since_flush['scanned'])
                    increment_counter(counters.items_filtered, since_flush['filtered'])
                    increment_counter(counters.items_enqueued, since_flush['enqueued'])
                    increment_counter(counters.items_written, num_written)

                since_flush = {'scanned': 0, 'filtered': 0, 'enqueued': 0}

            last_flush_time = time.time()

    # Completed processing entire range
    # Commit any remaining counts from partial buffer and update counters
    if local_counters:
        local_counters['scanned'] += since_flush['scanned']
        local_counters['filtered'] += since_flush['filtered']
        local_counters['enqueued'] += since_flush['enqueued']

        # Also update shared counters with final batch
        if counters:
            increment_counter(counters.items_scanned, since_flush['scanned'])
            increment_counter(counters.items_filtered, since_flush['filtered'])
            increment_counter(counters.items_enqueued, since_flush['enqueued'])

    return False, (0, 0, 0)


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
