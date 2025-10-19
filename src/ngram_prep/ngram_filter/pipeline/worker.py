"""Worker process for filtering ngram databases."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle

from ..config import FilterConfig, PipelineConfig
from ..filters.processor_factory import build_processor
from ..filters.core_cy import METADATA_PREFIX
from .progress import Counters, increment_counter
from .write_buffer import WriteBuffer
from ngram_prep.common_db.api import open_db, range_scan
from ngram_prep.tracking import WorkTracker, WorkUnit, SimpleOutputManager

__all__ = ["WorkerConfig", "worker_process"]


@dataclass
class WorkerConfig:
    """Configuration for individual worker processes."""

    buffer_size: int = 10_000  # Items to buffer before flushing
    buffer_bytes: int = 8 * 1024 * 1024  # 8MB buffer limit
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
        setproctitle(f"ngf:worker[{worker_id}]")

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

        # Main work loop
        while True:
            work_unit = _claim_next_work_unit(worker_id, work_tracker, output_manager)

            if work_unit is None:
                # Check if there's still any work in progress or pending
                progress = work_tracker.get_progress()

                if progress.processing > 0 or progress.pending > 0:
                    # Wait for more work to become available
                    import time
                    time.sleep(0.5)
                    continue
                # All work is done
                break

            try:
                # Check if range is empty first
                if _is_range_empty(work_unit, src_db_path, pipeline_config):
                    # Empty range - skip processing and splitting, just mark as ingested
                    work_tracker.ingest_work_unit(work_unit.unit_id)
                    continue

                # Check if we should split this unit before processing (worker-driven splitting)
                # Split if there are idle workers who could help with this work
                progress = work_tracker.get_progress(num_workers=pipeline_config.num_workers)

                if progress.processing < pipeline_config.num_workers and progress.idle_workers > 0:
                    # Only split if we haven't exceeded maximum split depth
                    split_depth = _get_split_depth(work_unit, work_tracker)
                    if split_depth >= pipeline_config.max_split_depth:
                        # Don't split - already split too many times
                        pass
                    else:
                        try:
                            # Try to split - may fail if unit has no progress yet or is too small
                            # This shrinks work_unit's end_key in DB and creates a child for the remainder
                            child = work_tracker.split_current_unit(work_unit.unit_id)
                            if child:
                                # Reload work_unit to get the updated (shrunk) end_key
                                work_unit = work_tracker.get_work_unit(work_unit.unit_id)
                                if not work_unit:
                                    # Unit disappeared (race condition) - skip to next iteration
                                    continue
                        except (ValueError, Exception) as e:
                            # Split failed - that's OK, continue processing full unit
                            pass

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
                    local_counters,
                    work_tracker
                )

                # Commit counters for the work we completed
                # All counts are accurate - no adjustments needed
                if counters and local_counters:
                    increment_counter(counters.items_scanned, local_counters['scanned'])
                    increment_counter(counters.items_filtered, local_counters['filtered'])
                    increment_counter(counters.items_enqueued, local_counters['enqueued'])
                    increment_counter(counters.items_written, local_counters['written'])

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
        with open_db(src_db_path, mode="r", profile=pipeline_config.writer_read_profile) as src_db:
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


def _get_split_depth(work_unit: WorkUnit, work_tracker: WorkTracker) -> int:
    """Calculate how many times this unit has been split (parent chain depth).

    This traverses the parent chain to determine split depth, which prevents
    infinite splitting by limiting how many times a unit can be recursively split.

    Args:
        work_unit: The work unit to check
        work_tracker: Work tracker to look up parent chain

    Returns:
        Depth (0 = original unit, 1 = split once, 2 = split twice, etc.)
    """
    depth = 0
    current_id = work_unit.parent_id

    # Traverse parent chain
    while current_id is not None:
        depth += 1
        parent = work_tracker.get_work_unit(current_id)
        if not parent:
            break  # Parent not found, stop traversal
        current_id = parent.parent_id

        # Safety limit to prevent infinite loops in case of circular parent references
        if depth > 100:
            break

    return depth


def _process_work_unit(
        work_unit: WorkUnit,
        src_db_path: Path,
        output_dir: Path,
        ingest_queue: Optional[mp.Queue],
        processor,
        config: WorkerConfig,
        pipeline_config: PipelineConfig,
        local_counters: Optional[dict] = None,
        work_tracker: Optional[WorkTracker] = None,
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
    buffer = WriteBuffer(config.buffer_size, config.buffer_bytes)

    # Process the work unit's key range
    was_split = False
    with open_db(
            src_db_path,
            mode="r",
            profile=pipeline_config.writer_read_profile
    ) as src_db:
        with open_db(
                output_path,
                mode="rw",
                profile=pipeline_config.writer_write_profile,
                create_if_missing=True
        ) as dst_db:
            was_split, _ = _process_key_range(
                work_unit,
                src_db,
                dst_db,
                processor,
                buffer,
                config,
                pipeline_config,
                local_counters,
                work_tracker
            )

            # Final flush and finalization
            # Always flush remaining buffer, whether split or not
            num_written = buffer.flush_and_count(dst_db, pipeline_config.writer_disable_wal)
            if local_counters:
                local_counters['written'] += num_written
            _finalize_output_database(dst_db)

    # Mark unit as completed now that filtering is done
    # This releases the worker from being counted as "active" in progress tracking
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
        processor,
        buffer: WriteBuffer,
        config: WorkerConfig,
        pipeline_config: PipelineConfig,
        local_counters: Optional[dict] = None,
        work_tracker: Optional[WorkTracker] = None,
) -> tuple[bool, tuple[int, int, int]]:
    """
    Process all keys in the work unit's range.

    This function scans the source database from start_key to end_key and writes
    filtered results to a buffer. It flushes periodically and checkpoints progress.

    Periodic splitting: Every split_check_interval_s (default 120s), the worker checks
    if there are idle workers. If so, it splits its remaining work at the current position,
    creating a new pending unit for idle workers to claim.

    Returns:
        Tuple of (was_split, unused_tuple):
        - was_split: Always False (splits no longer detected during processing)
        - unused_tuple: Always (0, 0, 0) - kept for API compatibility
    """
    import time

    start_key = work_unit.start_key if work_unit.start_key is not None else b""
    end_key = work_unit.end_key

    # Track last flushed key for checkpointing
    last_flushed_key = None

    # Track counts since last checkpoint (to avoid double-counting on splits)
    since_checkpoint = {'scanned': 0, 'filtered': 0, 'enqueued': 0}

    # Track time for periodic split checks
    last_split_check_time = time.time()
    split_check_interval_s = getattr(pipeline_config, 'split_check_interval_s', 120)

    for key, value in range_scan(src_db, start_key, end_key):
        # Skip metadata keys entirely (don't count them)
        if key.startswith(METADATA_PREFIX):
            continue

        since_checkpoint['scanned'] += 1

        # Apply filtering
        processed_key = processor(key)
        if not processed_key:
            since_checkpoint['filtered'] += 1
            continue

        # Prepare for buffering
        processed_key_bytes = _ensure_bytes(processed_key)
        value_bytes = _ensure_bytes(value)

        since_checkpoint['enqueued'] += 1

        # Add to buffer and flush if needed
        should_flush = buffer.add(processed_key_bytes, value_bytes)
        if should_flush:
            # Update checkpoint position (with more retries for high-worker scenarios)
            if work_tracker:
                work_tracker.checkpoint_position(work_unit.unit_id, key, max_retries=10)

            # Flush buffer to disk
            num_written = buffer.flush_and_count(dst_db, pipeline_config.writer_disable_wal)
            if local_counters:
                local_counters['written'] += num_written

            # Commit counts at checkpoint boundary
            if local_counters:
                local_counters['scanned'] += since_checkpoint['scanned']
                local_counters['filtered'] += since_checkpoint['filtered']
                local_counters['enqueued'] += since_checkpoint['enqueued']
                since_checkpoint = {'scanned': 0, 'filtered': 0, 'enqueued': 0}

            # Periodic split check: if enough time has passed and there are idle workers,
            # split off remaining work for them to claim
            if work_tracker and split_check_interval_s > 0:
                current_time = time.time()
                if current_time - last_split_check_time >= split_check_interval_s:
                    last_split_check_time = current_time

                    # Check if there are idle workers
                    progress = work_tracker.get_progress(num_workers=pipeline_config.num_workers)
                    if progress.idle_workers > 0:
                        # Check if we haven't exceeded maximum split depth
                        split_depth = _get_split_depth(work_unit, work_tracker)
                        if split_depth < pipeline_config.max_split_depth:
                            try:
                                # Split at current position - creates child for remaining work
                                # This preserves our completed work and gives idle workers the remainder
                                child = work_tracker.split_work_unit(work_unit.unit_id)
                                if child:
                                    # Unit was split and marked completed - exit scan loop
                                    # We've already flushed everything up to current_position
                                    break
                            except (ValueError, Exception):
                                # Split failed (e.g., no progress yet, or position too close to end)
                                # This is fine - just continue processing
                                pass

    # Completed processing entire range
    # Commit any remaining counts from partial buffer and update counters
    if local_counters:
        local_counters['scanned'] += since_checkpoint['scanned']
        local_counters['filtered'] += since_checkpoint['filtered']
        local_counters['enqueued'] += since_checkpoint['enqueued']

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
