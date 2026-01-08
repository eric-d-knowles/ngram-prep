"""Worker process for pivoting ngram databases."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle

from ..config import PipelineConfig
from ..encoding import decode_packed24_records, encode_year_ngram_key, encode_year_stats
from .progress import Counters, increment_counter
from .write_buffer import WriteBuffer
from ngramprep.common_db.api import open_db, range_scan
from ngramprep.tracking import WorkTracker, WorkUnit, SimpleOutputManager

__all__ = ["WorkerConfig", "worker_process", "worker_process_sst"]


@dataclass
class WorkerConfig:
    """Configuration for individual worker processes."""

    disable_wal: bool = True  # Disable WAL for performance
    disable_compaction: bool = True  # Disable auto-compaction


def worker_process_sst(
        worker_id: int,
        src_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
        pipeline_config: PipelineConfig,
        worker_config: WorkerConfig,
        counters: Optional[Counters] = None,
) -> None:
    """
    Worker process that writes shards directly as SST files (for direct_sst ingest mode).

    This is similar to worker_process() but writes data directly to SST files using
    SstFileWriter instead of creating RocksDB databases. This enables much faster
    ingestion via DB.ingest().

    Args:
        worker_id: Unique identifier for this worker
        src_db_path: Path to source database
        work_tracker_path: Path to work tracker database
        output_dir: Directory for output SST files
        pipeline_config: Configuration for DB operations
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking
    """
    try:
        # Set process title for system monitoring
        setproctitle(f"ngp:worker[{worker_id:03d}]")

        # Add small random startup delay to reduce thundering herd at initialization
        import random
        import time
        time.sleep(random.uniform(0, 0.5))

        # Initialize work tracker
        work_tracker = WorkTracker(
            work_tracker_path,
            claim_order=pipeline_config.work_unit_claim_order
        )

        # Initialize output manager
        output_manager = SimpleOutputManager(output_dir, extension=".sst")

        processed_units = 0
        consecutive_no_work = 0  # Track consecutive failed claims for backoff

        # Main work loop
        while True:
            work_unit = _claim_next_work_unit(worker_id, work_tracker, output_manager)

            if work_unit is None:
                # No work available - use exponential backoff to reduce thundering herd
                consecutive_no_work += 1

                # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s (max)
                # Add jitter to avoid synchronized retries
                base_delay = min(0.5 * (2 ** (consecutive_no_work - 1)), 8.0)
                jitter = random.uniform(0, base_delay * 0.3)
                delay = base_delay + jitter

                # Exit after several failed attempts to avoid infinite waiting
                if consecutive_no_work >= 5:
                    # No work for 5 attempts - likely all work is done
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
                local_counters = {'scanned': 0, 'decoded': 0, 'written': 0}

                _process_work_unit_to_sst(
                    work_unit,
                    src_db_path,
                    output_dir,
                    worker_config,
                    pipeline_config,
                    local_counters,
                    work_tracker,
                    counters
                )

                processed_units += 1

            except Exception as e:
                # Actual failure - mark unit as failed
                print(f"Worker {worker_id} failed on unit {work_unit.unit_id}: {e}")
                traceback.print_exc()
                work_tracker.fail_work_unit(work_unit.unit_id)

    except Exception as e:
        print(f"Worker {worker_id} crashed during startup: {e}")
        traceback.print_exc()


def worker_process(
        worker_id: int,
        src_db_path: Path,
        work_tracker_path: Path,
        output_dir: Path,
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
        pipeline_config: Configuration for DB operations
        worker_config: Worker-specific configuration
        counters: Optional shared counters for progress tracking
    """
    try:
        # Set process title for system monitoring
        setproctitle(f"ngp:worker[{worker_id:03d}]")

        # Add small random startup delay to reduce thundering herd at initialization
        import random
        import time
        time.sleep(random.uniform(0, 0.5))

        # Initialize work tracker
        work_tracker = WorkTracker(
            work_tracker_path,
            claim_order=pipeline_config.work_unit_claim_order
        )

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

                # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s (max)
                # Add jitter to avoid synchronized retries
                base_delay = min(0.5 * (2 ** (consecutive_no_work - 1)), 8.0)
                jitter = random.uniform(0, base_delay * 0.3)
                delay = base_delay + jitter

                # Exit after several failed attempts to avoid infinite waiting
                if consecutive_no_work >= 5:
                    # No work for 5 attempts - likely all work is done
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
                local_counters = {'scanned': 0, 'decoded': 0, 'written': 0}

                was_split, _ = _process_work_unit(
                    work_unit,
                    src_db_path,
                    output_dir,
                    worker_config,
                    pipeline_config,
                    local_counters,
                    work_tracker,
                    counters
                )

                processed_units += 1

            except Exception as e:
                # Actual failure - mark unit as failed
                print(f"Worker {worker_id} failed on unit {work_unit.unit_id}: {e}")
                traceback.print_exc()
                work_tracker.fail_work_unit(work_unit.unit_id)

    except Exception as e:
        print(f"Worker {worker_id} crashed during startup: {e}")
        traceback.print_exc()


def _claim_next_work_unit(
    worker_id: int, work_tracker: WorkTracker, output_manager: SimpleOutputManager
) -> Optional[WorkUnit]:
    """Attempt to claim the next available work unit and clean up partial outputs."""
    try:
        work_unit = work_tracker.claim_work_unit(f"worker-{worker_id}", max_retries=50)

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
            # Check if there's at least one key in the range
            for key, _ in range_scan(src_db, start_key, end_key):
                # Found at least one key
                return False
            # No keys found
            return True
    except Exception:
        # If we can't check, assume it's not empty to be safe
        return False


def _process_work_unit_to_sst(
        work_unit: WorkUnit,
        src_db_path: Path,
        output_dir: Path,
        config: WorkerConfig,
        pipeline_config: PipelineConfig,
        local_counters: Optional[dict] = None,
        work_tracker: Optional[WorkTracker] = None,
        counters: Optional[Counters] = None,
) -> None:
    """
    Process a work unit by writing pivoted data directly to an SST file.

    This function is similar to _process_work_unit but uses SstFileWriter to create
    SST files that can be directly ingested, bypassing memtable writes entirely.

    Args:
        work_unit: The work unit to process
        src_db_path: Path to source database
        output_dir: Directory for output SST files
        config: Worker configuration
        pipeline_config: Pipeline configuration
        local_counters: Optional local counters for tracking work
        work_tracker: Optional work tracker for checkpointing progress
    """
    import rocks_shim as rs

    # Create output SST file for this work unit
    output_path = output_dir / f"{work_unit.unit_id}.sst"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open SST writer
    writer = rs.SstFileWriter()
    writer.open(str(output_path))

    try:
        # Process key range and write to SST
        with open_db(
                src_db_path,
                mode="r",
                profile=pipeline_config.reader_profile
        ) as src_db:
            _process_key_range_to_sst(
                work_unit,
                src_db,
                writer,
                pipeline_config,
                local_counters,
                work_tracker,
                counters
            )

        # Finalize SST file
        writer.finish()

        # Mark unit as completed
        if work_tracker:
            work_tracker.complete_work_unit(work_unit.unit_id, max_retries=20)

    except Exception:
        # Clean up partial SST file on failure
        try:
            output_path.unlink()
        except Exception:
            pass
        raise


def _process_key_range_to_sst(
        work_unit: WorkUnit,
        src_db,
        writer,
        pipeline_config: PipelineConfig,
        local_counters: Optional[dict] = None,
        work_tracker: Optional[WorkTracker] = None,
        counters: Optional[Counters] = None,
) -> None:
    """
    Process key range and write directly to SST file using SstFileWriter.

    Args:
        work_unit: Work unit defining key range
        src_db: Source database handle
        writer: SstFileWriter instance
        pipeline_config: Pipeline configuration
        local_counters: Optional counters for tracking
        work_tracker: Optional work tracker for checkpointing
    """
    import time

    start_key = work_unit.start_key if work_unit.start_key is not None else b""
    end_key = work_unit.end_key

    # Track counts
    since_last_update = {'scanned': 0, 'decoded': 0, 'written': 0}
    last_update_time = time.time()
    update_interval_s = getattr(pipeline_config, 'flush_interval_s', 5.0)

    # Pre-compute year prefixes
    year_prefixes = _precompute_year_prefixes(min_year=1400, max_year=2021)

    # Collect all pivoted key-value pairs in memory before writing
    # (SstFileWriter requires keys to be added in sorted order)
    pivoted_data = []

    for ngram_key, packed_value in range_scan(src_db, start_key, end_key):
        since_last_update['scanned'] += 1

        try:
            # Decode all year records for this n-gram
            year_records = decode_packed24_records(packed_value)
            since_last_update['decoded'] += len(year_records)

            # Create pivoted key-value pairs
            for year, occurrences, documents in year_records:
                # Use pre-computed prefix (or cache outliers dynamically)
                if year not in year_prefixes:
                    year_prefixes[year] = encode_year_ngram_key(year, b"")
                target_key = year_prefixes[year] + ngram_key
                target_value = encode_year_stats(occurrences, documents)
                pivoted_data.append((target_key, target_value))

            # Periodically update shared counters for progress reporting
            if update_interval_s > 0:
                current_time = time.time()
                if current_time - last_update_time >= update_interval_s:
                    if local_counters:
                        local_counters['scanned'] += since_last_update['scanned']
                        local_counters['decoded'] += since_last_update['decoded']

                        # Update shared counters for live progress
                        if counters:
                            increment_counter(counters.items_scanned, since_last_update['scanned'])
                            increment_counter(counters.items_decoded, since_last_update['decoded'])

                        since_last_update = {'scanned': 0, 'decoded': 0, 'written': 0}

                    last_update_time = current_time

        except Exception as e:
            if pipeline_config.validate:
                raise
            # Otherwise skip this record and continue

    # Sort all pivoted data by key (required for SST writer)
    pivoted_data.sort(key=lambda x: x[0])

    # Write all sorted data to SST file
    for key, value in pivoted_data:
        writer.put(key, value)
        since_last_update['written'] += 1

    # Commit final counts
    if local_counters:
        local_counters['scanned'] += since_last_update['scanned']
        local_counters['decoded'] += since_last_update['decoded']
        local_counters['written'] += since_last_update['written']

        # Update shared counters with final batch
        if counters:
            increment_counter(counters.items_scanned, since_last_update['scanned'])
            increment_counter(counters.items_decoded, since_last_update['decoded'])
            increment_counter(counters.items_written, since_last_update['written'])


def _process_work_unit(
        work_unit: WorkUnit,
        src_db_path: Path,
        output_dir: Path,
        config: WorkerConfig,
        pipeline_config: PipelineConfig,
        local_counters: Optional[dict] = None,
        work_tracker: Optional[WorkTracker] = None,
        counters: Optional[Counters] = None,
) -> tuple[bool, tuple[int, int, int]]:
    """
    Process a single work unit by pivoting its key range.

    This function handles the core pivot work. It processes the entire range
    from start_key to end_key, flushing periodically and checkpointing progress.
    After pivoting completes, the shard is written to disk and marked as completed.

    Worker-Driven Splitting:
        Workers may split their own units before processing starts if idle workers
        are detected. This shrinks the work_unit's end_key, causing the worker to
        naturally complete earlier without any special detection logic needed.

    Args:
        work_unit: The work unit to process
        src_db_path: Path to source database
        output_dir: Directory for output databases
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
                buffer,
                config,
                pipeline_config,
                local_counters,
                work_tracker,
                counters
            )

            # Final flush and finalization
            # Always flush remaining buffer, whether split or not
            num_written = buffer.flush_and_count(dst_db, pipeline_config.writer_disable_wal)
            if local_counters:
                local_counters['written'] += num_written

            # Explicitly finalize BEFORE exiting with block to ensure data is on disk
            # The context manager will also try to finalize, but we need it to happen NOW
            # before we mark as completed, not in the finally block
            try:
                dst_db.finalize_bulk()
            except Exception:
                pass  # May not be available

    # Mark unit as completed now that pivoting is done
    # Database is now fully flushed (data on disk) and closed (with block exited above)
    # This makes the shard visible to ingest readers via work tracker (disk-based pre-queue!)
    # No queue handoff needed - readers poll work tracker for completed units
    if work_tracker:
        work_tracker.complete_work_unit(work_unit.unit_id, max_retries=20)

    return was_split, (0, 0, 0)


def _precompute_year_prefixes(min_year: int = 1400, max_year: int = 2021) -> dict:
    """
    Pre-compute year prefixes for common year range.

    This eliminates struct.pack() overhead in the tight encoding loop for years
    in the range [min_year, max_year). Outlier years will be cached dynamically.

    Args:
        min_year: Minimum year to pre-compute (inclusive)
        max_year: Maximum year to pre-compute (exclusive)

    Returns:
        Dictionary mapping year -> 4-byte big-endian prefix
    """
    import struct
    prefixes = {}
    for year in range(min_year, max_year):
        prefixes[year] = struct.pack('>I', year)
    return prefixes


def _process_key_range(
        work_unit: WorkUnit,
        src_db,
        dst_db,
        src_db_path: Path,
        buffer: WriteBuffer,
        config: WorkerConfig,
        pipeline_config: PipelineConfig,
        local_counters: Optional[dict] = None,
        work_tracker: Optional[WorkTracker] = None,
        counters: Optional[Counters] = None,
) -> tuple[bool, tuple[int, int, int]]:
    """
    Process all keys in the work unit's range.

    This function scans the source database from start_key to end_key and writes
    pivoted results to a buffer. It flushes periodically and checkpoints progress.

    Returns:
        Tuple of (was_split, unused_tuple):
        - was_split: Always False (splits no longer detected during processing)
        - unused_tuple: Always (0, 0, 0) - kept for API compatibility
    """
    import time

    start_key = work_unit.start_key if work_unit.start_key is not None else b""
    end_key = work_unit.end_key

    # Track counts
    since_flush = {'scanned': 0, 'decoded': 0}

    # Track time for periodic flushes
    last_flush_time = time.time()
    flush_interval_s = getattr(pipeline_config, 'flush_interval_s', 5.0)

    # Pre-compute year prefixes for common range (1400-2020)
    # Outlier years will be cached dynamically on first encounter
    year_prefixes = _precompute_year_prefixes(min_year=1400, max_year=2021)

    for ngram_key, packed_value in range_scan(src_db, start_key, end_key):
        since_flush['scanned'] += 1

        try:
            # Decode all year records for this n-gram
            year_records = decode_packed24_records(packed_value)
            since_flush['decoded'] += len(year_records)

            # Encode and buffer immediately (direct processing - no batch accumulation)
            for year, occurrences, documents in year_records:
                # Use pre-computed prefix (or cache outliers dynamically)
                if year not in year_prefixes:
                    # Outlier year outside 1400-2020 range - cache it
                    year_prefixes[year] = encode_year_ngram_key(year, b"")
                target_key = year_prefixes[year] + ngram_key
                target_value = encode_year_stats(occurrences, documents)
                buffer.add(target_key, target_value)

            # Check if we should flush (time-based only)
            should_flush = False
            if flush_interval_s > 0:
                current_time = time.time()
                if current_time - last_flush_time >= flush_interval_s:
                    should_flush = True

            if should_flush:
                    # Flush buffer to disk
                    num_written = buffer.flush_and_count(dst_db, pipeline_config.writer_disable_wal)
                    if local_counters:
                        local_counters['written'] += num_written

                    # Commit counts at flush boundary
                    if local_counters:
                        local_counters['scanned'] += since_flush['scanned']
                        local_counters['decoded'] += since_flush['decoded']

                        # Also update shared counters immediately for live progress
                        if counters:
                            increment_counter(counters.items_scanned, since_flush['scanned'])
                            increment_counter(counters.items_decoded, since_flush['decoded'])
                            increment_counter(counters.items_written, num_written)

                        since_flush = {'scanned': 0, 'decoded': 0}

                    last_flush_time = time.time()

        except Exception as e:
            if pipeline_config.validate:
                raise
            # Otherwise skip this record and continue

    # Completed processing entire range
    # Commit any remaining counts from partial buffer
    if local_counters:
        local_counters['scanned'] += since_flush['scanned']
        local_counters['decoded'] += since_flush['decoded']

        # Also update shared counters with final batch
        if counters:
            increment_counter(counters.items_scanned, since_flush['scanned'])
            increment_counter(counters.items_decoded, since_flush['decoded'])

    return False, (0, 0, 0)


def _finalize_output_database(dst_db) -> None:
    """Finalize the output database."""
    try:
        dst_db.finalize_bulk()
    except Exception:
        pass  # Method may not be implemented
