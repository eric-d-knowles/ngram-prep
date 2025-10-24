"""Main entry point for ngram acquisition pipeline."""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, Tuple, Type

from ngram_prep.ngram_acquire.coordinator import (
    discover_files,
    select_file_subset,
    filter_processed_files,
    randomize_file_order,
)
from ngram_prep.ngram_acquire.executor import process_files
from ngram_prep.ngram_acquire.reporter import print_pipeline_header, print_final_summary
from ngram_prep.ngram_acquire.utils.filters import make_ngram_type_predicate
from ngram_prep.ngram_acquire.utils.cleanup import safe_db_cleanup
from ngram_prep.ngram_acquire.db.build_path import build_db_path
from ngram_prep.ngram_acquire.db.write import DEFAULT_WRITE_BATCH_SIZE
from ngram_prep.common_db.api import open_db

logger = logging.getLogger(__name__)

__all__ = ["download_and_ingest_to_rocksdb"]

try:
    import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None


def download_and_ingest_to_rocksdb(
        ngram_size: int,
        repo_release_id: str,
        repo_corpus_id: str,
        db_path_stub: str,
        file_range: Optional[Tuple[int, int]] = None,
        workers: Optional[int] = None,
        use_threads: bool = False,
        ngram_type: str = "all",
        overwrite_db: bool = True,
        random_seed: Optional[int] = None,
        write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
        open_type: str = "read",
        compact_after_ingest: bool = False,
) -> None:
    """
    Main pipeline: discover, download, parse, and ingest ngram files into RocksDB.

    Orchestrates the complete ngram acquisition workflow:
    1. Builds database path from stub and parameters
    2. Discovers available files from repository
    3. Opens/creates RocksDB with specified profile
    4. Downloads and processes files concurrently
    5. Writes batched results to database
    6. Optionally performs post-ingestion compaction

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")
        db_path_stub: Base directory for database (will be expanded)
        file_range: Optional (start_idx, end_idx) to process subset of files
        workers: Number of concurrent workers (default: min(40, cpu_count * 2))
        use_threads: If True, use threads; otherwise use processes
        ngram_type: Filter type ("all", "tagged", etc.)
        overwrite_db: If True, remove existing database before starting
        random_seed: Optional seed for randomizing file processing order
        write_batch_size: Number of entries per batch write
        open_type: RocksDB profile ("read", "write", "read:packed24", "write:packed24")
        compact_after_ingest: If True, perform full compaction after ingestion
    """
    logger.info("Starting N-gram processing pipeline")

    # Set process title if available
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle("nga:main")
        except Exception:
            pass

    start_time = datetime.now()

    # Build full database path from stub
    db_path = build_db_path(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)
    logger.info("Database path: %s", db_path)

    # Determine worker count
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = min(40, cpu * 2)

    # Handle existing database
    if overwrite_db and os.path.exists(db_path):
        logger.info("Removing existing database for fresh start")
        if not safe_db_cleanup(db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {db_path}. "
                "Close open handles or remove it manually."
            )
        logger.info("Successfully removed existing database")

    # Ensure parent directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Discover available files
    page_url, file_urls_available = discover_files(ngram_size, repo_release_id, repo_corpus_id)

    # Select file subset
    file_urls_to_use, start_idx, end_idx = select_file_subset(file_urls_available, file_range)

    # Validate and normalize open_type
    open_type = open_type.lower()
    valid_profiles = {"read", "write", "read:packed24", "write:packed24"}
    if open_type not in valid_profiles:
        raise ValueError(f"open_type must be one of {valid_profiles}, got {open_type!r}")
    logger.info("Using RocksDB profile: %s", open_type)

    # Open database with specified profile (flush happens automatically on exit)
    with open_db(db_path, profile=open_type, create_if_missing=True) as db:
        # Resume mode: skip already-processed files
        files_to_skip = 0
        if not overwrite_db:
            file_urls_to_use, files_to_skip = filter_processed_files(file_urls_to_use, db)

        # Optional randomization
        if random_seed is not None:
            randomize_file_order(file_urls_to_use, random_seed)

        # Configure executor
        executor_class: Type = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        filter_pred = make_ngram_type_predicate(ngram_type)

        # Print run summary
        print_pipeline_header(
            start_time=start_time,
            page_url=page_url,
            db_path=db_path,
            start_idx=start_idx,
            end_idx=end_idx,
            total_files=len(file_urls_available),
            files_to_get=len(file_urls_to_use),
            files_to_skip=files_to_skip,
            workers=workers,
            write_batch_size=write_batch_size,
            ngram_size=ngram_size,
            ngram_type=ngram_type,
            overwrite_db=overwrite_db,
            open_type=open_type,
        )

        # Process and ingest files
        success, failure, written, batches, uncompressed_bytes = process_files(
            urls=file_urls_to_use,
            executor_class=executor_class,
            workers=workers,
            db=db,
            filter_pred=filter_pred,
            write_batch_size=write_batch_size,
        )
        # Database is automatically flushed by context manager on exit

        # Optional post-ingestion compaction
        if compact_after_ingest:
            _perform_compaction(db, db_path)

    # Report final statistics
    end_time = datetime.now()
    print_final_summary(
        start_time=start_time,
        end_time=end_time,
        success=success,
        failure=failure,
        written=written,
        batches=batches,
        uncompressed_bytes=uncompressed_bytes,
    )


def _perform_compaction(db, db_path: str) -> None:
    """
    Perform full compaction on the database using compact_all().

    Args:
        db: Open RocksDB handle
        db_path: Path to database (for logging)
    """
    from ngram_prep.utilities.display import format_bytes, format_banner

    logger.info("Starting post-ingestion compaction")
    print()
    print(format_banner("Post-Ingestion Compaction"))

    # Get initial size if possible
    try:
        initial_size = db.get_property("rocksdb.total-sst-files-size")
        initial_size = int(initial_size) if initial_size else None
        if initial_size:
            print(f"Initial DB size:         {format_bytes(initial_size)}")
    except Exception:
        initial_size = None

    sys.stdout.flush()

    start_time = time.time()
    try:
        db.compact_all()
        elapsed = time.time() - start_time

        print(f"Compaction completed in {timedelta(seconds=int(elapsed))}")

        # Get final size if possible
        try:
            final_size = db.get_property("rocksdb.total-sst-files-size")
            final_size = int(final_size) if final_size else None
            if initial_size and final_size:
                saved = initial_size - final_size
                pct = (saved / initial_size) * 100
                print(f"Size before:             {format_bytes(initial_size)}")
                print(f"Size after:              {format_bytes(final_size)}")
                print(f"Space saved:             {format_bytes(saved)} ({pct:.1f}%)")
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Compaction failed: {e}")
        print(f"Compaction failed: {e}")
        print("Database is still usable, but may not be optimally compacted.")
