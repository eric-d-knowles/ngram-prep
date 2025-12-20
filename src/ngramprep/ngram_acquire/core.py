"""Main entry point for ngram acquisition pipeline."""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, Tuple, Type

from ngramprep.ngram_acquire.coordinator import (
    discover_files,
    select_file_subset,
    filter_processed_files,
    randomize_file_order,
)
from ngramprep.ngram_acquire.executor import process_files
from ngramprep.ngram_acquire.reporter import print_pipeline_header, print_final_summary
from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate
from ngramprep.ngram_acquire.utils.cleanup import safe_db_cleanup
from ngramprep.ngram_acquire.db.build_path import build_db_path
from ngramprep.ngram_acquire.db.write import DEFAULT_WRITE_BATCH_SIZE
from ngramprep.common_db.api import open_db
from ngramprep.common_db.compress import compress_db

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
        ngram_type: str = "all",
        overwrite_db: bool = True,
        random_seed: Optional[int] = None,
        write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
        open_type: str = "read",
        compact_after_ingest: bool = False,
        archive_path_stub: Optional[str] = None,
        combined_bigrams: Optional[set] = None,
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
    7. Optionally archives (compresses) database to specified directory

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")
        db_path_stub: Base directory for database (will be expanded)
        file_range: Optional (start_idx, end_idx) to process subset of files. Use None for all files.
        workers: Number of concurrent workers (default: min(cpu_count - 1, num_files))
        ngram_type: Filter type ("all", "tagged", etc.)
        overwrite_db: If True, remove existing database before starting
        random_seed: Optional seed for randomizing file processing order
        write_batch_size: Number of entries per batch write
        open_type: RocksDB profile ("read", "write", "read:packed24", "write:packed24")
        compact_after_ingest: If True, perform full compaction after ingestion
        archive_path_stub: Optional archive stub directory. Creates structured path: {archive_path_stub}/{release}/{corpus}/{n}gram_files/{n}grams.db.tar.zst
        combined_bigrams: Optional set of bigrams to combine with hyphens (e.g., {"working class", "middle class"})
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

    # Determine worker count based on available CPUs and number of files
    if workers is None:
        cpu_count = os.cpu_count() or 4
        # Use CPUs - 1 for overhead, capped by number of files to process
        workers = min(cpu_count - 1, len(file_urls_to_use))
        # Ensure at least 1 worker
        workers = max(1, workers)

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

        # Configure executor (always use processes for better parallelism)
        executor_class: Type = ProcessPoolExecutor
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
            combined_bigrams=combined_bigrams,
        )
        # Database is automatically flushed by context manager on exit

        # Optional post-ingestion compaction
        if compact_after_ingest:
            _perform_compaction(db, db_path)

    # Optional archiving: compress DB to archive directory
    if archive_path_stub is not None:
        _archive_database(db_path, archive_path_stub, ngram_size, repo_release_id, repo_corpus_id)

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

    # Return None to avoid Jupyter displaying the path
    # (the path is already printed in the summary)
    return None


def _perform_compaction(db, db_path: str) -> None:
    """
    Perform full compaction on the database using compact_all().

    Args:
        db: Open RocksDB handle
        db_path: Path to database (for logging)
    """
    from ngramprep.utilities.display import format_bytes, format_banner

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


def _archive_database(
    db_path: str,
    archive_stub: str,
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str
) -> None:
    """
    Archive (compress) the database to a structured directory using zstd.

    Creates archive path following the pattern:
    {archive_stub}/{release_id}/{corpus_id}/{n}gram_files/{n}grams.db.tar.zst

    The compression is done in a temp directory on /scratch first, then
    rsync'd to the final destination for better performance.

    Args:
        db_path: Path to the database directory
        archive_stub: Base directory for archives (will be expanded with structured path)
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")
    """
    import subprocess
    import tempfile
    from pathlib import Path
    from ngramprep.utilities.display import format_banner, truncate_path_to_fit

    logger.info("Starting database archival")
    print()
    print(format_banner("Database Archival"))

    # Build structured archive path (mirrors database structure)
    archive_base = Path(archive_stub)
    final_archive = archive_base / repo_release_id / repo_corpus_id / f"{ngram_size}gram_files" / f"{Path(db_path).name}.tar.zst"
    logger.info(f"Target archive location: {final_archive}")

    # Determine output archive filename
    db_name = Path(db_path).name

    try:
        # Create temp directory on /scratch for compression
        # Use $TMPDIR if set, otherwise try /scratch/edk202, then fall back to default
        temp_base = os.environ.get('TMPDIR') or '/scratch/edk202'
        temp_dir = tempfile.mkdtemp(dir=temp_base, prefix="ngram_archive_")
        temp_archive = Path(temp_dir) / f"{db_name}.tar.zst"
        logger.info(f"Temporary archive location: {temp_archive}")

        try:
            # Compress the database to temp location
            print(f"Compressing database to temporary location: {truncate_path_to_fit(str(temp_archive), 'Compressing database to temporary location: ')}")
            compress_db(db_path, output_path=temp_archive)
            logger.info(f"Database compressed successfully to {temp_archive}")

            # Transfer the archive to final destination
            print(f"Transferring archive to {truncate_path_to_fit(str(final_archive), 'Transferring archive to ')}")

            # Create parent directories for destination
            final_archive.parent.mkdir(parents=True, exist_ok=True)

            # Copy the archive to final destination
            import shutil as shutil_copy
            shutil_copy.copy2(str(temp_archive), str(final_archive))

            logger.info(f"Archive transferred successfully to {final_archive}")
            print(f"Archive created: {truncate_path_to_fit(str(final_archive), 'Archive created: ')}")

        finally:
            # Clean up temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {cleanup_err}")

    except Exception as e:
        logger.error(f"Archival failed: {e}")
        print(f"Archival failed: {e}")
        raise
