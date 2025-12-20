"""Concurrent file processing and ingestion for ngram pipeline."""
from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

from tqdm import tqdm
import rocks_shim as rs

from ngramprep.ngram_acquire.worker import process_and_ingest_file
from ngramprep.ngram_acquire.batch_writer import BatchWriter

logger = logging.getLogger(__name__)

__all__ = ["process_files"]


def process_files(
        urls: Iterable[str],
        executor_class: Type,
        workers: int,
        db: rs.DB,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
        write_batch_size: int = 50_000,
        write_batch_bytes: int = 64 * (1 << 20),
        combined_bigrams: Optional[set] = None,
) -> Tuple[List[str], List[str], int, int, int]:
    """
    Process files concurrently and ingest results into RocksDB.

    Downloads, parses, and filters ngram files in parallel, batching writes
    to the database for efficiency. Tracks progress with tqdm.

    Args:
        urls: File URLs to process
        executor_class: ThreadPoolExecutor or ProcessPoolExecutor
        workers: Number of concurrent workers
        db: RocksDB database handle
        filter_pred: Optional predicate to filter ngrams by text
        write_batch_size: Max entries per batch before flush
        write_batch_bytes: Max bytes per batch before flush
        combined_bigrams: Optional set of bigrams to combine with hyphens

    Returns:
        Tuple of (success_msgs, failure_msgs, total_entries_written,
                  write_batches, total_uncompressed_bytes)
    """
    # Get log file path for worker processes
    log_file_path = _get_log_file_path()
    if log_file_path:
        logger.info("Log file path for workers: %s", log_file_path)

    # Result tracking
    success_msgs: List[str] = []
    failure_msgs: List[str] = []
    total_uncompressed_bytes = 0

    # Initialize batch writer
    batch_writer = BatchWriter(db, write_batch_size, write_batch_bytes)

    # Determine total for progress bar
    total = len(urls) if hasattr(urls, "__len__") else None

    with tqdm(
        total=total,
        desc="Files Processed:",
        unit="files",
        ncols=100,
        bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) as pbar:
        # Configure executor
        kwargs = {"max_workers": workers}
        if issubclass(executor_class, ProcessPoolExecutor):
            # Use fork to allow DB handle sharing across processes
            # (spawn would require pickling the C++ DB object, which fails)
            kwargs["mp_context"] = mp.get_context("fork")
            logger.info(
                "Using multiprocessing start method: %s",
                kwargs["mp_context"].get_start_method()
            )

        with executor_class(**kwargs) as executor:
            it = iter(urls)
            futures: Dict[object, str] = {}
            max_in_flight = max(1, workers * 2)  # Keep 2x workers worth of tasks queued
            idx = 0

            def submit_next(n: int = 1) -> None:
                """Submit next n tasks to executor."""
                nonlocal idx
                for _ in range(n):
                    try:
                        url = next(it)
                    except StopIteration:
                        return
                    idx += 1
                    fut = executor.submit(
                        process_and_ingest_file,
                        url,
                        idx,
                        filter_pred,
                        log_file_path,
                        combined_bigrams=combined_bigrams,
                    )
                    futures[fut] = url

            # Start initial batch of tasks
            submit_next(max_in_flight)

            # Process completed tasks as they finish
            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for fut in done:
                    url = futures.pop(fut)
                    filename = PurePosixPath(url).name

                    try:
                        # Unpack result: (status_msg, parsed_data, uncompressed_bytes)
                        result_msg, parsed_data, uncompressed_bytes = fut.result()

                        if result_msg.startswith("SUCCESS"):
                            success_msgs.append(result_msg)
                            total_uncompressed_bytes += uncompressed_bytes

                            # Add to batch writer
                            should_flush = batch_writer.add(filename, parsed_data)
                            if should_flush:
                                batch_writer.flush()

                            logger.info("Processed: %s", filename)
                        else:
                            failure_msgs.append(result_msg)

                    except Exception as exc:
                        msg = f"ERROR: {filename} - {exc}"
                        failure_msgs.append(msg)
                        logger.error(msg)
                    finally:
                        pbar.update(1)

                # Submit new tasks to replace completed ones
                submit_next(len(done))

    # Flush any remaining data
    batch_writer.flush()

    # Get final statistics
    total_entries_written, write_batches = batch_writer.get_stats()

    return (
        success_msgs,
        failure_msgs,
        total_entries_written,
        write_batches,
        total_uncompressed_bytes,
    )


def _get_log_file_path() -> Optional[str]:
    """Extract log file path from root logger's handlers."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            return handler.baseFilename
    return None
