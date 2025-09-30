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

from ngram_acquire.pipeline.worker import process_and_ingest_file
from ngram_acquire.db.metadata import mark_file_as_processed
from ngram_acquire.db.write import write_batch_to_db

logger = logging.getLogger(__name__)

__all__ = ["process_files", "DEFAULT_WRITE_BATCH_SIZE", "DEFAULT_WRITE_BATCH_BYTES"]

DEFAULT_WRITE_BATCH_SIZE = 50_000  # entries (keys)
DEFAULT_WRITE_BATCH_BYTES = 64 * (1 << 20)  # 64 MiB (keys+values)


def process_files(
        urls: Iterable[str],
        executor_class: Type,
        workers: int,
        db: rs.DB,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
        write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
        write_batch_bytes: int = DEFAULT_WRITE_BATCH_BYTES,
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

    Returns:
        Tuple of (success_msgs, failure_msgs, total_entries_written, 
                  write_batches, total_uncompressed_bytes)
    """
    # Get log file path for worker processes
    log_file_path = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            log_file_path = handler.baseFilename
            break

    if log_file_path:
        logger.info("Log file path for workers: %s", log_file_path)

    # Result tracking
    success_msgs: List[str] = []
    failure_msgs: List[str] = []
    total_entries_written = 0
    write_batches = 0
    total_uncompressed_bytes = 0

    # Batch accumulation
    pending_data: Dict[str, bytes] = {}
    pending_files: List[str] = []
    pending_bytes = 0

    def _approx_kv_bytes(d: Dict[str, bytes]) -> int:
        """Estimate total bytes for key-value pairs."""
        return sum(len(k.encode("utf-8")) + len(v) for k, v in d.items())

    def flush() -> None:
        """Flush pending batch to database and mark files as processed."""
        nonlocal total_entries_written, write_batches, pending_bytes

        if not pending_data:
            return

        try:
            entries_written = write_batch_to_db(db, pending_data)
            total_entries_written += entries_written
            write_batches += 1

            # Mark all files in batch as processed
            for fname in pending_files:
                mark_file_as_processed(db, fname)

            logger.info(
                "Flushed batch: %d entries, %d files",
                entries_written, len(pending_files)
            )
        except Exception as exc:
            msg = f"DB_WRITE_ERROR: {exc}"
            failure_msgs.extend(f"{msg} ({fn})" for fn in pending_files)
            logger.error("%s; aborting pipeline to avoid data loss", msg)
            raise
        else:
            pending_data.clear()
            pending_files.clear()
            pending_bytes = 0

    # Determine total for progress bar
    total = len(urls) if hasattr(urls, "__len__") else None

    with tqdm(total=total, desc="Processing Files", unit="files", colour="blue") as pbar:
        # Configure executor
        kwargs = {"max_workers": workers}
        if issubclass(executor_class, ProcessPoolExecutor):
            kwargs["mp_context"] = mp.get_context("spawn")
            logger.info(
                "Using multiprocessing start method: %s",
                kwargs["mp_context"].get_start_method()
            )

        with executor_class(**kwargs) as executor:
            it = iter(urls)
            futures: Dict[object, str] = {}
            max_in_flight = max(1, workers * 2)
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

                            # Accumulate data for batching
                            pending_data.update(parsed_data)
                            pending_files.append(filename)
                            pending_bytes += _approx_kv_bytes(parsed_data)

                            # Flush if batch thresholds exceeded
                            if (
                                    len(pending_data) >= write_batch_size
                                    or pending_bytes >= write_batch_bytes
                            ):
                                flush()

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
    flush()

    return (
        success_msgs,
        failure_msgs,
        total_entries_written,
        write_batches,
        total_uncompressed_bytes,
    )