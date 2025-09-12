# ngram_acquire/pipeline/worker.py
from __future__ import annotations

import gzip
import logging
import os
import struct
from contextlib import closing
from pathlib import PurePosixPath
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

import requests

from src.ngram_acquire.io.download import stream_download_with_retries
from src.ngram_acquire.io.parse import parse_line

if TYPE_CHECKING:
    from src.ngram_acquire.io.parse import NgramRecord

logger = logging.getLogger(__name__)

try:
    import setproctitle as _setproctitle  # optional
except Exception:  # pragma: no cover
    _setproctitle = None


def process_and_ingest_file(
        url: str,
        worker_id: int,
        filter_pred: Optional[Callable[[str], bool]] = None,
        log_file_path: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
) -> Tuple[str, Dict[str, bytes], int]:
    """
    Download a gzip n-gram file, parse line-by-line, and pack values to bytes.

    Returns
    -------
    (status_message, {ngram_key: packed_bytes}, uncompressed_bytes)

    Packing format
    --------------
    Values packed as little-endian uint64 triplets per year:
      (year, frequency, document_count) → struct fmt '<{3*N}Q'
    """

    # Set up worker logging if log file path provided
    if log_file_path:
        worker_logger = logging.getLogger(f"worker_{os.getpid()}")

        # Only add handler if not already present
        if not worker_logger.handlers:
            try:
                file_handler = logging.FileHandler(log_file_path, mode='a')
                formatter = logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(formatter)
                worker_logger.addHandler(file_handler)
                worker_logger.setLevel(logging.INFO)
                worker_logger.propagate = False  # Don't duplicate to root logger
            except Exception as e:
                # Fallback to standard logger if file logging fails
                logger.warning(f"Worker {os.getpid()}: Could not set up file logging: {e}")
                worker_logger = logger
    else:
        # Use standard logger if no log file path provided
        worker_logger = logger

    if _setproctitle is not None:  # harmless if library is absent
        try:
            _setproctitle.setproctitle("PROC_WORKER")
        except Exception:  # pragma: no cover
            pass

    filename = PurePosixPath(url).name
    parsed_data: Dict[str, bytes] = {}
    uncompressed_bytes = 0  # Track total uncompressed bytes processed
    pid = os.getpid()

    try:
        worker_logger.info("Worker %s (PID %s): Processing %s", worker_id, pid, filename)
        resp = stream_download_with_retries(url, session=session)

        with closing(resp):
            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    worker_logger.info(
                        "Worker %s (PID %s): File size: %s bytes (compressed)",
                        worker_id,
                        pid,
                        f"{int(content_length):,}",
                    )
                except ValueError:
                    worker_logger.debug(
                        "Worker %s (PID %s): Non-numeric content-length=%r",
                        worker_id,
                        pid,
                        content_length,
                    )

            lines_processed = 0
            with gzip.GzipFile(fileobj=resp.raw, mode="rb") as gz:
                for raw in gz:
                    # Track uncompressed bytes (length of each decompressed line)
                    uncompressed_bytes += len(raw)
                    lines_processed += 1

                    try:
                        key, rec = parse_line(
                            raw.decode("utf-8"), filter_pred=filter_pred
                        )
                        if key and rec:
                            parsed_data[key] = _pack_record(rec)
                    except UnicodeDecodeError as exc:
                        worker_logger.warning(
                            "Worker %s (PID %s): Unicode error in %s line %s: %s",
                            worker_id,
                            pid,
                            filename,
                            lines_processed,
                            exc,
                        )
                    except Exception as exc:
                        worker_logger.warning(
                            "Worker %s (PID %s): Error processing line %s from %s: %s",
                            worker_id,
                            pid,
                            lines_processed,
                            filename,
                            exc,
                        )

        # Enhanced success message with byte counts
        msg = (
            f"SUCCESS: {filename} - {lines_processed:,} lines, "
            f"{len(parsed_data):,} entries, {uncompressed_bytes:,} uncompressed bytes"
        )
        worker_logger.info("Worker %s (PID %s): %s", worker_id, pid, msg)
        return msg, parsed_data, uncompressed_bytes

    except requests.Timeout:
        msg = f"TIMEOUT: {filename}"
        worker_logger.error("Worker %s (PID %s): Timeout - %s", worker_id, pid, filename)
        return msg, {}, 0
    except requests.RequestException as exc:
        msg = f"NETWORK_ERROR: {filename}"
        worker_logger.error(
            "Worker %s (PID %s): Network error - %s (%s)", worker_id, pid, filename, exc
        )
        return msg, {}, 0
    except Exception as exc:
        msg = f"ERROR: {filename} - {exc}"
        worker_logger.error("Worker %s (PID %s): Error - %s: %s", worker_id, pid, filename, exc)
        return msg, {}, 0


def _pack_record(rec: NgramRecord) -> bytes:
    """Pack (year, frequency, document_count) as <uint64> triplets."""
    freqs = rec.get("frequencies", [])
    if not freqs:
        return b""
    flat: list[int] = []
    extend = flat.extend
    for f in freqs:
        extend((int(f["year"]), int(f["frequency"]), int(f["document_count"])))
    return struct.pack(f"<{len(flat)}Q", *flat)