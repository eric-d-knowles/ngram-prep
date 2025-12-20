"""Worker process for downloading and parsing ngram files."""
from __future__ import annotations

import gzip
import logging
import os
import struct
from contextlib import closing
from pathlib import PurePosixPath
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

import requests

from ngramprep.ngram_acquire.io.download import stream_download_with_retries
from ngramprep.ngram_acquire.io.parse import parse_line

if TYPE_CHECKING:
    from ngram_acquire.io.parse import NgramRecord

logger = logging.getLogger(__name__)

__all__ = ["process_and_ingest_file"]

try:
    import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None


def process_and_ingest_file(
    url: str,
    worker_id: int,
    filter_pred: Optional[Callable[[str], bool]] = None,
    log_file_path: Optional[str] = None,
    *,
    session: Optional[requests.Session] = None,
    combined_bigrams: Optional[set] = None,
) -> Tuple[str, Dict[str, bytes], int]:
    """
    Download, decompress, and parse a gzipped ngram file.

    Downloads the file from the given URL, decompresses it line-by-line,
    parses each line into structured data, and packs the results into
    compact binary format.

    Args:
        url: Download URL for the gzipped ngram file
        worker_id: Worker identifier for logging
        filter_pred: Optional predicate to filter ngrams by text
        log_file_path: Optional path to log file for worker output
        session: Optional requests.Session for connection pooling
        combined_bigrams: Optional set of bigrams to combine with hyphens

    Returns:
        Tuple of (status_message, parsed_data_dict, uncompressed_bytes)
        - status_message: Success or error message
        - parsed_data_dict: Mapping of ngram keys to packed binary values
        - uncompressed_bytes: Total bytes of uncompressed data processed

    Packing Format:
        Values are packed as little-endian uint64 triplets per year:
        (year, frequency, document_count) → struct format '<{3*N}Q'
    """
    # Set up worker-specific logging
    if log_file_path:
        worker_logger = logging.getLogger(f"worker_{os.getpid()}")

        # Only add handler if not already present
        if not worker_logger.handlers:
            try:
                file_handler = logging.FileHandler(log_file_path, mode="a")
                formatter = logging.Formatter(
                    "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler.setFormatter(formatter)
                worker_logger.addHandler(file_handler)
                worker_logger.setLevel(logging.INFO)
                worker_logger.propagate = False
            except Exception as e:
                # Fallback to standard logger if file logging fails
                logger.warning(
                    "Worker %s (PID %s): Could not set up file logging: %s",
                    worker_id, os.getpid(), e
                )
                worker_logger = logger
    else:
        worker_logger = logger

    # Set process title if available (helps with process monitoring)
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle(f"nga:worker[{worker_id:03d}]")
        except Exception:
            pass

    filename = PurePosixPath(url).name
    parsed_data: Dict[str, bytes] = {}
    uncompressed_bytes = 0
    pid = os.getpid()

    try:
        worker_logger.info(
            "Worker %s (PID %s): Processing %s",
            worker_id, pid, filename
        )

        # Download file with retry logic
        resp = stream_download_with_retries(url, session=session)

        with closing(resp):
            # Log compressed file size if available
            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    worker_logger.info(
                        "Worker %s (PID %s): File size: %s bytes (compressed)",
                        worker_id, pid, f"{int(content_length):,}"
                    )
                except ValueError:
                    worker_logger.debug(
                        "Worker %s (PID %s): Non-numeric content-length=%r",
                        worker_id, pid, content_length
                    )

            # Process gzipped content line by line
            lines_processed = 0
            with gzip.GzipFile(fileobj=resp.raw, mode="rb") as gz:
                for raw in gz:
                    # Track uncompressed bytes
                    uncompressed_bytes += len(raw)
                    lines_processed += 1

                    try:
                        # Parse line and pack if valid
                        key, rec = parse_line(
                            raw.decode("utf-8"),
                            filter_pred=filter_pred,
                            combined_bigrams=combined_bigrams
                        )
                        if key and rec:
                            parsed_data[key] = _pack_record(rec)

                    except UnicodeDecodeError as exc:
                        worker_logger.warning(
                            "Worker %s (PID %s): Unicode error in %s line %s: %s",
                            worker_id, pid, filename, lines_processed, exc
                        )
                    except Exception as exc:
                        worker_logger.warning(
                            "Worker %s (PID %s): Error processing line %s from %s: %s",
                            worker_id, pid, lines_processed, filename, exc
                        )

        # Success message
        msg = (
            f"SUCCESS: {filename} - {lines_processed:,} lines, "
            f"{len(parsed_data):,} entries, {uncompressed_bytes:,} uncompressed bytes"
        )
        worker_logger.info("Worker %s (PID %s): %s", worker_id, pid, msg)
        return msg, parsed_data, uncompressed_bytes

    except requests.Timeout:
        msg = f"TIMEOUT: {filename}"
        worker_logger.error(
            "Worker %s (PID %s): Timeout - %s",
            worker_id, pid, filename
        )
        return msg, {}, 0

    except requests.RequestException as exc:
        msg = f"NETWORK_ERROR: {filename}"
        worker_logger.error(
            "Worker %s (PID %s): Network error - %s (%s)",
            worker_id, pid, filename, exc
        )
        return msg, {}, 0

    except Exception as exc:
        msg = f"ERROR: {filename} - {exc}"
        worker_logger.error(
            "Worker %s (PID %s): Error - %s: %s",
            worker_id, pid, filename, exc
        )
        return msg, {}, 0


def _pack_record(rec: NgramRecord) -> bytes:
    """
    Pack frequency data into compact binary format.

    Converts year/frequency/document_count triplets into little-endian
    uint64 values for efficient storage.

    Args:
        rec: NgramRecord containing frequency data

    Returns:
        Packed binary data as bytes
    """
    freqs = rec.get("frequencies", [])
    if not freqs:
        return b""

    # Flatten triplets into single list (cache extend for performance)
    flat: list[int] = []
    extend = flat.extend
    for f in freqs:
        extend((
            int(f["year"]),
            int(f["frequency"]),
            int(f["document_count"])
        ))

    return struct.pack(f"<{len(flat)}Q", *flat)