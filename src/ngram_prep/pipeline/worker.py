# ngram_prep/pipeline/worker.py
from __future__ import annotations

import gzip
import logging
import os
import struct
from contextlib import closing
from pathlib import PurePosixPath
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

import requests

from ngram_prep.io.download import stream_download_with_retries
from ngram_prep.io.parse import parse_line
if TYPE_CHECKING:
    from ngram_prep.io.parse import NgramRecord

logger = logging.getLogger(__name__)

try:
    import setproctitle as _setproctitle  # optional
except Exception:  # pragma: no cover
    _setproctitle = None


def process_and_ingest_file(
    url: str,
    worker_id: int,
    filter_pred: Optional[Callable[[str], bool]] = None,
    *,
    session: Optional[requests.Session] = None,
) -> Tuple[str, Dict[str, bytes]]:
    """
    Download a gzip n-gram file, parse line-by-line, and pack values to bytes.

    Returns
    -------
    (status_message, {ngram_key: packed_bytes})

    Packing format
    --------------
    Values packed as little-endian uint64 triplets per year:
      (year, frequency, document_count) → struct fmt '<{3*N}Q'
    """
    if _setproctitle is not None:  # harmless if library is absent
        try:
            _setproctitle.setproctitle("PROC_WORKER")
        except Exception:  # pragma: no cover
            pass

    filename = PurePosixPath(url).name
    parsed_data: Dict[str, bytes] = {}
    pid = os.getpid()

    try:
        logger.info("Worker %s: Processing %s", worker_id, filename)
        resp = stream_download_with_retries(url, session=session)

        with closing(resp):
            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    logger.info(
                        "Worker %s: File size: %s bytes",
                        worker_id,
                        f"{int(content_length):,}",
                    )
                except ValueError:
                    logger.debug(
                        "Worker %s: Non-numeric content-length=%r",
                        worker_id,
                        content_length,
                    )

            lines_processed = 0
            with gzip.GzipFile(fileobj=resp.raw, mode="rb") as gz:
                for raw in gz:
                    lines_processed += 1
                    try:
                        key, rec = parse_line(
                            raw.decode("utf-8"), filter_pred=filter_pred
                        )
                        if key and rec:
                            parsed_data[key] = _pack_record(rec)
                    except UnicodeDecodeError as exc:
                        logger.warning(
                            "Worker %s (PID %s): Unicode error in %s line %s: %s",
                            worker_id,
                            pid,
                            filename,
                            lines_processed,
                            exc,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Worker %s (PID %s): Error processing line %s from %s: %s",
                            worker_id,
                            pid,
                            lines_processed,
                            filename,
                            exc,
                        )

        msg = (
            f"SUCCESS: {filename} - {lines_processed:,} lines, "
            f"{len(parsed_data):,} entries"
        )
        logger.info("Worker %s: %s", worker_id, msg)
        return msg, parsed_data

    except requests.Timeout:
        msg = f"TIMEOUT: {filename}"
        logger.error("Worker %s: Timeout - %s", worker_id, filename)
        return msg, {}
    except requests.RequestException as exc:
        msg = f"NETWORK_ERROR: {filename}"
        logger.error(
            "Worker %s: Network error - %s (%s)", worker_id, filename, exc
        )
        return msg, {}
    except Exception as exc:
        msg = f"ERROR: {filename} - {exc}"
        logger.error("Worker %s: Error - %s: %s", worker_id, filename, exc)
        return msg, {}


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
