from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def stream_download_with_retries(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 2.0,
    timeout: float = 60.0,
) -> requests.Response:
    """
    GET a URL with stream=True, retrying on transient failures.

    Returns a requests.Response (caller must close it). Raises on failure.
    """
    sess = session or requests.Session()
    delay = delay_seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading %s (attempt %d)", url, attempt)
            resp = sess.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # broad: also works with fake sessions in tests
            if attempt == max_retries:
                logger.error("Failed to download %s after %d attempts: %s",
                             url, max_retries, exc)
                raise
            logger.warning("Download failed (%s); retrying in %.1fs...", exc, delay)
            time.sleep(delay)
            delay *= backoff
