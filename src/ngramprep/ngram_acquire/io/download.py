"""HTTP download utilities with retry logic."""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

__all__ = ["stream_download_with_retries"]


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
    Download a URL with streaming and exponential backoff retry.
    
    Args:
        url: URL to download
        session: Optional requests.Session for connection pooling
        max_retries: Maximum number of download attempts
        delay_seconds: Initial retry delay in seconds
        backoff: Multiplier for exponential backoff (delay *= backoff)
        timeout: Request timeout in seconds
        
    Returns:
        Streaming requests.Response object (caller must close)
        
    Raises:
        requests.RequestException: After all retries exhausted
        
    Example:
        >>> resp = stream_download_with_retries(url)
        >>> try:
        ...     for chunk in resp.iter_content(chunk_size=8192):
        ...         process(chunk)
        ... finally:
        ...     resp.close()
    """
    sess = session or requests.Session()
    delay = delay_seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", url, attempt, max_retries)
            resp = sess.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            logger.debug("Download succeeded: %s", url)
            return resp
        except requests.RequestException as exc:
            if attempt == max_retries:
                logger.error(
                    "Failed to download %s after %d attempts: %s",
                    url, max_retries, exc
                )
                raise
            logger.warning(
                "Download failed (attempt %d/%d): %s - retrying in %.1fs",
                attempt, max_retries, exc, delay
            )
            time.sleep(delay)
            delay *= backoff
        except Exception:
            # Non-request exceptions (KeyboardInterrupt, etc.) should fail fast
            logger.exception("Unexpected error downloading %s", url)
            raise

    # Unreachable due to raise in loop, but helps type checkers
    raise RuntimeError(f"Failed to download {url}")