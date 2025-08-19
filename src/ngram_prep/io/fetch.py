from __future__ import annotations

import logging
import re
import time
from pathlib import PurePosixPath
from typing import Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


def fetch_file_urls(
    ngram_repo_url: str,
    file_pattern: re.Pattern[str],
    *,
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 2.0,
    timeout: float = 30.0,
    session: Optional[requests.Session] = None,
) -> list[str]:
    """
    Fetch downloadable file URLs from an export page, with simple retry/backoff.
    """
    sess = session or requests.Session()
    delay = delay_seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching URLs from %s (attempt %d)", ngram_repo_url, attempt)
            resp = sess.get(ngram_repo_url, timeout=timeout)
            resp.raise_for_status()

            hrefs = re.findall(r'href="([^"]+)"', resp.text)
            found: list[str] = []
            for href in hrefs:
                name = PurePosixPath(href).name
                if file_pattern.fullmatch(name):
                    found.append(urljoin(ngram_repo_url, href))

            deduped = list(dict.fromkeys(found))
            deduped.sort(key=lambda u: PurePosixPath(u).name)

            logger.info("Found %d files", len(deduped))
            return deduped

        except Exception as exc:  # broadened to catch custom session errors in tests
            if attempt == max_retries:
                logger.error(
                    "Failed to fetch file URLs after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise RuntimeError("Failed to fetch file URLs") from exc

            logger.warning("Fetch failed (%s); retrying in %.1fs...", exc, delay)
            time.sleep(delay)
            delay *= backoff
