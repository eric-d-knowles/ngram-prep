from __future__ import annotations

import logging
import re
import time
from bs4 import BeautifulSoup
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit, urldefrag

import requests

logger = logging.getLogger(__name__)


def fetch_file_urls(
    ngram_repo_url: str,
    file_pattern: re.Pattern[str],
    timeout: tuple[int, int] = (10, 30),
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> list[str]:
    """
    Fetch absolute file URLs from an HTML index whose *filenames* match `file_pattern`.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching URLs from %s (attempt %d)", ngram_repo_url, attempt)

            with requests.Session() as sess:
                resp = sess.get(ngram_repo_url, timeout=timeout)
            resp.raise_for_status()

            if not resp.encoding:
                resp.encoding = resp.apparent_encoding

            base_url = resp.url
            soup = BeautifulSoup(resp.text, "html.parser")

            found: list[str] = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#", "mailto:", "javascript:")):
                    continue

                abs_url = urljoin(base_url, href)
                if file_pattern.search(abs_url):
                    found.append(abs_url)

            found.sort(key=lambda u: PurePosixPath(u).name)
            logger.info("Found %d files", len(found))
            return found

        except Exception as exc:
            if attempt == max_retries:
                logger.error("Failed after %d attempts: %s", max_retries, exc)
                raise RuntimeError("Failed to fetch file URLs") from exc

            logger.warning("Fetch failed (%s); retrying in %.1fs...", exc, delay)
            time.sleep(delay)
            delay *= backoff
