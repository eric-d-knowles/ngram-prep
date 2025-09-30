"""Fetch file URLs from repository listings."""
from __future__ import annotations

import logging
import re
import time
import warnings
from pathlib import PurePosixPath
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

logger = logging.getLogger(__name__)

__all__ = ["fetch_file_urls"]

# Suppress BeautifulSoup warning about parsing XML with HTML parser
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def fetch_file_urls(
        base_url: str,
        file_pattern: re.Pattern[str],
        timeout: tuple[int, int] = (10, 30),
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
) -> list[str]:
    """
    Fetch file URLs from a repository listing matching a pattern.

    Handles both HTML index pages and GCS XML bucket listings.
    Extracts links and filters by filename pattern.

    Args:
        base_url: Base URL of the directory, index page, or GCS bucket listing
        file_pattern: Compiled regex to match against filenames only
        timeout: (connect_timeout, read_timeout) in seconds
        max_retries: Maximum fetch attempts
        delay: Initial retry delay in seconds
        backoff: Multiplier for exponential backoff

    Returns:
        Sorted list of absolute URLs matching the pattern

    Raises:
        RuntimeError: After all retries exhausted
    """
    retry_delay = delay

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Fetching file listing from %s (attempt %d/%d)",
                base_url, attempt, max_retries
            )

            with requests.Session() as sess:
                resp = sess.get(base_url, timeout=timeout)
            resp.raise_for_status()

            # Handle encoding
            if not resp.encoding:
                resp.encoding = resp.apparent_encoding

            # Parse content - html.parser works for both HTML and XML
            # Note: html.parser lowercases XML tags, so <Key> becomes <key>
            soup = BeautifulSoup(resp.text, "html.parser")

            found: list[str] = []

            # Extract <a> tags (HTML) or <key> tags (GCS XML, lowercased by parser)
            for tag in soup.find_all(["a", "key"]):
                if tag.name == "a":
                    # HTML anchor tag
                    href = tag.get("href", "").strip()
                    if not href or href.startswith(("#", "mailto:", "javascript:", "?")):
                        continue
                    abs_url = urljoin(base_url, href)
                    filename = PurePosixPath(abs_url).name
                else:
                    # GCS XML <key> tag (lowercased from <Key> by html.parser)
                    key_text = tag.get_text(strip=True)
                    if not key_text:
                        continue
                    filename = PurePosixPath(key_text).name
                    # Reconstruct full URL: key is relative path like "ngrams/books/..."
                    abs_url = f"https://storage.googleapis.com/books/{key_text}"

                # Match pattern against filename only
                if file_pattern.match(filename):
                    found.append(abs_url)

            # Remove duplicates and sort by filename
            found = sorted(set(found), key=lambda u: PurePosixPath(u).name)

            logger.info("Found %d matching files", len(found))
            return found

        except requests.RequestException as exc:
            if attempt == max_retries:
                logger.error(
                    "Failed to fetch file listing after %d attempts: %s",
                    max_retries, exc
                )
                raise RuntimeError(
                    f"Failed to fetch file URLs from {base_url}"
                ) from exc

            logger.warning(
                "Fetch failed (attempt %d/%d): %s - retrying in %.1fs",
                attempt, max_retries, exc, retry_delay
            )
            time.sleep(retry_delay)
            retry_delay *= backoff

        except Exception:
            # Non-request exceptions should fail fast
            logger.exception("Unexpected error fetching URLs from %s", base_url)
            raise

    # Unreachable, but helps type checkers
    raise RuntimeError(f"Failed to fetch file URLs from {base_url}")