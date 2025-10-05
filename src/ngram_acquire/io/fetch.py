"""Fetch file URLs from repository listings."""
from __future__ import annotations

import logging
import re
import time
import warnings
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

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

    Handles both HTML index pages and GCS XML bucket listings with pagination.
    Extracts links and filters by filename pattern.

    Args:
        base_url: Base URL of the directory, index page, or GCS bucket listing
        file_pattern: Compiled regex to match against filenames only
        timeout: (connect_timeout, read_timeout) in seconds
        max_retries: Maximum fetch attempts per page
        delay: Initial retry delay in seconds
        backoff: Multiplier for exponential backoff

    Returns:
        Sorted list of absolute URLs matching the pattern

    Raises:
        RuntimeError: After all retries exhausted
    """
    all_found: list[str] = []
    marker = None
    page_num = 0

    while True:
        page_num += 1
        retry_delay = delay

        # Build URL with marker for pagination
        fetch_url = base_url
        if marker:
            parsed = urlparse(base_url)
            params = parse_qs(parsed.query)
            params['marker'] = [marker]
            new_query = urlencode(params, doseq=True)
            fetch_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Fetching file listing page %d from %s (attempt %d/%d)",
                    page_num, base_url, attempt, max_retries
                )

                with requests.Session() as sess:
                    resp = sess.get(fetch_url, timeout=timeout)
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

                all_found.extend(found)
                logger.info("Page %d: Found %d matching files (total: %d)",
                           page_num, len(found), len(all_found))

                # Check for pagination (GCS XML format)
                is_truncated = soup.find("istruncated")
                if is_truncated and is_truncated.get_text(strip=True).lower() == "true":
                    # Find the next marker
                    next_marker_tag = soup.find("nextmarker")
                    if next_marker_tag:
                        marker = next_marker_tag.get_text(strip=True)
                        logger.info("Results truncated, continuing with marker: %s", marker)
                        break  # Break retry loop, continue to next page
                    else:
                        logger.warning("IsTruncated=true but no NextMarker found")
                        # Try to get last key as marker
                        all_keys = soup.find_all("key")
                        if all_keys:
                            marker = all_keys[-1].get_text(strip=True)
                            logger.info("Using last key as marker: %s", marker)
                            break
                        else:
                            # No more pages can be fetched
                            marker = None
                            break
                else:
                    # No more pages
                    marker = None
                    break

            except requests.RequestException as exc:
                if attempt == max_retries:
                    logger.error(
                        "Failed to fetch file listing page %d after %d attempts: %s",
                        page_num, max_retries, exc
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

        # If marker is None, we're done with all pages
        if marker is None:
            break

    # Remove duplicates and sort by filename
    all_found = sorted(set(all_found), key=lambda u: PurePosixPath(u).name)

    logger.info("Fetched %d total matching files across %d pages", len(all_found), page_num)
    return all_found