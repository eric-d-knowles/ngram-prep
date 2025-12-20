"""URL and pattern construction for Google Ngrams repository."""
from __future__ import annotations

import re
from typing import Tuple

BASE_URL = "https://storage.googleapis.com/books/ngrams/books"

__all__ = ["BASE_URL", "build_location_info"]


def build_location_info(
        ngram_size: int,
        repo_release_id: str,
        repo_corpus_id: str,
) -> Tuple[str, re.Pattern[str]]:
    """
    Build the listing URL and filename pattern for Google Ngrams files.

    Returns a URL that can be fetched (either HTML index or GCS XML listing)
    and a regex pattern that matches against filenames.

    Handles three release versions:
    - V3 (20200217): GCS XML API with prefix for "{n}-{shard}-of-{total}.gz"
    - V2 (20120701): HTML index with "googlebooks-{corpus}-all-{n}gram-{date}-{letter}.gz"
    - V1 (20090715): HTML index with "googlebooks-{corpus}-all-{n}gram-{date}-{num}.csv.zip"

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us", "fre")

    Returns:
        Tuple of (listing_url, filename_pattern)
        - listing_url: URL to fetch for file discovery
        - filename_pattern: Regex that matches just the filename (not full URL)

    Raises:
        ValueError: If parameters are invalid or release version unknown

    Examples:
        >>> url, pattern = build_location_info(1, "20200217", "eng")
        >>> url
        'https://books.storage.googleapis.com/?prefix=ngrams/books/20200217/eng/1-'
        >>> pattern.match("1-00012-of-00024.gz")
        <re.Match object...>
    """
    # Validate ngram_size
    if ngram_size not in (1, 2, 3, 4, 5):
        raise ValueError(f"ngram_size must be 1-5, got {ngram_size}")

    # Validate release ID format
    if not re.fullmatch(r"\d{8}", repo_release_id):
        raise ValueError(
            f"repo_release_id must be 8-digit YYYYMMDD, got {repo_release_id!r}"
        )

    # Validate corpus ID format
    if not re.fullmatch(r"[A-Za-z0-9-]+", repo_corpus_id):
        raise ValueError(
            f"repo_corpus_id must contain only [A-Za-z0-9-], got {repo_corpus_id!r}"
        )

    # Build URL and pattern based on release version
    if repo_release_id == "20200217":
        # V3: GCS bucket XML API listing
        # Files named: "1-00012-of-00024.gz"
        prefix = f"ngrams/books/{repo_release_id}/{repo_corpus_id}/{ngram_size}-"
        listing_url = f"https://books.storage.googleapis.com/?prefix={prefix}"
        filename_pattern = re.compile(rf"^{ngram_size}-\d{{5}}-of-\d{{5}}\.gz$")

    elif repo_release_id == "20120701":
        # V2: HTML index page (datasetsv3.html contains all versions)
        # Files named: "googlebooks-eng-all-2gram-20120701-qu.gz"
        listing_url = f"{BASE_URL}/datasetsv3.html"
        filename_pattern = re.compile(
            rf"^googlebooks-{re.escape(repo_corpus_id)}-all-{ngram_size}gram-{repo_release_id}-\w+\.gz$"
        )

    elif repo_release_id == "20090715":
        # V1: HTML index page (datasetsv3.html contains all versions)
        # Files named: "googlebooks-eng-all-5gram-20090715-296.csv.zip"
        listing_url = f"{BASE_URL}/datasetsv3.html"
        filename_pattern = re.compile(
            rf"^googlebooks-{re.escape(repo_corpus_id)}-all-{ngram_size}gram-{repo_release_id}-\d+\.csv\.zip$"
        )

    else:
        raise ValueError(
            f"Unknown release version {repo_release_id}. "
            f"Supported: 20200217 (V3), 20120701 (V2), 20090715 (V1)"
        )

    return listing_url, filename_pattern