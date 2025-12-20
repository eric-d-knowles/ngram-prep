"""Coordination logic for file discovery, validation, and resume handling."""
from __future__ import annotations

import logging
import random
from pathlib import PurePosixPath
from typing import List, Tuple

import rocks_shim as rs

from ngramprep.ngram_acquire.io.locations import build_location_info
from ngramprep.ngram_acquire.io.fetch import fetch_file_urls
from ngramprep.ngram_acquire.db.metadata import is_file_processed

logger = logging.getLogger(__name__)

__all__ = [
    "discover_files",
    "select_file_subset",
    "filter_processed_files",
    "randomize_file_order",
]


def discover_files(
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str,
) -> Tuple[str, List[str]]:
    """
    Discover available ngram files from repository.

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")

    Returns:
        Tuple of (repo_url, list_of_file_urls)

    Raises:
        RuntimeError: If no files found in repository
    """
    page_url, file_rx = build_location_info(ngram_size, repo_release_id, repo_corpus_id)
    file_urls = fetch_file_urls(page_url, file_rx)

    if not file_urls:
        raise RuntimeError("No n-gram files found in the repository")

    logger.info("Discovered %d files from %s", len(file_urls), page_url)
    return page_url, file_urls


def select_file_subset(
    file_urls: List[str],
    file_range: Tuple[int, int] | None = None,
) -> Tuple[List[str], int, int]:
    """
    Select subset of files based on range.

    Args:
        file_urls: All available file URLs
        file_range: Optional (start_idx, end_idx) tuple. Use None for all files.

    Returns:
        Tuple of (selected_urls, start_idx, end_idx)

    Raises:
        ValueError: If file_range is invalid
    """
    # Handle "all files" case
    if file_range is None:
        start_idx, end_idx = 0, len(file_urls) - 1
        selected = file_urls
        logger.info("Selected all %d files", len(selected))
        return selected, start_idx, end_idx

    start_idx, end_idx = file_range

    if start_idx < 0 or end_idx >= len(file_urls) or start_idx > end_idx:
        raise ValueError(
            f"Invalid file range {file_range}. Available: 0..{len(file_urls) - 1}"
        )

    selected = file_urls[start_idx: end_idx + 1]
    logger.info("Selected files %d to %d (%d total)", start_idx, end_idx, len(selected))
    return selected, start_idx, end_idx


def filter_processed_files(
    file_urls: List[str],
    db: rs.DB,
) -> Tuple[List[str], int]:
    """
    Filter out already-processed files for resume mode.

    Args:
        file_urls: URLs to check
        db: RocksDB database handle

    Returns:
        Tuple of (unprocessed_urls, num_skipped)
    """
    to_keep = []
    for url in file_urls:
        filename = PurePosixPath(url).name
        if not is_file_processed(db, filename):
            to_keep.append(url)

    num_skipped = len(file_urls) - len(to_keep)

    if num_skipped:
        logger.info("Resume mode: skipping %d processed files", num_skipped)

    return to_keep, num_skipped


def randomize_file_order(
    file_urls: List[str],
    seed: int,
) -> None:
    """
    Randomize file processing order (in-place).

    Args:
        file_urls: List of file URLs to shuffle
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    random.shuffle(file_urls)
    logger.info("Randomized file order with seed %d", seed)
