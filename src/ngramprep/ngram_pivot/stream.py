# ngram_pivot/stream.py
"""Streaming interface for year-indexed n-gram database."""
from typing import Iterator, Tuple, Optional
import logging

from ngramprep.common_db.api import open_db, prefix_scan, range_scan
from ngramprep.ngram_pivot.encoding import (
    decode_year_ngram_key,
    decode_year_stats,
    encode_year_ngram_key,
    year_prefix,
)

logger = logging.getLogger(__name__)


def stream_year_ngrams(
        db_path,
        year: int,
        ngram_filter: Optional[callable] = None,
) -> Iterator[Tuple[bytes, int, int]]:
    """
    Stream all n-grams for a specific year.

    This is optimized for training word2vec models - it sequentially
    reads all n-grams from a given year with minimal memory overhead.

    Args:
        db_path: Path to pivoted database
        year: Year to stream (e.g., 1958)
        ngram_filter: Optional filter function(ngram_bytes) -> bool

    Yields:
        Tuples of (ngram_bytes, occurrences, documents)

    Example:
        >>> for ngram, count, docs in stream_year_ngrams(db_path, 1958):
        ...     if count >= 100:  # Only frequent n-grams
        ...         process_for_word2vec(ngram)
    """
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        prefix = year_prefix(year)

        for key, value in prefix_scan(db, prefix):
            # Decode key and value
            year_decoded, ngram = decode_year_ngram_key(key)
            occurrences, documents = decode_year_stats(value)

            # Apply optional filter
            if ngram_filter is not None and not ngram_filter(ngram):
                continue

            yield ngram, occurrences, documents


def stream_year_range(
        db_path,
        start_year: int,
        end_year: int,
        ngram_filter: Optional[callable] = None,
) -> Iterator[Tuple[int, bytes, int, int]]:
    """
    Stream n-grams across a range of years.

    Args:
        db_path: Path to pivoted database
        start_year: Starting year (inclusive)
        end_year: Ending year (inclusive)
        ngram_filter: Optional filter function(ngram_bytes) -> bool

    Yields:
        Tuples of (year, ngram_bytes, occurrences, documents)
    """
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        lower = year_prefix(start_year)
        # Upper bound is first key of (end_year + 1)
        upper = year_prefix(end_year + 1)

        for key, value in range_scan(db, lower, upper):
            year, ngram = decode_year_ngram_key(key)
            occurrences, documents = decode_year_stats(value)

            # Apply optional filter
            if ngram_filter is not None and not ngram_filter(ngram):
                continue

            yield year, ngram, occurrences, documents


def get_ngram_stats(
        db_path,
        year: int,
        ngram: bytes,
) -> Optional[Tuple[int, int]]:
    """
    Get statistics for a specific n-gram in a specific year.

    Args:
        db_path: Path to pivoted database
        year: Year to query
        ngram: N-gram as bytes (e.g., b"the united states")

    Returns:
        Tuple of (occurrences, documents) or None if not found

    Example:
        >>> stats = get_ngram_stats(db_path, 1958, b"<UNK> united state <UNK> america")
        >>> if stats:
        ...     occurrences, documents = stats
        ...     print(f"Found in {documents} documents with {occurrences} occurrences")
    """
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        key = encode_year_ngram_key(year, ngram)
        value = db.get(key)

        if value is None:
            return None

        return decode_year_stats(value)