"""Database path construction utilities."""
from __future__ import annotations

from pathlib import Path

__all__ = ["build_db_path"]


def build_db_path(
        db_path_stub: str,
        ngram_size: int,
        repo_release_id: str,
        repo_corpus_id: str,
) -> str:
    """
    Build a complete database path from a stub and ngram parameters.

    Creates a structured path following the pattern:
    {stub}/{release_id}/{corpus_id}/{n}gram_files/{n}grams.db

    Args:
        db_path_stub: Base directory path (e.g., "/data/ngrams/")
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")

    Returns:
        Complete database path string

    Examples:
        >>> build_db_path("/data/ngrams/", 1, "20200217", "eng")
        '/data/ngrams/20200217/eng/1gram_files/1grams.db'

        >>> build_db_path("/data/ngrams", 2, "20120701", "eng-us")
        '/data/ngrams/20120701/eng-us/2gram_files/2grams.db'
    """
    # Normalize stub to ensure no trailing slash issues
    stub = Path(db_path_stub)

    # Build hierarchical path
    db_path = stub / repo_release_id / repo_corpus_id / f"{ngram_size}gram_files" / f"{ngram_size}grams.db"

    return str(db_path)