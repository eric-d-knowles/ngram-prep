"""Database writing utilities for Davies acquisition."""
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from ngramprep.common_db.api import open_db
from .encoding import (
    encode_sentence_key,
    encode_sentence_key_with_genre,
    encode_occurrence_count,
    decode_occurrence_count,
)

logger = logging.getLogger(__name__)

__all__ = [
    "write_sentences_batch",
    "write_sentences_batch_with_genre",
    "SentenceBatchWriter",
    "SentenceBatchWriterWithGenre",
]

DEFAULT_BATCH_SIZE = 100_000


def write_sentences_batch(
    db,
    sentence_counts: Dict[Tuple[int, str], int],
    disable_wal: bool = True,
) -> int:
    """
    Write a batch of sentences with occurrence counts to the database.

    Args:
        db: Open RocksDB handle
        sentence_counts: Dict mapping (year, sentence_str) to occurrence count
        disable_wal: If True, disable write-ahead log for better performance

    Returns:
        Number of unique sentences written

    Example:
        >>> with open_db(db_path, mode="w") as db:
        ...     counts = {
        ...         (1950, "the cat sat"): 2,
        ...         (1951, "on the mat"): 1,
        ...     }
        ...     count = write_sentences_batch(db, counts)
        >>> count
        2
    """
    if not sentence_counts:
        return 0

    # Use context manager for batch writing with merge operator
    with db.write_batch(disable_wal=disable_wal, sync=False) as wb:
        for (year, sentence_str), count in sentence_counts.items():
            # Split sentence back into tokens for key encoding
            tokens = sentence_str.split()
            key = encode_sentence_key(year, tokens)

            # Use merge() to let RocksDB handle count accumulation
            # Store as 24-byte (year, occurrences, documents) for packed24 merge operator
            value = encode_occurrence_count(count, year)
            wb.merge(key, value)
    # Context manager auto-commits on exit

    return len(sentence_counts)


class SentenceBatchWriter:
    """
    Buffered writer for sentences with occurrence counting.

    Accumulates sentences in memory, counts duplicates, and writes them in batches
    for better performance.
    """

    def __init__(
        self,
        db,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the batch writer.

        Args:
            db: Open RocksDB handle
            batch_size: Number of unique sentences per batch
        """
        self.db = db
        self.batch_size = batch_size
        # Use dict to accumulate counts: (year, sentence_str) -> count
        self.buffer: Dict[Tuple[int, str], int] = defaultdict(int)
        self.total_written = 0

    def add(self, year: int, tokens: List[str]) -> None:
        """
        Add a sentence to the buffer, incrementing its count.

        Flushes automatically when buffer reaches batch_size unique sentences.

        Args:
            year: Year for this sentence
            tokens: List of word tokens
        """
        # Join tokens into string for deduplication
        sentence_str = ' '.join(tokens)
        self.buffer[(year, sentence_str)] += 1

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """
        Write buffered sentences with counts to database.

        Returns:
            Number of unique sentences written
        """
        if not self.buffer:
            return 0

        count = write_sentences_batch(self.db, self.buffer)
        self.total_written += count
        self.buffer.clear()

        return count

    def close(self) -> int:
        """
        Flush any remaining buffered sentences.

        Returns:
            Total number of unique sentences written
        """
        self.flush()
        return self.total_written


def write_sentences_batch_with_genre(
    db,
    sentence_data: Dict[Tuple[str, int, str], int],
    disable_wal: bool = True,
) -> int:
    """
    Write a batch of sentences with genre in the key to the database.

    Each sentence/genre combination gets its own key, allowing easy filtering.
    Uses the packed24 merge operator for accumulating counts.

    Args:
        db: Open RocksDB handle
        sentence_data: Dict mapping (genre, year, sentence_str) to occurrence count
        disable_wal: If True, disable write-ahead log for better performance

    Returns:
        Number of unique (genre, sentence) combinations written

    Example:
        >>> with open_db(db_path, mode="w") as db:
        ...     data = {
        ...         ('fic', 1950, "the cat sat"): 1,
        ...         ('mag', 1950, "the cat sat"): 1,
        ...         ('nf', 1951, "on the mat"): 1,
        ...     }
        ...     count = write_sentences_batch_with_genre(db, data)
        >>> count
        3
    """
    if not sentence_data:
        return 0

    with db.write_batch(disable_wal=disable_wal, sync=False) as wb:
        for (genre, year, sentence_str), count in sentence_data.items():
            # Split sentence back into tokens for key encoding
            tokens = sentence_str.split()
            key = encode_sentence_key_with_genre(genre, year, tokens)

            # Use merge operator to accumulate counts
            value = encode_occurrence_count(count, year)
            wb.merge(key, value)

    return len(sentence_data)


class SentenceBatchWriterWithGenre:
    """
    Buffered writer for sentences with genre in the key.

    Accumulates sentences in memory with (genre, year, sentence) as the key,
    and writes them in batches for better performance.
    """

    def __init__(
        self,
        db,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the batch writer with genre support.

        Args:
            db: Open RocksDB handle
            batch_size: Number of unique (genre, sentence) combinations per batch
        """
        self.db = db
        self.batch_size = batch_size
        # Use dict to accumulate counts: (genre, year, sentence_str) -> count
        self.buffer: Dict[Tuple[str, int, str], int] = defaultdict(int)
        self.total_written = 0

    def add(self, year: int, tokens: List[str], genre: Optional[str] = None) -> None:
        """
        Add a sentence to the buffer, incrementing its count for the given genre.

        Flushes automatically when buffer reaches batch_size unique combinations.

        Args:
            year: Year for this sentence
            tokens: List of word tokens
            genre: Genre code (e.g., 'fic', 'mag', 'nf')
        """
        # Join tokens into string for deduplication
        sentence_str = ' '.join(tokens)

        # Use 'unknown' if genre is None
        genre_key = genre if genre is not None else 'unknown'

        self.buffer[(genre_key, year, sentence_str)] += 1

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """
        Write buffered sentences with genre to database.

        Returns:
            Number of unique (genre, sentence) combinations written
        """
        if not self.buffer:
            return 0

        count = write_sentences_batch_with_genre(self.db, self.buffer)
        self.total_written += count
        self.buffer.clear()

        return count

    def close(self) -> int:
        """
        Flush any remaining buffered sentences.

        Returns:
            Total number of unique (genre, sentence) combinations written
        """
        self.flush()
        return self.total_written
