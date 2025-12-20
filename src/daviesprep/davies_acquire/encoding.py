"""Encoding utilities for Davies corpus database format.

Davies corpora are stored in the same pivoted format as ngrams:
    Key: (genre, year, sentence_tokens) or (year, sentence_tokens) for backward compat
    Value: occurrence counts (using packed24 format)

This allows the same training code to work for both corpora.
"""
from __future__ import annotations

from typing import List, Optional
import struct

# Import encoding utilities from ngramprep
from ngramprep.ngram_pivot.encoding import (
    encode_year_ngram_key,
    decode_year_ngram_key,
    encode_year_stats,
    decode_year_stats,
)

__all__ = [
    "encode_sentence_key",
    "decode_sentence_key",
    "encode_sentence_key_with_genre",
    "decode_sentence_key_with_genre",
    "encode_occurrence_count",
    "decode_occurrence_count",
]


def encode_occurrence_count(count: int, year: int = 0) -> bytes:
    """
    Encode occurrence count as 24-byte value for packed24 merge operator.

    The packed24 merge operator expects 24-byte records: (year, occurrences, documents).
    We store (year, occurrences, 1) where year is redundant with the key but needed for merge.

    Args:
        count: Number of occurrences
        year: Year value (optional, defaults to 0)

    Returns:
        24-byte packed value
    """
    import struct
    return struct.pack('<QQQ', year, count, 1)


def decode_occurrence_count(value: bytes) -> int:
    """
    Decode occurrence count from packed value.

    Handles both 16-byte (old format) and 24-byte (new format with year) values.

    Args:
        value: Packed value (16 or 24 bytes)

    Returns:
        Number of occurrences
    """
    import struct
    if len(value) == 24:
        # New format: (year, occurrences, documents)
        year, occurrences, documents = struct.unpack('<QQQ', value)
        return occurrences
    elif len(value) == 16:
        # Old format: (occurrences, documents)
        occurrences, documents = decode_year_stats(value)
        return occurrences
    else:
        raise ValueError(f"Unexpected value length: {len(value)} bytes (expected 16 or 24)")


def encode_sentence_key(year: int, tokens: List[str]) -> bytes:
    """
    Encode a sentence into a database key.

    Uses the same format as pivoted ngrams: (year, tokens).

    Args:
        year: Year for this sentence
        tokens: List of word tokens

    Returns:
        Encoded key bytes

    Example:
        >>> tokens = ["the", "cat", "sat"]
        >>> key = encode_sentence_key(1950, tokens)
        >>> isinstance(key, bytes)
        True
    """
    # Join tokens with spaces to create the "ngram"
    sentence_bytes = ' '.join(tokens).encode('utf-8')

    # Use ngram encoding from ngramprep
    return encode_year_ngram_key(year, sentence_bytes)


def decode_sentence_key(key: bytes) -> tuple[int, List[str]]:
    """
    Decode a database key into year and tokens.

    Args:
        key: Encoded key bytes

    Returns:
        Tuple of (year, tokens)

    Example:
        >>> key = encode_sentence_key(1950, ["the", "cat", "sat"])
        >>> year, tokens = decode_sentence_key(key)
        >>> year
        1950
        >>> tokens
        ['the', 'cat', 'sat']
    """
    # Decode using ngram encoding
    year, sentence_bytes = decode_year_ngram_key(key)

    # Split back into tokens
    tokens = sentence_bytes.decode('utf-8').split()

    return year, tokens


def encode_sentence_key_with_genre(genre: str, year: int, tokens: List[str]) -> bytes:
    """
    Encode a sentence with genre prefix into a database key.

    Key format: [genre_code (4 bytes)][year (8 bytes)][sentence_bytes]
    Genre code is stored as a 4-byte fixed-width field (padded with nulls if needed).

    Args:
        genre: Genre code (e.g., 'fic', 'mag', 'nf')
        year: Year for this sentence
        tokens: List of word tokens

    Returns:
        Encoded key bytes

    Example:
        >>> tokens = ["the", "cat", "sat"]
        >>> key = encode_sentence_key_with_genre("fic", 1950, tokens)
        >>> isinstance(key, bytes)
        True
    """
    # Encode genre as 4-byte fixed-width field
    genre_bytes = genre.encode('utf-8')[:4].ljust(4, b'\x00')

    # Encode year as 8-byte little-endian unsigned long
    year_bytes = struct.pack('<Q', year)

    # Encode sentence
    sentence_bytes = ' '.join(tokens).encode('utf-8')

    return genre_bytes + year_bytes + sentence_bytes


def decode_sentence_key_with_genre(key: bytes) -> tuple[str, int, List[str]]:
    """
    Decode a genre-aware database key into genre, year, and tokens.

    Args:
        key: Encoded key bytes

    Returns:
        Tuple of (genre, year, tokens)

    Example:
        >>> key = encode_sentence_key_with_genre("fic", 1950, ["the", "cat", "sat"])
        >>> genre, year, tokens = decode_sentence_key_with_genre(key)
        >>> genre
        'fic'
        >>> year
        1950
        >>> tokens
        ['the', 'cat', 'sat']
    """
    # Extract genre (first 4 bytes)
    genre_bytes = key[:4]
    genre = genre_bytes.rstrip(b'\x00').decode('utf-8')

    # Extract year (next 8 bytes)
    year = struct.unpack('<Q', key[4:12])[0]

    # Extract sentence
    sentence_bytes = key[12:]
    tokens = sentence_bytes.decode('utf-8').split()

    return genre, year, tokens
