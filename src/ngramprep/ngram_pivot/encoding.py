# ngram_pivot/encoding.py (updated year range)
"""Key encoding/decoding for year-indexed database."""
import struct
from typing import Tuple, List


def encode_year_ngram_key(year: int, ngram: bytes) -> bytes:
    """
    Encode a composite key: year (4 bytes, big-endian) + ngram.

    Using big-endian ensures lexicographic ordering matches numeric ordering.
    This allows efficient prefix scans by year.

    Args:
        year: Year value (e.g., 1958)
        ngram: N-gram as bytes (e.g., b"the cat")

    Returns:
        Composite key as bytes
    """
    if not isinstance(ngram, bytes):
        raise TypeError(f"ngram must be bytes, got {type(ngram).__name__}")
    if not (0 <= year <= 9999):  # Sanity check - allow full 4-digit range
        raise ValueError(f"Year {year} outside expected range [0, 9999]")

    # Pack year as 4-byte big-endian unsigned int + ngram
    return struct.pack('>I', year) + ngram


def decode_year_ngram_key(key: bytes) -> Tuple[int, bytes]:
    """
    Decode a composite key into (year, ngram).

    Args:
        key: Composite key as bytes

    Returns:
        Tuple of (year, ngram_bytes)
    """
    if len(key) < 4:
        raise ValueError(f"Key too short: {len(key)} bytes")

    year = struct.unpack('>I', key[:4])[0]
    ngram = key[4:]
    return year, ngram


def year_prefix(year: int) -> bytes:
    """
    Get the key prefix for all n-grams in a given year.

    Args:
        year: Year value

    Returns:
        4-byte prefix for that year
    """
    if not (0 <= year <= 9999):
        raise ValueError(f"Year {year} outside expected range [0, 9999]")

    return struct.pack('>I', year)


def encode_year_stats(occurrences: int, documents: int) -> bytes:
    """
    Encode year statistics as 16-byte packed value.

    Args:
        occurrences: Number of occurrences that year
        documents: Number of unique documents that year

    Returns:
        16-byte packed value (2 × uint64, little-endian)
    """
    return struct.pack('<QQ', occurrences, documents)


def decode_year_stats(value: bytes) -> Tuple[int, int]:
    """
    Decode year statistics from packed value.

    Args:
        value: Packed value (16 or 24 bytes)
               16 bytes: (occurrences, documents) - ngram format
               24 bytes: (year, occurrences, documents) - Davies format

    Returns:
        Tuple of (occurrences, documents)
    """
    if len(value) == 16:
        # Ngram format: (occurrences, documents)
        return struct.unpack('<QQ', value)
    elif len(value) == 24:
        # Davies format: (year, occurrences, documents) - skip year, return rest
        year, occurrences, documents = struct.unpack('<QQQ', value)
        return (occurrences, documents)
    else:
        raise ValueError(f"Expected 16 or 24 byte value, got {len(value)} bytes")


def decode_packed24_records(value: bytes) -> List[Tuple[int, int, int]]:
    """
    Decode Packed24Merge format into list of (year, occurrences, documents) tuples.

    Args:
        value: Packed24 format bytes (multiple of 24 bytes)

    Returns:
        List of (year, occurrences, documents) tuples
    """
    if len(value) % 24 != 0:
        raise ValueError(f"Value length {len(value)} not multiple of 24")

    records = []
    for i in range(0, len(value), 24):
        year, occurrences, documents = struct.unpack('<QQQ', value[i:i+24])
        records.append((year, occurrences, documents))

    return records