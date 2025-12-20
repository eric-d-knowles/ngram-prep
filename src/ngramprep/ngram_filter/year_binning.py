"""Year binning utilities for aggregating temporal ngram data."""
from __future__ import annotations

import struct
from typing import List, Tuple

__all__ = ["bin_year_records", "get_bin_start"]

# Packed24 format: (year, occurrences, documents) as 3x uint64 little-endian
FMT = "<QQQ"
RECORD_SIZE = struct.calcsize(FMT)


def get_bin_start(year: int, bin_size: int) -> int:
    """
    Get the start year for a bin containing the given year.

    Args:
        year: Year to bin
        bin_size: Size of bins (1 = annual, 5 = 5-year bins, etc.)

    Returns:
        Start year of the bin

    Examples:
        >>> get_bin_start(2003, 1)
        2003
        >>> get_bin_start(2003, 5)
        2000
        >>> get_bin_start(1998, 5)
        1995
    """
    if bin_size <= 1:
        return year
    return (year // bin_size) * bin_size


def bin_year_records(value_bytes: bytes, bin_size: int) -> bytes:
    """
    Aggregate year records into bins.

    Takes packed year records in Packed24 format and aggregates them into bins
    according to bin_size. Years within each bin are summed together.

    Args:
        value_bytes: Packed records in <QQQ format (year, occurrences, documents)
        bin_size: Size of bins (1 = no change, 5 = 5-year bins, etc.)

    Returns:
        New packed records with binned data

    Examples:
        If bin_size=5 and input has years [2001, 2002, 2003], they all map to
        bin 2000 and their occurrences/documents are summed.
    """
    # Fast path: bin_size=1 means no binning needed
    if bin_size <= 1:
        return value_bytes

    # Parse all records
    if len(value_bytes) < RECORD_SIZE:
        return b""

    usable_len = (len(value_bytes) // RECORD_SIZE) * RECORD_SIZE
    records: List[Tuple[int, int, int]] = []

    for i in range(0, usable_len, RECORD_SIZE):
        year, occurrences, documents = struct.unpack(FMT, value_bytes[i:i+RECORD_SIZE])
        records.append((year, occurrences, documents))

    # Group by bin
    bin_dict: dict[int, Tuple[int, int]] = {}  # bin_start -> (total_occurrences, total_documents)

    for year, occurrences, documents in records:
        bin_start = get_bin_start(year, bin_size)

        if bin_start in bin_dict:
            prev_occ, prev_doc = bin_dict[bin_start]
            bin_dict[bin_start] = (prev_occ + occurrences, prev_doc + documents)
        else:
            bin_dict[bin_start] = (occurrences, documents)

    # Convert back to packed format, sorted by year
    result = bytearray()
    for bin_start in sorted(bin_dict.keys()):
        occurrences, documents = bin_dict[bin_start]
        result.extend(struct.pack(FMT, bin_start, occurrences, documents))

    return bytes(result)
