"""Parser for Google Ngrams data format."""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)

__all__ = ["YearFreq", "NgramRecord", "parse_line"]


class YearFreq(TypedDict):
    """Year-level frequency data for an n-gram."""
    year: int
    frequency: int
    document_count: int


class NgramRecord(TypedDict):
    """Complete n-gram record with all year frequencies."""
    frequencies: List[YearFreq]


def parse_line(
        line: str,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
) -> Tuple[Optional[str], Optional[NgramRecord]]:
    """
    Parse Google Ngrams line format into structured data.

    Format: "ngram\\tYEAR,FREQ,DOC\\tYEAR,FREQ,DOC..."

    Args:
        line: Raw line from ngrams file
        filter_pred: Optional predicate to filter n-grams by text
                     (applied before parsing frequency data)

    Returns:
        (ngram_text, record) if valid and passes filter, else (None, None)

    Notes:
        - Returns (None, None) for empty/malformed lines or filtered n-grams
        - Skips malformed frequency tuples silently
        - Requires at least one valid frequency tuple to succeed

    Examples:
        >>> parse_line("hello\\t2000,100,50\\t2001,150,60")
        ('hello', {'frequencies': [
            {'year': 2000, 'frequency': 100, 'document_count': 50},
            {'year': 2001, 'frequency': 150, 'document_count': 60}
        ]})
    """
    s = line.strip()
    if not s:
        return None, None

    # Split into n-gram text and frequency data
    parts = s.split("\t", 1)
    if len(parts) != 2:
        return None, None

    ngram_text, freq_blob = parts

    # Apply filter before parsing frequency data (optimization)
    if filter_pred is not None and not filter_pred(ngram_text):
        return None, None

    frequencies: List[YearFreq] = []

    # Parse frequency tuples: "year,frequency,document_count"
    for entry in freq_blob.split("\t"):
        parts = entry.split(",")
        if len(parts) != 3:
            continue

        try:
            frequencies.append({
                "year": int(parts[0]),
                "frequency": int(parts[1]),
                "document_count": int(parts[2]),
            })
        except ValueError:
            # Skip malformed numeric values
            continue

    # Only return if we got at least one valid frequency
    if not frequencies:
        return None, None

    return ngram_text, {"frequencies": frequencies}