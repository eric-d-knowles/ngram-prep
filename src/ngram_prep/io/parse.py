# ngram_prep/io/parse.py
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple, TypedDict, List

logger = logging.getLogger(__name__)


class YearFreq(TypedDict):
    year: int
    frequency: int
    document_count: int


class NgramRecord(TypedDict):
    frequencies: List[YearFreq]


def parse_line(
    line: str,
    *,
    filter_pred: Optional[Callable[[str], bool]] = None,
) -> Tuple[Optional[str], Optional[NgramRecord]]:
    """
    Parse "ngram\\tYEAR,FREQ,DOC\\tYEAR,FREQ,DOC..." into (key, record).

    - Applies filter_pred to the ngram text (before parsing tuples) if provided.
    - Returns (None, None) on empty/malformed lines or if filtered out.
    - Skips malformed tuples; requires at least one valid tuple to succeed.
    """
    s = line.strip()
    if not s:
        return None, None

    # Split into 'ngram text' and the rest (first tab only)
    try:
        ngram_text, freq_blob = s.split("\t", 1)
    except ValueError:
        return None, None

    if filter_pred is not None and not filter_pred(ngram_text):
        return None, None

    rec: NgramRecord = {"frequencies": []}

    # Each entry is "year,frequency,document_count" separated by tabs
    for entry in freq_blob.split("\t"):
        try:
            y_str, f_str, d_str = entry.split(",")
            rec["frequencies"].append(
                {
                    "year": int(y_str),
                    "frequency": int(f_str),
                    "document_count": int(d_str),
                }
            )
        except (ValueError, IndexError):
            # Skip malformed frequency entries quietly
            continue

    return (ngram_text, rec) if rec["frequencies"] else (None, None)
