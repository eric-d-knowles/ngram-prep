# ngram_pivot/__init__.py
"""N-gram database pivoting for year-based queries."""

from .pivot import pivot_database, PivotStats
from .stream import stream_year_ngrams, stream_year_range, get_ngram_stats
from .config import PivotConfig
from .encoding import (
    encode_year_ngram_key,
    decode_year_ngram_key,
    year_prefix,
    encode_year_stats,
    decode_year_stats,
)

__all__ = [
    # Core pivot functionality
    "pivot_database",
    "PivotStats",

    # Query interface
    "stream_year_ngrams",
    "stream_year_range",
    "get_ngram_stats",

    # Configuration
    "PivotConfig",

    # Low-level encoding utilities
    "encode_year_ngram_key",
    "decode_year_ngram_key",
    "year_prefix",
    "encode_year_stats",
    "decode_year_stats",
]