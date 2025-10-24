# ngram_pivot/__init__.py
"""N-gram database pivoting for year-based queries."""

from .stream import stream_year_ngrams, stream_year_range, get_ngram_stats
from .config import PipelineConfig
from .encoding import (
    encode_year_ngram_key,
    decode_year_ngram_key,
    year_prefix,
    encode_year_stats,
    decode_year_stats,
)
from .pipeline import run_pivot_pipeline, PivotOrchestrator

__all__ = [
    # Pipeline API
    "run_pivot_pipeline",
    "PivotOrchestrator",

    # Query interface
    "stream_year_ngrams",
    "stream_year_range",
    "get_ngram_stats",

    # Configuration
    "PipelineConfig",

    # Low-level encoding utilities
    "encode_year_ngram_key",
    "decode_year_ngram_key",
    "year_prefix",
    "encode_year_stats",
    "decode_year_stats",
]
