"""
Davies corpus filtering pipeline for preprocessing and cleaning.

This module provides filtering and preprocessing capabilities for
Davies corpus data stored in RocksDB, preparing it for word2vec training.

Main entry point:
    filter_davies_corpus() - Full filtering pipeline

Key components:
    - config: Configuration for filtering options
    - processor: Apply filters (lowercase, lemmatization, stopwords)
    - core: Main filtering orchestration
"""

from .core import filter_davies_corpus
from .config import FilterConfig, PipelineConfig
from .whitelist import write_whitelist, load_whitelist

__all__ = [
    "filter_davies_corpus",
    "FilterConfig",
    "PipelineConfig",
    "write_whitelist",
    "load_whitelist",
]
