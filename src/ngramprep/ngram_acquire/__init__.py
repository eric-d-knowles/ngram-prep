"""
N-gram acquisition pipeline for downloading and ingesting Google Ngrams data.

This module provides a complete pipeline for downloading, parsing, and ingesting
n-gram data from the Google Ngrams corpus into RocksDB.

Main entry point:
    download_and_ingest_to_rocksdb() - Full pipeline orchestration

Key components:
    - core: Main pipeline orchestration
    - coordinator: File discovery and validation
    - executor: Parallel file processing
    - worker: Individual file download and parsing
    - batch_writer: Batched database writes
    - reporter: Progress and statistics display
"""

from ngramprep.ngram_acquire.core import download_and_ingest_to_rocksdb

__all__ = ["download_and_ingest_to_rocksdb"]
