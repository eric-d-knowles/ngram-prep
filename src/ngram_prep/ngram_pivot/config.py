# ngram_pivot/config.py
"""Configuration for pivot operations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline orchestration configuration for parallel pivot execution."""

    # I/O
    src_db: Path
    dst_db: Path
    tmp_dir: Path

    # Parallelism
    num_workers: int = 8
    num_initial_work_units: Optional[int] = None  # If None, defaults to num_workers
    max_split_depth: int = 5  # Maximum depth for recursive splitting (2^5 = 32 splits per unit)
    split_check_interval_s: float = 120.0  # How often workers check if they should split (0 to disable)

    # Progress reporting
    progress_every_s: float = 5.0

    # Worker configuration
    max_items_per_bucket: int = 100_000
    max_bytes_per_bucket: int = 128 * 1024 * 1024

    # Reader/Writer profiles
    reader_profile: str = "read:packed24"
    writer_profile: str = 'write:packed24'
    writer_disable_wal: bool = True

    # Ingest configuration (runs as separate stage after workers complete)
    ingest_read_profile: str = 'read:packed24'
    ingest_write_profile: str = 'write:packed24'
    ingest_batch_items: int = 10_000_000  # 10M for larger write batches on NVMe
    ingest_disable_wal: bool = True
    num_ingest_readers: int = 8  # Number of parallel reader processes for ingestion
    ingest_queue_depth: int = 50  # Queue depth between reader processes and writer
    num_ingest_reader_threads: int = 16  # DEPRECATED: No longer used (parallel processes instead)

    # Pipeline control
    mode: Literal["restart", "resume", "reprocess"] = "resume"
    enable_ingest: bool = True  # If False, skip ingestion stage (shards remain on disk)
    compact_after_ingest: bool = True
    work_unit_claim_order: Literal["sequential", "random"] = "sequential"

    # Validation
    validate: bool = True
