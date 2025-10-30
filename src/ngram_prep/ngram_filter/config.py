# ngram_filter/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Dict, Any, Union


# Filtering options used by filter processor
@dataclass(frozen=True)
class FilterConfig:
    lowercase: bool = True
    alpha_only: bool = True  # Filter out tokens with punctuation/numbers; accepts alphabetic chars from all languages
    filter_short: bool = True
    filter_stops: bool = True
    apply_lemmatization: bool = True
    min_len: int = 3
    stop_set: Optional[Set[str]] = None
    lemma_gen: Any = None
    tag_map: Optional[Dict] = None
    vocab_path: Optional[Union[str, Path]] = None
    vocab_view: Any = None

    # Whitelist configuration (for input filtering)
    whitelist_path: Optional[Union[str, Path]] = None
    whitelist_min_count: int = 1
    whitelist_top_n: Optional[int] = None
    whitelist: Optional[Set[bytes]] = None


# Pipeline orchestration options
@dataclass(frozen=True)
class PipelineConfig:
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

    # Writer profiles
    writer_read_profile: str = "read:packed24"
    writer_write_profile: str = 'write:packed24'
    writer_disable_wal: bool = True

    # Ingest configuration (parallel reads, sequential writes)
    # Note: Shards are always deleted after successful ingestion
    ingest_read_profile: str = 'read:packed24'
    ingest_write_profile: str = 'write:packed24'
    ingest_batch_items: int = 2_000_000
    ingest_disable_wal: bool = True
    ingest_num_readers: int = 4  # Number of parallel shard reader processes
    ingest_queue_size: int = 8  # Max shards buffered in memory (controls memory usage)

    # Pipeline control
    mode: str = "resume"  # "restart", "resume", or "reprocess"
    compact_after_ingest: bool = True
    work_unit_claim_order: str = "sequential"  # "sequential" or "random"

    # Output whitelist generation (from filtered results)
    output_whitelist_path: Optional[Union[str, Path]] = None
    output_whitelist_top_n: Optional[int] = None