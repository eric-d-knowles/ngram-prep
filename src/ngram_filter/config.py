# ngram_filter/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Dict, Any, Union


# Filtering options used by readers.build_processor(...)
@dataclass(frozen=True)
class FilterConfig:
    opt_lower: bool = True
    opt_alpha: bool = True
    opt_shorts: bool = True
    opt_stops: bool = True
    opt_lemmas: bool = True
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
    dst_db: PathW
    tmp_dir: Path

    # Parallelism
    readers: int = 8
    work_units_per_reader: int = 8
    prefix_length: int = 2
    partitioning_sample_rate: float = 0.001

    # Progress reporting
    progress_every_s: float = 5.0

    # Worker configuration
    max_items_per_bucket: int = 100_000
    max_bytes_per_bucket: int = 128 * 1024 * 1024

    # Writer profiles
    writer_read_profile: str = "read:packed24"
    writer_write_profile: str = 'write:packed24'
    writer_disable_wal: bool = True

    # Ingest/merge configuration
    ingest_read_profile: str = 'read:packed24'
    ingest_write_profile: str = 'write:packed24'
    ingest_batch_bytes: int = 128 * 1024 * 1024
    ingest_batch_items: int = 200_000
    ingest_disable_wal: bool = True
    ingestors: int = 4
    delete_after_ingest: bool = False

    # Pipeline control
    mode: str = "resume"  # "restart", "resume", or "reprocess"
    force_cache_use: bool = True
    enable_ingest: bool = True
    overwrite_checkpoint: bool = False
    post_compact: bool = False
    prefix_bytes: int = None

    # Output whitelist generation (from filtered results)
    output_whitelist_path: Optional[Union[str, Path]] = None
    output_whitelist_top_n: Optional[int] = None