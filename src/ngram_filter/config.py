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
    dst_db: Path
    tmp_dir: Path

    # Parallelism
    readers: int = 16

    # Progress reporting
    progress_every_s: float = 5.0

    # Worker configuration
    max_items_per_bucket: int = 25_000
    max_bytes_per_bucket: int = 16 * 1024 * 1024

    # Writer profiles
    writer_profile: str = 'write:packed24'
    writer_disable_wal: bool = True

    # Ingest/merge configuration
    ingest_read_profile: str = 'read:packed24'
    ingest_write_profile: str = 'write:packed24'
    ingest_batch_bytes: int = 64 * 1024 * 1024
    ingest_batch_items: int = 100_000
    ingest_disable_wal: bool = True

    # Pipeline control
    force_restart: bool = False

    # Output whitelist generation (from filtered results)
    output_whitelist_path: Optional[Union[str, Path]] = None
    output_whitelist_top_n: Optional[int] = None