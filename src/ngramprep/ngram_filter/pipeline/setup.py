"""Setup and initialization for the ngram filter pipeline."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..config import FilterConfig, PipelineConfig
from .whitelist import load_whitelist


__all__ = [
    "PipelineSetup",
    "prepare_paths",
    "prepare_directories",
    "prepare_vocabulary_index",
    "prepare_whitelist",
]


class PipelineSetup:
    """Handles initialization and setup for the pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        filter_config: FilterConfig,
    ):
        """
        Initialize pipeline setup.

        Args:
            pipeline_config: Configuration for pipeline execution
            filter_config: Configuration for ngram filtering
        """
        self.pipeline_config = pipeline_config
        self.filter_config = filter_config
        self.temp_paths = prepare_paths(pipeline_config)

    def prepare_all(self) -> tuple[dict[str, Path], FilterConfig]:
        """
        Prepare all resources for pipeline execution.

        Returns:
            Tuple of (paths dictionary, updated filter config)
        """
        prepare_directories(self.temp_paths, self.pipeline_config)
        self.filter_config = prepare_vocabulary_index(self.filter_config)
        self.filter_config = prepare_whitelist(self.filter_config)

        return self.temp_paths, self.filter_config


def prepare_paths(pipeline_config: PipelineConfig) -> dict[str, Path]:
    """
    Initialize and prepare all required paths.

    Args:
        pipeline_config: Pipeline configuration

    Returns:
        Dictionary mapping path names to resolved Path objects
    """
    paths = {
        'src_db': pipeline_config.src_db.resolve(),
        'dst_db': pipeline_config.dst_db.resolve(),
        'tmp_dir': pipeline_config.tmp_dir.resolve(),
    }

    # Store partition cache in parent directory (not in tmp_dir) so it survives restarts
    # The cache is based on source DB characteristics, not processing state
    paths['cache_dir'] = paths['tmp_dir'].parent / ".partition_cache"
    paths['base'] = paths['cache_dir']  # Used by PartitionCache
    paths['work_tracker'] = paths['tmp_dir'] / "work_tracker.db"
    paths['output_dir'] = paths['tmp_dir'] / "worker_outputs"

    return paths


def prepare_directories(
    temp_paths: dict[str, Path],
    pipeline_config: PipelineConfig,
) -> None:
    """
    Clean and create necessary directories based on mode.

    Mode behavior:
        - "restart": Wipes output DB and temp directory (including cache)
        - "resume": Preserves all existing state
        - "reprocess": Wipes output DB but preserves temp directory (cache)

    Args:
        temp_paths: Dictionary of pipeline paths
        pipeline_config: Pipeline configuration
    """
    mode = getattr(pipeline_config, 'mode', 'resume')

    # Clean destination DB for restart/reprocess modes
    if mode in ('restart', 'reprocess') and temp_paths['dst_db'].exists():
        shutil.rmtree(temp_paths['dst_db'])

    temp_paths['dst_db'].parent.mkdir(parents=True, exist_ok=True)

    # Clean temp directory only for restart mode (reprocess preserves cache)
    if mode == 'restart' and temp_paths['tmp_dir'].exists():
        shutil.rmtree(temp_paths['tmp_dir'])

    # Create temp directories
    temp_paths['tmp_dir'].mkdir(parents=True, exist_ok=True)
    temp_paths['output_dir'].mkdir(exist_ok=True)

    # Create cache directory (survives restarts, outside of tmp_dir)
    temp_paths['cache_dir'].mkdir(parents=True, exist_ok=True)


def prepare_vocabulary_index(filter_config: FilterConfig) -> FilterConfig:
    """
    Prepare memory-mapped vocabulary index if needed.

    Args:
        filter_config: Filter configuration

    Returns:
        Updated filter configuration with vocabulary index path
    """
    vocab_path = getattr(filter_config, "vocab_path", None)
    if not vocab_path:
        return filter_config

    from ..filters.shared_vocab import build_vocab_index

    vocab_path = Path(vocab_path)
    idx_prefix = vocab_path.parent / "vocab_mmap"
    idx_file = idx_prefix.with_suffix(".idx")
    lex_file = idx_prefix.with_suffix(".lex")

    # Check if index needs rebuilding
    if _vocabulary_index_needs_rebuild(vocab_path, idx_file, lex_file):
        print("Building vocabulary index...")
        build_vocab_index(vocab_path, idx_prefix)

    # Update filter config to use the index
    return replace(filter_config, vocab_path=idx_prefix)


def _vocabulary_index_needs_rebuild(
    vocab_path: Path,
    idx_file: Path,
    lex_file: Path,
) -> bool:
    """
    Check if vocabulary index needs to be rebuilt.

    Args:
        vocab_path: Path to vocabulary file
        idx_file: Path to index file
        lex_file: Path to lexicon file

    Returns:
        True if rebuild needed, False otherwise
    """
    if not idx_file.exists() or not lex_file.exists():
        return True

    # Check if vocab file is newer than index files
    vocab_mtime = vocab_path.stat().st_mtime
    idx_mtime = max(
        idx_file.stat().st_mtime,
        lex_file.stat().st_mtime,
    )
    return vocab_mtime > idx_mtime


def prepare_whitelist(filter_config: FilterConfig) -> FilterConfig:
    """
    Load whitelist if specified in filter config.

    Args:
        filter_config: Filter configuration

    Returns:
        Updated filter configuration with loaded whitelist
    """
    whitelist_path = getattr(filter_config, "whitelist_path", None)
    if not whitelist_path:
        return filter_config

    whitelist_path = Path(whitelist_path)
    if not whitelist_path.exists():
        raise FileNotFoundError(
            f"Whitelist file not found: {whitelist_path}"
        )

    print("Loading whitelist...")

    # Load whitelist with optional parameters from config
    min_count = getattr(filter_config, "whitelist_min_count", 1)
    top_n = getattr(filter_config, "whitelist_top_n", None)

    whitelist = load_whitelist(
        whitelist_path=whitelist_path,
        min_count=min_count,
        top_n=top_n,
    )

    print(f"Loaded {len(whitelist):,} tokens")

    # Store the loaded whitelist in filter_config for workers
    return replace(filter_config, whitelist=whitelist)
