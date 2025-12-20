"""Configuration and data structures for Davies acquisition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CorpusConfig:
    """Configuration for a Davies corpus.

    Attributes:
        name: Corpus name (e.g., "COHA", "COCA")
        path: Path to corpus directory
        text_dir: Subdirectory containing text files (default: "text")
        has_decades: Whether files are organized by decades (e.g., 1810s, 1820s)
        decade_pattern: Regex pattern for extracting decade from filename
        year_pattern: Regex pattern for extracting year from filename
    """
    name: str
    path: Path
    text_dir: str = "text"
    has_decades: bool = True
    decade_pattern: str = r"text_(\d{4})s_"
    year_pattern: Optional[str] = None


@dataclass
class AcquisitionConfig:
    """Configuration for acquisition pipeline execution.

    Attributes:
        corpus: Corpus configuration
        db_path: Path for output RocksDB
        workers: Number of parallel workers
        overwrite_db: Whether to remove existing DB before starting
        write_batch_size: Number of entries per batch write
    """
    corpus: CorpusConfig
    db_path: Path
    workers: int = 8
    overwrite_db: bool = True
    write_batch_size: int = 100_000


def build_coha_config(corpus_path: str | Path) -> CorpusConfig:
    """Build configuration for COHA corpus.

    Args:
        corpus_path: Path to COHA corpus directory

    Returns:
        CorpusConfig for COHA
    """
    return CorpusConfig(
        name="COHA",
        path=Path(corpus_path),
        text_dir="text",
        has_decades=True,
        decade_pattern=r"text_(\d{4})s_",
        year_pattern=None,  # COHA uses decades
    )


def build_db_path(
    db_path_stub: str | Path,
    corpus_name: str,
) -> Path:
    """Build database path from stub and corpus name.

    Args:
        db_path_stub: Base directory for databases
        corpus_name: Name of corpus (e.g., "COHA", "COCA")

    Returns:
        Full path to database directory

    Example:
        >>> build_db_path("/scratch/corpora", "COHA")
        Path('/scratch/corpora/COHA/raw.db')
    """
    stub = Path(db_path_stub)
    return stub / corpus_name / "raw.db"
