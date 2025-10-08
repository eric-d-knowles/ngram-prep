# ngram_pivot/config.py
"""Configuration for pivot operations."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PivotConfig:
    """Configuration for pivoting n-gram database."""

    source_db_path: Path
    """Path to source database (n-gram indexed)"""

    target_db_path: Path
    """Path to target database (year indexed)"""

    batch_size: int = 10000
    """Number of source records to process per batch"""

    write_batch_size: int = 5000
    """Number of target records to write per batch"""

    source_profile: str = "read:packed24"
    """RocksDB profile for source database"""

    target_profile: str = "write:packed24"
    """RocksDB profile for target database"""

    log_interval: int = 100000
    """Log progress every N source records"""

    validate: bool = True
    """Validate source data during pivot"""

    @classmethod
    def from_dict(cls, d: dict) -> "PivotConfig":
        """Create config from dictionary."""
        d = d.copy()
        d["source_db_path"] = Path(d["source_db_path"])
        d["target_db_path"] = Path(d["target_db_path"])
        return cls(**d)