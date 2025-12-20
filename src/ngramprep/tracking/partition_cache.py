"""Caching for work unit partitioning results."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

from .types import WorkUnit

__all__ = ["PartitionCache", "PartitionCacheKey"]


@dataclass(frozen=True)
class PartitionCacheKey:
    """Key for identifying cached partition results."""
    db_path: str
    num_units: int
    samples_per_worker: int
    num_sampling_workers: int

    def to_hash(self) -> str:
        """Generate a hash for this cache key."""
        key_str = f"{self.db_path}:{self.num_units}:{self.samples_per_worker}:{self.num_sampling_workers}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class PartitionCache:
    """Manages caching of work unit partitioning results."""

    CACHE_VERSION = "1.0"
    CACHE_FILENAME = "partition_cache.json"

    def __init__(self, cache_dir: Path):
        """
        Initialize partition cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / self.CACHE_FILENAME
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_work_units(self, work_units: List[WorkUnit]) -> List[Dict[str, Any]]:
        """
        Serialize work units to JSON-compatible format.

        Args:
            work_units: List of WorkUnit objects

        Returns:
            List of dictionaries with base64-encoded byte keys
        """
        serialized = []
        for unit in work_units:
            serialized.append({
                "unit_id": unit.unit_id,
                "start_key": unit.start_key.hex() if unit.start_key else None,
                "end_key": unit.end_key.hex() if unit.end_key else None,
            })
        return serialized

    def _deserialize_work_units(self, serialized: List[Dict[str, Any]]) -> List[WorkUnit]:
        """
        Deserialize work units from JSON format.

        Args:
            serialized: List of dictionaries with hex-encoded byte keys

        Returns:
            List of WorkUnit objects
        """
        work_units = []
        for item in serialized:
            work_units.append(WorkUnit(
                unit_id=item["unit_id"],
                start_key=bytes.fromhex(item["start_key"]) if item["start_key"] else None,
                end_key=bytes.fromhex(item["end_key"]) if item["end_key"] else None,
            ))
        return work_units

    def save(
        self,
        cache_key: PartitionCacheKey,
        work_units: List[WorkUnit],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save work units to cache.

        Args:
            cache_key: Cache key identifying this partition
            work_units: List of work units to cache
            metadata: Optional metadata to store with cache
        """
        cache_data = {
            "version": self.CACHE_VERSION,
            "cache_key": asdict(cache_key),
            "cache_key_hash": cache_key.to_hash(),
            "num_work_units": len(work_units),
            "work_units": self._serialize_work_units(work_units),
            "metadata": metadata or {},
        }

        # Load existing cache or create new
        all_caches = self._load_cache_file()

        # Add/update this cache entry
        cache_hash = cache_key.to_hash()
        all_caches[cache_hash] = cache_data

        # Write back to file
        self._write_cache_file(all_caches)

    def load(self, cache_key: PartitionCacheKey) -> Optional[List[WorkUnit]]:
        """
        Load work units from cache.

        Args:
            cache_key: Cache key identifying the partition to load

        Returns:
            List of WorkUnit objects, or None if not found
        """
        all_caches = self._load_cache_file()
        cache_hash = cache_key.to_hash()

        if cache_hash not in all_caches:
            return None

        cache_data = all_caches[cache_hash]

        # Verify version compatibility
        if cache_data.get("version") != self.CACHE_VERSION:
            return None

        # Deserialize and return work units
        return self._deserialize_work_units(cache_data["work_units"])

    def exists(self, cache_key: PartitionCacheKey) -> bool:
        """
        Check if cached partition exists.

        Args:
            cache_key: Cache key to check

        Returns:
            True if cache exists, False otherwise
        """
        all_caches = self._load_cache_file()
        return cache_key.to_hash() in all_caches

    def get_info(self, cache_key: PartitionCacheKey) -> Optional[Dict[str, Any]]:
        """
        Get information about a cached partition.

        Args:
            cache_key: Cache key to query

        Returns:
            Dictionary with cache metadata, or None if not found
        """
        all_caches = self._load_cache_file()
        cache_hash = cache_key.to_hash()

        if cache_hash not in all_caches:
            return None

        cache_data = all_caches[cache_hash]
        return {
            "version": cache_data.get("version"),
            "num_work_units": cache_data.get("num_work_units"),
            "metadata": cache_data.get("metadata", {}),
            "cache_key": cache_data.get("cache_key"),
        }

    def clear(self, cache_key: Optional[PartitionCacheKey] = None) -> None:
        """
        Clear cached partitions.

        Args:
            cache_key: Specific cache to clear, or None to clear all
        """
        if cache_key is None:
            # Clear all caches
            self._write_cache_file({})
        else:
            # Clear specific cache
            all_caches = self._load_cache_file()
            cache_hash = cache_key.to_hash()
            if cache_hash in all_caches:
                del all_caches[cache_hash]
                self._write_cache_file(all_caches)

    def list_all(self) -> List[Dict[str, Any]]:
        """
        List all cached partitions.

        Returns:
            List of cache info dictionaries
        """
        all_caches = self._load_cache_file()
        return [
            {
                "cache_hash": cache_hash,
                "version": data.get("version"),
                "num_work_units": data.get("num_work_units"),
                "cache_key": data.get("cache_key"),
                "metadata": data.get("metadata", {}),
            }
            for cache_hash, data in all_caches.items()
        ]

    def _load_cache_file(self) -> Dict[str, Any]:
        """Load all caches from file."""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Corrupted cache file, start fresh
            return {}

    def _write_cache_file(self, all_caches: Dict[str, Any]) -> None:
        """Write all caches to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(all_caches, f, indent=2)
