# ngram_filter/partitioning/intelligent.py
"""
Intelligent data-aware partitioning for work units.

This module creates work units based on actual data density sampling
rather than naive key range division.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from common_db.api import open_db
from ngram_filter.pipeline.work_tracker import WorkUnit
from utilities.reservoir_sampler import reservoir_sampling


@dataclass
class KeyRangeDensity:
    """Represents the data density for a key range."""
    start_key: bytes
    end_key: bytes
    estimated_count: int
    sample_size: int


@dataclass
class DBFingerprint:
    """Database fingerprint for cache validation."""
    estimated_keys: int

    def matches(self, other: 'DBFingerprint', tolerance: float = 0.1) -> bool:
        """Check if fingerprints match within tolerance."""
        # Allow some variation in estimated keys
        if self.estimated_keys == 0 or other.estimated_keys == 0:
            return self.estimated_keys == other.estimated_keys

        ratio = abs(self.estimated_keys - other.estimated_keys) / max(self.estimated_keys, other.estimated_keys)
        return ratio <= tolerance


class WorkUnitCache:
    """Manages caching of work units to avoid repeated sampling."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.cache_path = db_path.parent / f"{db_path.name}.work_units.json"

    def _compute_fingerprint(self) -> DBFingerprint:
        """Compute a fingerprint of the database for validation."""
        with open_db(self.db_path, mode="ro") as db:
            # Get estimated key count
            total_records_str = db.get_property("rocksdb.estimate-num-keys")
            estimated_keys = int(total_records_str) if total_records_str else 0

            return DBFingerprint(estimated_keys=estimated_keys)

    def load(self, num_units: int, sample_rate: float, prefix_length: int) -> Optional[List[WorkUnit]]:
        """
        Load work units from cache if valid.

        Returns:
            List of WorkUnit objects if cache is valid, None otherwise
        """
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, 'r') as f:
                cache_data = json.load(f)

            # Validate configuration matches
            config = cache_data.get('sampling_config', {})
            if (config.get('num_units') != num_units or
                    config.get('sample_rate') != sample_rate or
                    config.get('prefix_length') != prefix_length):
                print(f"  Cache config mismatch, will resample")
                return None

            # Validate fingerprint
            current_fp = self._compute_fingerprint()
            cached_fp = DBFingerprint(**cache_data['db_fingerprint'])

            if not current_fp.matches(cached_fp):
                print(f"  Database changed, cache invalid")
                return None

            # Reconstruct work units
            work_units = []
            for unit_data in cache_data['work_units']:
                work_units.append(WorkUnit(
                    unit_id=unit_data['unit_id'],
                    start_key=bytes.fromhex(unit_data['start_key']) if unit_data['start_key'] else None,
                    end_key=bytes.fromhex(unit_data['end_key']) if unit_data['end_key'] else None
                ))

            return work_units

        except Exception as e:
            print(f"  Cache load failed: {e}")
            return None

    def save(self, work_units: List[WorkUnit], num_units: int, sample_rate: float, prefix_length: int):
        """Save work units to cache."""
        try:
            fingerprint = self._compute_fingerprint()

            cache_data = {
                'version': 1,
                'source_db_path': str(self.db_path),
                'created_at': datetime.now().isoformat(),
                'db_fingerprint': asdict(fingerprint),
                'sampling_config': {
                    'num_units': num_units,
                    'sample_rate': sample_rate,
                    'prefix_length': prefix_length
                },
                'work_units': [
                    {
                        'unit_id': unit.unit_id,
                        'start_key': unit.start_key.hex() if unit.start_key else None,
                        'end_key': unit.end_key.hex() if unit.end_key else None
                    }
                    for unit in work_units
                ]
            }

            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)

            print(f"  Saved work units to cache: {self.cache_path}")

        except Exception as e:
            print(f"  Cache save failed: {e}")


class IntelligentPartitioner:
    """Creates work units based on actual data density sampling."""

    def __init__(self, src_db_path: Path, sample_rate: float = 0.001):
        """
        Initialize partitioner.

        Args:
            src_db_path: Path to source database
            sample_rate: Fraction of database to sample (0.001 = 0.1%)
        """
        self.src_db_path = src_db_path
        self.sample_rate = sample_rate
        self.density_map: Dict[bytes, int] = {}

    def sample_database_density(self, prefix_length: int = 1) -> Dict[bytes, int]:
        print(f"  Sampling database at {self.sample_rate:.5f} rate (prefix_length={prefix_length})...")

        # Get approximate target sample size
        with open_db(self.src_db_path, mode="ro") as db:
            total_records_str = db.get_property("rocksdb.estimate-num-keys")
            if total_records_str:
                target_samples = int(int(total_records_str) * self.sample_rate)
            else:
                target_samples = 250000  # Fallback default

        print(f"  Targeting {target_samples:,} samples using reservoir sampling")

        # Use reservoir sampling to get samples
        samples = reservoir_sampling(
            str(self.src_db_path),
            sample_size=target_samples,
            return_keys=True,
            progress_interval=10000000
        )

        # Build prefix counts from samples
        prefix_counts = {}
        for key_bytes, _ in samples:
            prefix = key_bytes[:prefix_length]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        # Scale up to estimated totals
        scale_factor = 1.0 / self.sample_rate
        estimated_counts = {p: int(c * scale_factor) for p, c in prefix_counts.items()}

        print(f"  Sampling complete: {len(samples):,} samples collected, {len(prefix_counts)} unique prefixes")
        self.density_map = estimated_counts
        return estimated_counts

    def create_balanced_work_units(self, num_units: int, target_records_per_unit: int = None) -> List[WorkUnit]:
        """
        Create work units with roughly equal estimated record counts.

        Args:
            num_units: Desired number of work units
            target_records_per_unit: Target records per unit (auto-calculated if None)

        Returns:
            List of WorkUnit objects with balanced workloads
        """
        if not self.density_map:
            raise ValueError("Must call sample_database_density() first")

        total_estimated_records = sum(self.density_map.values())
        if target_records_per_unit is None:
            target_records_per_unit = total_estimated_records // num_units

        print(f"  Creating {num_units} work units targeting {target_records_per_unit:,} records each")

        # Sort prefixes by key to maintain ordering
        sorted_prefixes = sorted(self.density_map.items(), key=lambda x: x[0])

        work_units = []
        current_unit_count = 0
        current_start_key = None
        unit_index = 0

        for prefix, estimated_count in sorted_prefixes:
            current_unit_count += estimated_count

            # Check if we should close this work unit
            should_close = (
                    current_unit_count >= target_records_per_unit or
                    unit_index == num_units - 1
            )

            if should_close:
                # Create work unit
                end_key = self._get_next_key(prefix) if unit_index < num_units - 1 else None

                work_units.append(WorkUnit(
                    unit_id=f"unit_{unit_index:04d}",
                    start_key=current_start_key,
                    end_key=end_key
                ))

                print(f"    Unit {unit_index}: {current_start_key.hex() if current_start_key else 'start'} → "
                      f"{end_key.hex() if end_key else 'end'} (~{current_unit_count:,} records)")

                # Reset for next unit
                current_start_key = end_key
                current_unit_count = 0
                unit_index += 1

                if unit_index >= num_units:
                    break

        print(f"  Created {len(work_units)} balanced work units")
        return work_units

    def _get_next_key(self, key: bytes) -> bytes:
        """Get the next key in lexicographic order."""
        if not key:
            return b'\x00'

        # Try to increment the last byte
        key_list = list(key)
        for i in range(len(key_list) - 1, -1, -1):
            if key_list[i] < 255:
                key_list[i] += 1
                return bytes(key_list)
            else:
                key_list[i] = 0

        # All bytes were 255, need to extend
        return key + b'\x00'

    def analyze_balance(self, work_units: List[WorkUnit]) -> None:
        """Analyze the balance of work units based on density estimates."""
        if not self.density_map:
            print("  No density data available for analysis")
            return

        unit_estimates = []
        for unit in work_units:
            estimated_records = self._estimate_unit_records(unit)
            unit_estimates.append(estimated_records)

        if unit_estimates:
            avg_records = sum(unit_estimates) / len(unit_estimates)
            min_records = min(unit_estimates)
            max_records = max(unit_estimates)

            print(f"  Work unit balance analysis:")
            print(f"    Average: {avg_records:,.0f} records per unit")
            print(f"    Range: {min_records:,} to {max_records:,}")
            print(f"    Ratio: {max_records / min_records:.1f}x difference")

    def _estimate_unit_records(self, unit: WorkUnit) -> int:
        """Estimate record count for a work unit based on density map."""
        total = 0
        for prefix, count in self.density_map.items():
            if self._key_in_unit_range(prefix, unit):
                total += count
        return total

    def _key_in_unit_range(self, key: bytes, unit: WorkUnit) -> bool:
        """Check if a key falls within a work unit's range."""
        if unit.start_key is not None and key < unit.start_key:
            return False
        if unit.end_key is not None and key >= unit.end_key:
            return False
        return True


def create_intelligent_work_units(
        src_db_path: Path,
        num_units: int = 128,
        sample_rate: float = 0.001,
        prefix_length: int = 2,
        sort_largest_first: bool = True,
        use_cache: bool = True,
        force_resample: bool = False
) -> List[WorkUnit]:
    """
    Create intelligently balanced work units based on actual data density.

    Args:
        src_db_path: Path to source database
        num_units: Number of work units to create
        sample_rate: Fraction of database to sample for density estimation
        prefix_length: Length of key prefix for density bucketing
        sort_largest_first: If True, sort units largest-first for optimal parallelization
        use_cache: If True, attempt to load from cache before sampling
        force_resample: If True, ignore cache and resample database

    Returns:
        List of balanced WorkUnit objects
    """
    cache = WorkUnitCache(src_db_path)

    # Try to load from cache first
    if use_cache and not force_resample:
        cached_units = cache.load(num_units, sample_rate, prefix_length)
        if cached_units:
            print(f"  Loaded {len(cached_units)} work units from cache")
            return cached_units

    # Cache miss or forced resample - do the full sampling
    partitioner = IntelligentPartitioner(src_db_path, sample_rate=sample_rate)

    # Sample the database to understand density
    partitioner.sample_database_density(prefix_length=prefix_length)

    # Create balanced work units
    work_units = partitioner.create_balanced_work_units(num_units)

    # Analyze the balance
    partitioner.analyze_balance(work_units)

    # Sort by size if requested (for better parallel efficiency)
    if sort_largest_first:
        work_units_with_estimates = [
            (unit, partitioner._estimate_unit_records(unit))
            for unit in work_units
        ]
        work_units_with_estimates.sort(key=lambda x: x[1], reverse=True)
        work_units = [unit for unit, _ in work_units_with_estimates]

        print(f"  Sorted {len(work_units)} work units by size (largest first) for optimal parallelization")

    # Save to cache
    if use_cache:
        cache.save(work_units, num_units, sample_rate, prefix_length)

    return work_units