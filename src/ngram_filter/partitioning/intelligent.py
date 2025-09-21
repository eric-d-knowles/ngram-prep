# ngram_filter/partitioning/intelligent.py
"""
Intelligent data-aware partitioning for work units.

This module creates work units based on actual data density sampling
rather than naive key range division.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass

from common_db.api import open_db, scan_all
from ngram_filter.pipeline.work_tracker import WorkUnit


@dataclass
class KeyRangeDensity:
    """Represents the data density for a key range."""
    start_key: bytes
    end_key: bytes
    estimated_count: int
    sample_size: int


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
        """
        Sample the database to estimate record counts by key prefix.

        Args:
            prefix_length: Number of bytes to use for prefix grouping

        Returns:
            Dictionary mapping key prefixes to estimated record counts
        """
        print(f"  Sampling database at {self.sample_rate:.3f} rate...")

        prefix_counts = {}
        total_sampled = 0

        with open_db(self.src_db_path, mode="ro") as db:
            for key, _ in scan_all(db):
                # Randomly sample based on sample_rate
                if random.random() > self.sample_rate:
                    continue

                total_sampled += 1

                # Extract prefix
                prefix = key[:prefix_length] if len(key) >= prefix_length else key
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

                # Progress indicator
                if total_sampled % 10000 == 0:
                    print(f"    Sampled {total_sampled:,} records, found {len(prefix_counts)} unique prefixes")

        # Scale up sample counts to estimated totals
        scale_factor = 1.0 / self.sample_rate
        estimated_counts = {
            prefix: int(count * scale_factor)
            for prefix, count in prefix_counts.items()
        }

        print(
            f"  Sampling complete: {total_sampled:,} records sampled, estimated {sum(estimated_counts.values()):,} total")
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
            if current_start_key is None:
                current_start_key = prefix

            current_unit_count += estimated_count

            # Check if we should close this work unit
            should_close = (
                    current_unit_count >= target_records_per_unit or  # Hit target size
                    unit_index == num_units - 1  # Last unit gets everything remaining
            )

            if should_close:
                # Create work unit
                end_key = self._get_next_key(prefix) if unit_index < num_units - 1 else None

                work_units.append(WorkUnit(
                    unit_id=f"unit_{unit_index:04d}",
                    start_key=current_start_key if unit_index > 0 else None,
                    end_key=end_key
                ))

                print(f"    Unit {unit_index}: {current_start_key.hex() if current_start_key else 'start'} → "
                      f"{end_key.hex() if end_key else 'end'} (~{current_unit_count:,} records)")

                # Reset for next unit
                current_start_key = self._get_next_key(prefix) if unit_index < num_units - 1 else None
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


def create_intelligent_work_units(src_db_path: Path, num_units: int = 128, sample_rate: float = 0.001) -> List[
    WorkUnit]:
    """
    Create intelligently balanced work units based on actual data density.

    Args:
        src_db_path: Path to source database
        num_units: Number of work units to create
        sample_rate: Fraction of database to sample for density estimation

    Returns:
        List of balanced WorkUnit objects
    """
    partitioner = IntelligentPartitioner(src_db_path, sample_rate=sample_rate)

    # Sample the database to understand density
    partitioner.sample_database_density(prefix_length=2)

    # Create balanced work units
    work_units = partitioner.create_balanced_work_units(num_units)

    # Analyze the balance
    partitioner.analyze_balance(work_units)

    return work_units