"""Density-based work unit partitioning through keyspace sampling."""

from __future__ import annotations

import random
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .types import WorkUnit
from .partitioning import make_unit_id
from ..common_db.api import open_db, range_scan


@dataclass
class SamplingResult:
    """Result of parallel keyspace sampling."""
    all_samples: List[bytes]  # All samples, sorted
    total_keys_scanned: int   # Total keys seen across all workers
    empty_ranges: int         # Number of ranges with no data


class DensitySampler:
    """Sample keyspace density to create balanced work partitions."""

    @staticmethod
    def reservoir_sample(
        db,
        start_key: Optional[bytes],
        end_key: Optional[bytes],
        k: int,
        metadata_prefix: bytes = b"__"
    ) -> Tuple[List[bytes], int]:
        """
        Reservoir sample k keys from a database range.

        Uses Algorithm R (Vitter, 1985) to uniformly sample k keys
        from a range with unknown size, using O(k) memory.

        Args:
            db: RocksDB database handle
            start_key: Range start (None for beginning)
            end_key: Range end exclusive (None for end)
            k: Number of samples to collect
            metadata_prefix: Prefix to skip (metadata keys)

        Returns:
            (samples, total_keys_seen)
        """
        reservoir = []
        n = 0  # Total keys processed

        start = start_key if start_key is not None else b""

        for key, _ in range_scan(db, start, end_key):
            # Skip metadata keys
            if key.startswith(metadata_prefix):
                continue

            n += 1

            if len(reservoir) < k:
                # Fill reservoir
                reservoir.append(key)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, n - 1)
                if j < k:
                    reservoir[j] = key

        return reservoir, n

    def parallel_sample_keyspace(
        self,
        db_path: Path,
        num_workers: int,
        samples_per_worker: int,
        read_profile: Optional[str] = None
    ) -> SamplingResult:
        """
        Sample keyspace density using parallel workers.

        Divides keyspace into uniform ranges and samples each in parallel.
        Each worker performs reservoir sampling independently.

        Args:
            db_path: Path to database to sample
            num_workers: Number of parallel sampling workers
            samples_per_worker: Reservoir size per worker
            read_profile: Optional RocksDB read profile

        Returns:
            SamplingResult with sorted samples and statistics
        """
        # Create uniform sampling ranges (1-byte keyspace for <=256 workers)
        if num_workers <= 256:
            keyspace_size = 256
            key_bytes = 1
        else:
            keyspace_size = 65536
            key_bytes = 2

        step = keyspace_size / num_workers
        ranges = []

        for i in range(num_workers):
            start_val = int(i * step)
            end_val = int((i + 1) * step)

            if i == 0:
                start = None
            else:
                start = start_val.to_bytes(key_bytes, 'big')

            if i == num_workers - 1:
                end = None
            else:
                end = end_val.to_bytes(key_bytes, 'big')

            ranges.append((start, end))

        # Launch parallel sampling
        with mp.Pool(processes=num_workers) as pool:
            args = [
                (db_path, start, end, samples_per_worker, read_profile)
                for start, end in ranges
            ]
            results = pool.starmap(self._sample_range_worker, args)

        # Aggregate results
        all_samples = []
        total_keys = 0
        empty_ranges = 0

        for samples, key_count in results:
            all_samples.extend(samples)
            total_keys += key_count
            if key_count == 0:
                empty_ranges += 1

        # Sort all samples
        all_samples.sort()

        return SamplingResult(
            all_samples=all_samples,
            total_keys_scanned=total_keys,
            empty_ranges=empty_ranges
        )

    @staticmethod
    def _sample_range_worker(
        db_path: Path,
        start_key: Optional[bytes],
        end_key: Optional[bytes],
        k: int,
        read_profile: Optional[str]
    ) -> Tuple[List[bytes], int]:
        """
        Worker function for parallel sampling.

        This runs in a separate process and samples one range.
        """
        with open_db(db_path, mode='r', profile=read_profile) as db:
            sampler = DensitySampler()
            return sampler.reservoir_sample(db, start_key, end_key, k)

    def create_balanced_work_units(
        self,
        samples: List[bytes],
        total_estimated_keys: int,
        num_units: int
    ) -> List[WorkUnit]:
        """
        Create work units with approximately equal data density.

        Partitions the keyspace at sample boundaries to create units
        with equal numbers of samples, which approximates equal key counts.

        Args:
            samples: Sorted sample keys from keyspace
            total_estimated_keys: Estimated total keys in database
            num_units: Number of work units to create

        Returns:
            List of WorkUnit objects with balanced data distribution
        """
        if not samples:
            # No samples - fall back to single unit
            return [WorkUnit(unit_id="unit_", start_key=None, end_key=None)]

        if len(samples) < num_units:
            # Insufficient samples - use what we have
            num_units = len(samples)

        # Calculate samples per unit for equal distribution
        samples_per_unit = len(samples) / num_units

        # Create boundaries at sample positions
        boundaries = []
        for i in range(1, num_units):
            idx = int(i * samples_per_unit)
            boundaries.append(samples[idx])

        # Build work units from boundaries
        work_units = []

        for i in range(num_units):
            if i == 0:
                start = None
                end = boundaries[0]
            elif i == num_units - 1:
                start = boundaries[-1]
                end = None
            else:
                start = boundaries[i - 1]
                end = boundaries[i]

            unit_id = make_unit_id(start, end)
            work_units.append(WorkUnit(
                unit_id=unit_id,
                start_key=start,
                end_key=end
            ))

        return work_units
