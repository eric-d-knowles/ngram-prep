"""Work unit partitioning and splitting logic."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

from .types import WorkUnit

__all__ = ["create_uniform_work_units", "create_smart_work_units", "find_midpoint_key", "make_unit_id"]


def make_unit_id(start_key: Optional[bytes], end_key: Optional[bytes]) -> str:
    """
    Create a short, readable unit ID from key boundaries.

    Uses short hex prefixes for human readability plus hash for uniqueness.
    Format: unit_<start_prefix>_<end_prefix>_<hash6>

    Args:
        start_key: Start boundary (None for beginning)
        end_key: End boundary (None for end)

    Returns:
        Compact unit ID like "unit_20_40_a1b2c3"

    Examples:
        >>> make_unit_id(None, b'\\x20')
        'unit__20_...'
        >>> make_unit_id(b'\\x20', b'\\x40')
        'unit_20_40_...'
    """
    # Use first 2 bytes (4 hex chars) for human readability
    if start_key is None:
        start_prefix = ""
    else:
        start_prefix = start_key[:2].hex() if len(start_key) >= 2 else start_key.hex()

    if end_key is None:
        end_prefix = ""
    else:
        end_prefix = end_key[:2].hex() if len(end_key) >= 2 else end_key.hex()

    # Add hash for uniqueness (6 hex chars = 24 bits should be plenty)
    # Hash the full keys to ensure uniqueness
    h = hashlib.sha256()
    h.update(start_key if start_key else b'')
    h.update(b'|')  # Separator
    h.update(end_key if end_key else b'')
    hash_suffix = h.hexdigest()[:6]

    return f"unit_{start_prefix}_{end_prefix}_{hash_suffix}"


def create_uniform_work_units(num_units: int) -> List[WorkUnit]:
    """
    Create work units by uniformly dividing the byte keyspace.

    This creates initial work units with equal byte-range coverage. Unlike
    intelligent/sampling-based partitioning, this completes in milliseconds
    and relies on dynamic splitting at runtime to handle actual data imbalance.

    The keyspace is divided into equal slices:
    - For ≤256 units: Uses 1-byte boundaries (0x00 to 0xFF)
    - For >256 units: Uses 2-byte boundaries (0x0000 to 0xFFFF)

    - Unit 0: None → boundary_1 (beginning of keyspace to first boundary)
    - Unit 1: boundary_1 → boundary_2
    - Unit 2: boundary_2 → boundary_3
    - ...
    - Unit N-1: boundary_N-1 → None (last boundary to end of keyspace)

    Args:
        num_units: Number of work units to create (1 to 65536)

    Returns:
        List of WorkUnit objects with uniform byte-range coverage

    Example:
        >>> # Create 8 work units (1-byte boundaries)
        >>> units = create_uniform_work_units(8)
        >>> len(units)
        8
        >>> # First unit covers beginning of keyspace
        >>> units[0].start_key is None
        True
        >>> # Last unit covers end of keyspace
        >>> units[-1].end_key is None
        True
        >>> # Create 513 work units (2-byte boundaries)
        >>> units = create_uniform_work_units(513)
        >>> len(units)
        513
        >>> len(units[1].start_key)
        2

    Note:
        - Automatically switches to 2-byte boundaries for >256 units
        - Supports up to 65,536 initial work units
        - Dynamic splitting will handle actual data imbalance at runtime
        - Much faster than sampling-based approaches (milliseconds vs hours)
        - No caching needed - can regenerate instantly
    """
    if num_units < 1:
        raise ValueError(f"num_units must be >= 1, got {num_units}")

    if num_units == 1:
        # Single unit covers entire keyspace
        return [WorkUnit(unit_id="unit__", start_key=None, end_key=None)]

    work_units = []

    # Determine keyspace size based on number of units
    # Use 1-byte boundaries for <= 256 units, 2-byte for more
    if num_units <= 256:
        # Single-byte keyspace (0x00 to 0xFF)
        keyspace_size = 256
        key_bytes = 1
    else:
        # Two-byte keyspace (0x0000 to 0xFFFF)
        keyspace_size = 65536
        key_bytes = 2

    # Calculate byte boundaries
    step = keyspace_size / num_units

    for i in range(num_units):
        # Calculate boundary values
        start_val = int(i * step)
        end_val = int((i + 1) * step)

        # Convert to keys
        # First unit starts at None (beginning of keyspace)
        if i == 0:
            start_key = None
        else:
            start_key = start_val.to_bytes(key_bytes, 'big')

        # Last unit ends at None (end of keyspace)
        if i == num_units - 1:
            end_key = None
        else:
            end_key = end_val.to_bytes(key_bytes, 'big')

        # Create compact unit_id
        unit_id = make_unit_id(start_key, end_key)

        work_units.append(
            WorkUnit(unit_id=unit_id, start_key=start_key, end_key=end_key)
        )

    return work_units


def create_smart_work_units(
    db_path: Path,
    num_units: int,
    num_sampling_workers: Optional[int] = None,
    samples_per_worker: int = 10000,
    read_profile: Optional[str] = None
) -> List[WorkUnit]:
    """
    Create density-based work units through parallel sampling.

    Uses reservoir sampling to understand actual key distribution
    and creates balanced work units with approximately equal data density.

    This is slower than uniform partitioning (requires database scan) but
    produces much better load balancing for skewed data distributions.

    Args:
        db_path: Path to database to sample
        num_units: Number of work units to create
        num_sampling_workers: Number of parallel sampling workers (default: min(num_units, 40))
        samples_per_worker: Reservoir size per worker (default: 10000)
        read_profile: Optional RocksDB read profile

    Returns:
        List of WorkUnit objects with balanced data distribution

    Example:
        >>> from pathlib import Path
        >>> units = create_smart_work_units(
        ...     db_path=Path("/data/ngrams.db"),
        ...     num_units=40,
        ...     num_sampling_workers=20,
        ...     samples_per_worker=10000
        ... )
        >>> len(units)
        40
    """
    from .density_sampling import DensitySampler

    if num_sampling_workers is None:
        num_sampling_workers = min(num_units, 40)

    sampler = DensitySampler()

    # Run parallel sampling
    sampling_result = sampler.parallel_sample_keyspace(
        db_path,
        num_sampling_workers,
        samples_per_worker,
        read_profile
    )

    # Create balanced partitions
    work_units = sampler.create_balanced_work_units(
        sampling_result.all_samples,
        sampling_result.total_keys_scanned,
        num_units
    )

    return work_units


# ============================================================================
# Work Unit Splitting Logic
# ============================================================================


def find_midpoint_key(start_key: Optional[bytes], end_key: Optional[bytes]) -> Optional[bytes]:
    """
    Find the midpoint between two keys.

    Args:
        start_key: Start of range (None = beginning)
        end_key: End of range (None = end)

    Returns:
        Midpoint key, or None if cannot find midpoint
    """
    if start_key is None and end_key is None:
        return None  # Can't split infinite range

    if start_key is None:
        return _halve_key(end_key)

    if end_key is None:
        return start_key + b'\x80'  # Append mid-range byte

    # Both keys exist - find byte-wise midpoint
    return _midpoint_between_keys(start_key, end_key)


def _midpoint_between_keys(start: bytes, end: bytes) -> Optional[bytes]:
    """Calculate midpoint between two byte strings."""
    # Make them the same length by padding with zeros
    max_len = max(len(start), len(end))
    start_padded = start + b'\x00' * (max_len - len(start))
    end_padded = end + b'\x00' * (max_len - len(end))

    # Convert to integers
    start_int = int.from_bytes(start_padded, 'big')
    end_int = int.from_bytes(end_padded, 'big')

    if start_int >= end_int:
        return None

    # Calculate midpoint
    mid_int = (start_int + end_int) // 2

    if mid_int == start_int:
        # Keys are adjacent at current precision - extend by one byte to split finer
        # e.g., split between b'\x50' and b'\x51' at b'\x50\x80'
        # This allows splitting units even when they span adjacent byte values
        extended_start = start_padded + b'\x00'
        extended_end = end_padded + b'\x00'

        start_int_ext = int.from_bytes(extended_start, 'big')
        end_int_ext = int.from_bytes(extended_end, 'big')

        mid_int_ext = (start_int_ext + end_int_ext) // 2

        if mid_int_ext == start_int_ext:
            return None  # Still too close even with extension

        mid_bytes = mid_int_ext.to_bytes(max_len + 1, 'big')
    else:
        # Convert back to bytes
        mid_bytes = mid_int.to_bytes(max_len, 'big')

    # Trim trailing zeros
    mid_bytes = mid_bytes.rstrip(b'\x00')

    return mid_bytes if mid_bytes else b'\x00'


def _halve_key(key: bytes) -> bytes:
    """Return a key that's roughly half of the given key."""
    key_int = int.from_bytes(key, 'big')
    half_int = key_int // 2
    return half_int.to_bytes(len(key), 'big')
