# parallel/partitioning.py
"""Work unit partitioning and splitting logic."""

from __future__ import annotations

from typing import List, Optional

from ngram_prep.parallel.types import WorkUnit

__all__ = ["create_uniform_work_units", "find_midpoint_key"]


def create_uniform_work_units(num_units: int) -> List[WorkUnit]:
    """
    Create work units by uniformly dividing the byte keyspace.

    This creates initial work units with equal byte-range coverage. Unlike
    intelligent/sampling-based partitioning, this completes in milliseconds
    and relies on dynamic splitting at runtime to handle actual data imbalance.

    The keyspace (0x00 to 0xFF for the first byte) is divided into equal slices:
    - Unit 0: None → boundary_1 (beginning of keyspace to first boundary)
    - Unit 1: boundary_1 → boundary_2
    - Unit 2: boundary_2 → boundary_3
    - ...
    - Unit N-1: boundary_N-1 → None (last boundary to end of keyspace)

    Args:
        num_units: Number of work units to create (typically num_workers * 1-4)

    Returns:
        List of WorkUnit objects with uniform byte-range coverage

    Example:
        >>> # Create 8 work units
        >>> units = create_uniform_work_units(8)
        >>> len(units)
        8
        >>> # First unit covers beginning of keyspace
        >>> units[0].start_key is None
        True
        >>> units[0].unit_id
        'unit__20'
        >>> # Last unit covers end of keyspace
        >>> units[-1].end_key is None
        True
        >>> units[-1].unit_id
        'unit_e0_'

    Note:
        - Uses single-byte boundaries for simplicity (divides 0x00-0xFF range)
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

    # Calculate byte boundaries (divide 0-256 range)
    # Use 256 (not 255) so last unit ends at None (end of keyspace)
    step = 256 / num_units

    for i in range(num_units):
        # Calculate boundary values
        start_val = int(i * step)
        end_val = int((i + 1) * step)

        # Convert to keys
        # First unit starts at None (beginning of keyspace)
        start_key = None if i == 0 else bytes([start_val])

        # Last unit ends at None (end of keyspace)
        end_key = None if i == num_units - 1 else bytes([end_val])

        # Create unit_id with start/end hash encoding
        start_hash = start_key.hex() if start_key else ""
        end_hash = end_key.hex() if end_key else ""
        unit_id = f"unit_{start_hash}_{end_hash}"

        work_units.append(
            WorkUnit(unit_id=unit_id, start_key=start_key, end_key=end_key)
        )

    return work_units


def format_work_units_summary(work_units: List[WorkUnit]) -> str:
    """
    Format a summary of work units for display.

    Args:
        work_units: List of work units to summarize

    Returns:
        Formatted string with work unit boundaries

    Example:
        >>> units = create_uniform_work_units(4)
        >>> print(format_work_units_summary(units))
        Created 4 uniform work units:
          unit__40: <start> → 40
          unit_40_80: 40 → 80
          unit_80_c0: 80 → c0
          unit_c0_: c0 → <end>
    """
    lines = [f"Created {len(work_units)} uniform work units:"]

    for unit in work_units:
        start_str = unit.start_key.hex() if unit.start_key else "<start>"
        end_str = unit.end_key.hex() if unit.end_key else "<end>"
        lines.append(f"  {unit.unit_id}: {start_str} → {end_str}")

    return "\n".join(lines)


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
