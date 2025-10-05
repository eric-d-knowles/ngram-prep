"""Incremental RocksDB compaction for Jupyter notebooks."""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm

import rocks_shim


def _truncate_path_to_fit(path, prefix, total_width=100):
    from ngram_filter.pipeline.orchestrator import _truncate_path_to_fit as _tp
    return _tp(path, prefix, total_width)


def format_bytes(num_bytes: int) -> str:
    """Convert bytes to human-readable format.

    Args:
        num_bytes: Number of bytes to format.

    Returns:
        Human-readable string with appropriate unit.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def get_db_size(db: rocks_shim.DB) -> Optional[int]:
    """Get approximate database size in bytes.

    Args:
        db: RocksDB database instance.

    Returns:
        Total SST file size in bytes, or None if unavailable.
    """
    prop = db.get_property("rocksdb.total-sst-files-size")
    return int(prop) if prop else None


def generate_prefix_ranges(prefix_bytes: int = 1) -> List[Tuple[Optional[bytes], Optional[bytes]]]:
    """Generate key ranges based on byte prefixes.

    Args:
        prefix_bytes: Number of prefix bytes (creates 256^n ranges).

    Returns:
        List of (start_key, end_key) tuples.

    Raises:
        ValueError: If prefix_bytes < 1.
    """
    if prefix_bytes < 1:
        raise ValueError("prefix_bytes must be >= 1")

    ranges = []
    total_ranges = 256 ** prefix_bytes

    for i in range(total_ranges):
        # Convert range index to prefix bytes
        prefix = i.to_bytes(prefix_bytes, byteorder='big')
        # End key is prefix + 0xFF padding
        end = prefix + b'\xff'
        ranges.append((prefix, end))

    # First range starts from beginning
    ranges[0] = (None, ranges[0][1])

    return ranges


def compact_incremental(
        db_path: str,
        prefix_bytes: Optional[int] = None,
        overwrite_checkpoint: bool = False,
) -> None:
    """Compact database incrementally using cached work units or prefix ranges.

    This function can operate in two modes:
    1. Cache-based: Uses pre-computed work units from a JSON file
    2. Prefix-based: Generates ranges on-the-fly based on key prefixes

    Checkpoint files are automatically named based on the database name and mode.

    Args:
        db_path: Path to RocksDB database (e.g., '/path/to/5grams.db').
        prefix_bytes: If provided, use prefix-based ranges (1, 2, 3, etc.).
                     If None, look for cached work units file.
        overwrite_checkpoint: If True, remove existing checkpoint before starting.

    Raises:
        FileNotFoundError: If database or required cache file not found.
        ValueError: If prefix_bytes is less than 1.
    """
    db_path = Path(db_path)
    start_time = datetime.now()

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Determine compaction mode
    use_cache = prefix_bytes is None

    # Construct config path: strip "_processed" suffix if present
    db_stem = db_path.stem.replace('_processed', '')
    config_path = db_path.parent / f"{db_stem}.db.work_units.json"

    # Create checkpoint filename matching work_units style
    if use_cache:
        checkpoint_path = db_path.parent / f"{db_stem}.db.compaction.checkpoint"
    else:
        checkpoint_path = (db_path.parent /
                           f"{db_stem}.db.compaction_prefix{prefix_bytes}.checkpoint")

    # Remove checkpoint if overwrite requested
    if overwrite_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    # Check if we should fall back to full compaction
    use_full_compaction = use_cache and not config_path.exists()

    if use_full_compaction:
        print(f"\nWork units config not found: {config_path}", flush=True)
        print("Falling back to full database compaction (single pass).", flush=True)

    # Header
    if use_full_compaction:
        mode = "FULL COMPACTION"
    elif use_cache:
        mode = "CACHE-BASED"
    else:
        mode = "PREFIX-BASED"
    print(f"\nINCREMENTAL COMPACTION ({mode})", flush=True)
    print("━" * 100, flush=True)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Load work units or generate ranges
    if use_full_compaction:
        # Full compaction: single work unit covering entire keyspace
        work_units = [{'unit_id': 'full', 'start_key': None, 'end_key': None}]
        total_units = 1
        estimated_keys = None
        use_cache = True  # Use cache-style processing for full compaction
    elif use_cache:
        with open(config_path) as f:
            config = json.load(f)
        work_units = config['work_units']
        total_units = len(work_units)
        estimated_keys = config['db_fingerprint'].get('estimated_keys', 0)
    else:
        work_units = generate_prefix_ranges(prefix_bytes)
        total_units = len(work_units)
        estimated_keys = None

    # Check for checkpoint
    completed = set()
    if checkpoint_path.exists() and not use_full_compaction:
        with open(checkpoint_path) as f:
            if use_cache:
                completed = set(line.strip() for line in f)
            else:
                completed = set(int(line.strip()) for line in f)

    units_remaining = total_units - len(completed)

    # Configuration
    print("\nConfiguration", flush=True)
    print("—" * 100, flush=True)
    db_path_str = _truncate_path_to_fit(
        db_path,
        "DB path:           ",
        100
    )
    print(f"DB path:           {db_path_str}", flush=True)
    if use_full_compaction:
        print(f"Mode:              Full compaction (no work units file found)", flush=True)
    elif use_cache:
        print(f"Work units file:   {config_path.name}", flush=True)
        if estimated_keys:
            print(f"Estimated keys:    {estimated_keys:,}", flush=True)
    else:
        print(f"Prefix bytes:      {prefix_bytes}", flush=True)
    print(f"Total units:       {total_units:,}", flush=True)
    print(f"Units remaining:   {units_remaining:,}", flush=True)
    print(f"Checkpoint file:   {checkpoint_path.name}", flush=True)

    # Open database to get initial size
    db = rocks_shim.open(
        str(db_path),
        mode="rw",
        profile="read:packed24",
        create_if_missing=False
    )

    initial_size = get_db_size(db)
    if initial_size:
        print(f"Initial DB size:   {format_bytes(initial_size)}", flush=True)

    # Progress
    print("\nProgress", flush=True)
    print("—" * 100, flush=True)
    sys.stdout.flush()

    completed_this_run = 0
    failed_units = []
    process_start = time.time()

    try:
        # Process work units with progress bar
        with tqdm(total=total_units,
                  initial=len(completed),
                  desc="Compacting",
                  unit="unit",
                  ncols=100,
                  bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            for i, unit in enumerate(work_units):
                # Skip if already completed
                if use_cache:
                    unit_id = unit['unit_id']
                    if unit_id in completed:
                        continue
                    checkpoint_value = unit_id
                    start_key = (bytes.fromhex(unit['start_key'])
                                 if unit['start_key'] else None)
                    end_key = (bytes.fromhex(unit['end_key'])
                               if unit['end_key'] else None)
                    label = unit_id
                else:
                    if i in completed:
                        continue
                    checkpoint_value = str(i)
                    start_key, end_key = unit
                    prefix_str = (start_key.hex()[:prefix_bytes * 2]
                                  if start_key else "start")
                    label = f"prefix {prefix_str}"

                # Perform compaction
                try:
                    start_time_unit = time.time()
                    db.compact_range(start=start_key, end=end_key, exclusive=True)
                    elapsed = time.time() - start_time_unit

                    # Write checkpoint (atomic append)
                    with open(checkpoint_path, 'a') as f:
                        f.write(f"{checkpoint_value}\n")

                    completed_this_run += 1
                    pbar.update(1)

                    # Adjust label width for longer prefixes
                    label_width = max(12, prefix_bytes * 2 + 7) if not use_cache else 12
                    pbar.set_description(f"[{label:<{label_width}} {elapsed:5.1f}s]")

                except Exception as e:
                    failed_units.append((i, label, str(e)))
                    pbar.write(f"✗ Unit {i + 1} ({label}): FAILED - {e}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved to checkpoint.", flush=True)
    except Exception as e:
        print(f"\n\nError: {e}", flush=True)
        print("Progress saved to checkpoint. Re-run to resume.", flush=True)
    finally:
        # Final summary
        elapsed_total = time.time() - process_start
        end_time = datetime.now()

        final_size = get_db_size(db)
        db.close()

        print("\nIncremental Compaction Summary", flush=True)
        print("—" * 100, flush=True)
        print(f"Units compacted this run:    {completed_this_run:,}", flush=True)

        if failed_units:
            print(f"Units failed:                {len(failed_units):,}", flush=True)

        print(f"Total units completed:       "
              f"{len(completed) + completed_this_run:,}/{total_units:,}", flush=True)

        if initial_size and final_size:
            saved = initial_size - final_size
            pct = (saved / initial_size) * 100
            print(f"Size before:                 {format_bytes(initial_size)}", flush=True)
            print(f"Size after:                  {format_bytes(final_size)}", flush=True)
            print(f"Space saved:                 "
                  f"{format_bytes(saved)} ({pct:.1f}%)", flush=True)

        print(f"Total runtime:               "
              f"{str(timedelta(seconds=int(elapsed_total)))}", flush=True)

        if completed_this_run > 0:
            time_per_unit = elapsed_total / completed_this_run
            units_per_hour = 3600 / time_per_unit
            print(f"Time per unit:               "
                  f"{str(timedelta(seconds=int(time_per_unit)))}", flush=True)
            print(f"Units per hour:              {units_per_hour:.1f}", flush=True)

        if failed_units:
            print("\nFailed Units:", flush=True)
            for unit_index, label, error in failed_units:
                print(f"  Unit {unit_index + 1} ({label}): {error}", flush=True)