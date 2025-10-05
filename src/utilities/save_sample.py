#!/usr/bin/env python3
"""
Save reservoir sampling results to a new RocksDB database.
"""

import shutil
import struct
import sys
from pathlib import Path
import rocks_shim as rs


def unpack_ngram_value(value_bytes):
    """
    Unpack N-gram frequency data from packed bytes.

    Args:
        value_bytes: Packed bytes from N-gram database

    Returns:
        List of (year, frequency, document_count) tuples
    """
    if not value_bytes or len(value_bytes) % 24 != 0:  # Each record is 3 uint64s = 24 bytes
        return []

    num_values = len(value_bytes) // 8  # Number of uint64 values
    unpacked = struct.unpack(f"<{num_values}Q", value_bytes)

    # Group into triplets: (year, frequency, document_count)
    frequencies = []
    for i in range(0, len(unpacked), 3):
        if i + 2 < len(unpacked):
            year = unpacked[i]
            frequency = unpacked[i + 1]
            document_count = unpacked[i + 2]
            frequencies.append((year, frequency, document_count))

    return frequencies


def format_ngram_data(key_str, frequencies, max_years=5):
    """
    Format N-gram data for readable display.

    Args:
        key_str: The N-gram key (decoded string)
        frequencies: List of (year, frequency, document_count) tuples
        max_years: Maximum number of years to display

    Returns:
        Formatted string representation
    """
    if not frequencies:
        return f"N-gram: '{key_str}' - No frequency data"

    lines = [f"N-gram: '{key_str}'"]

    # Sort by year and show most recent years
    sorted_freqs = sorted(frequencies, key=lambda x: x[0], reverse=True)
    displayed_freqs = sorted_freqs[:max_years]

    for year, freq, doc_count in displayed_freqs:
        lines.append(f"  {year}: freq={freq:,}, docs={doc_count:,}")

    if len(frequencies) > max_years:
        lines.append(f"  ... and {len(frequencies) - max_years} more years")

    return "\n".join(lines)


def save_sample_to_db(sample, output_db_path, overwrite=False):
    """
    Save sample data to a new RocksDB database.

    Args:
        sample: List of (key, value) tuples from reservoir sampling
        output_db_path: Path for the new database
        overwrite: If True, remove existing database first
    """
    output_path = Path(output_db_path)

    # Handle existing database
    if output_path.exists():
        if overwrite:
            print(f"Removing existing database at {output_db_path}")
            shutil.rmtree(output_path)
        else:
            raise ValueError(f"Database already exists at {output_db_path}. Use overwrite=True to replace it.")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(sample):,} samples to {output_db_path}")

    try:
        with rs.open(output_db_path, mode="rw", create_if_missing=True) as db:
            with db.write_batch() as batch:
                # Sample contains (key, value) tuples
                for key_str, value_bytes in sample:
                    key = key_str.encode('utf-8') if isinstance(key_str, str) else key_str
                    batch.put(key, value_bytes)

        print(f"Successfully saved {len(sample):,} samples")
        return True

    except Exception as e:
        print(f"Error saving sample: {e}")
        return False


def verify_sample_db(db_path, show_count=5, decode_output=False, unpack_ngram=False):
    """
    Verify the saved database and show sample entries.

    Args:
        db_path: Path to the database
        show_count: Number of entries to display
        decode_output: If True, attempt to decode bytes as UTF-8
        unpack_ngram: If True, unpack and format N-gram frequency data
    """
    try:
        with rs.open(db_path, mode="r") as db:
            iterator = db.iterator()
            iterator.seek(b"")

            count = 0
            total_count = 0

            print(f"\nSample entries from {db_path}:")
            print("-" * 60)

            while iterator.valid():
                if count < show_count:
                    key_bytes = iterator.key()
                    value_bytes = iterator.value()

                    if decode_output:
                        try:
                            key_display = key_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            key_display = f"<bytes: {key_bytes}>"
                    else:
                        key_display = key_bytes

                    if unpack_ngram and decode_output:
                        try:
                            key_str = key_bytes.decode('utf-8')
                            frequencies = unpack_ngram_value(value_bytes)
                            value_display = format_ngram_data(key_str, frequencies)
                        except Exception as e:
                            value_display = f"Error unpacking N-gram data: {e}"
                    elif decode_output:
                        try:
                            value_display = value_bytes.decode('utf-8')
                            if len(value_display) > 100:
                                value_display = value_display[:100] + "..."
                        except UnicodeDecodeError:
                            value_preview = value_bytes[:50] + b"..." if len(value_bytes) > 50 else value_bytes
                            value_display = f"<bytes: {value_preview}>"
                    else:
                        value_preview = value_bytes[:50] + b"..." if len(value_bytes) > 50 else value_bytes
                        value_display = value_preview

                    if unpack_ngram and decode_output:
                        print(value_display)  # Already formatted
                    else:
                        print(f"Key: {key_display}")
                        print(f"Value: {value_display}")
                    print("-" * 30)
                    count += 1

                total_count += 1
                iterator.next()

            del iterator  # Clean up iterator

            print(f"Total entries in sample database: {total_count:,}")
            return True

    except Exception as e:
        print(f"Error verifying database: {e}")
        return False
