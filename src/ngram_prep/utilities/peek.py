from ngram_prep.common_db.api import open_db, range_scan, prefix_scan, scan_all
from typing import Optional

import struct


def format_data(data: bytes, encoding: str) -> str:
    """Format bytes for display based on encoding preference"""
    if encoding == "hex":
        return data.hex()
    elif encoding == "raw":
        return repr(data)
    elif encoding == "summary":
        # Aggregate across all years to show totals
        try:
            if len(data) % 24 == 0:
                num_records = len(data) // 24
                total_count = 0
                total_volumes = 0
                year_range = []

                for i in range(num_records):
                    offset = i * 24
                    year, count, volume_count = struct.unpack('<QQQ', data[offset:offset + 24])
                    total_count += count
                    total_volumes += volume_count
                    year_range.append(year)

                if year_range:
                    min_year, max_year = min(year_range), max(year_range)
                    return f"Total: {total_count:,} occurrences in {total_volumes:,} volumes ({min_year}-{max_year}, {num_records} years)"
                else:
                    return "No data"
            else:
                return f"<non-24-byte-aligned: {len(data)} bytes>"
        except struct.error as e:
            return f"<pack error: {e}>"
    elif encoding == "packed":
        # Decode <QQQ format (3 uint64s) - could be multiple records
        try:
            if len(data) % 24 == 0:  # Multiple of 24 bytes
                num_records = len(data) // 24
                records = []
                for i in range(min(num_records, 10)):  # Show first 10 records max
                    offset = i * 24
                    vals = struct.unpack('<QQQ', data[offset:offset + 24])
                    records.append(f"({vals[0]}, {vals[1]}, {vals[2]})")

                if num_records <= 10:
                    return f"[{num_records} records] " + ", ".join(records)
                else:
                    return f"[{num_records} records] " + ", ".join(records) + f", ... +{num_records - 10} more"
            else:
                return f"<non-24-byte-aligned: {len(data)} bytes>"
        except struct.error as e:
            return f"<pack error: {e}>"
    else:  # utf-8
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            return f"<binary: {data.hex()[:40]}{'...' if len(data) > 20 else ''}>"


def format_data(data: bytes, encoding: str) -> str:
    """Format bytes for display based on encoding preference"""
    if encoding == "hex":
        return data.hex()
    elif encoding == "raw":
        return repr(data)
    elif encoding == "summary":
        # Aggregate across all years to show totals
        try:
            if len(data) % 24 == 0:
                num_records = len(data) // 24
                total_count = 0
                total_volumes = 0
                year_range = []

                for i in range(num_records):
                    offset = i * 24
                    year, count, volume_count = struct.unpack('<QQQ', data[offset:offset + 24])
                    total_count += count
                    total_volumes += volume_count
                    year_range.append(year)

                if year_range:
                    min_year, max_year = min(year_range), max(year_range)
                    return f"Total: {total_count:,} occurrences in {total_volumes:,} volumes ({min_year}-{max_year}, {num_records} years)"
                else:
                    return "No data"
            else:
                return f"<non-24-byte-aligned: {len(data)} bytes>"
        except struct.error as e:
            return f"<pack error: {e}>"
    elif encoding == "packed":
        # Decode <QQQ format (3 uint64s) - show last records, wrap at 100 cols
        try:
            if len(data) % 24 == 0:  # Multiple of 24 bytes
                num_records = len(data) // 24
                records = []

                # Show last 10 records instead of first 10
                start_idx = max(0, num_records - 10)
                for i in range(start_idx, num_records):
                    offset = i * 24
                    vals = struct.unpack('<QQQ', data[offset:offset + 24])
                    records.append(f"({vals[0]}, {vals[1]}, {vals[2]})")

                # Format with wrapping at ~100 columns
                prefix = f"[{num_records} records] "
                if num_records > 10:
                    prefix += f"... +{start_idx} earlier, "

                # Indent continuation lines to align under first tuple
                indent = 12

                # Build wrapped output
                lines = []
                current_line = prefix

                for rec in records:
                    test_line = current_line + rec + ", "
                    if len(test_line) > 100 and current_line != prefix:
                        lines.append(current_line.rstrip(", "))
                        current_line = " " * indent + rec + ", "
                    else:
                        current_line = test_line

                # Add final line, remove trailing comma
                if current_line.strip():
                    lines.append(current_line.rstrip(", "))

                return "\n".join(lines) if lines else ""
            else:
                return f"<non-24-byte-aligned: {len(data)} bytes>"
        except struct.error as e:
            return f"<pack error: {e}>"
    else:  # utf-8
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            return f"<binary: {data.hex()[:40]}{'...' if len(data) > 20 else ''}>"


def db_head(db_path: str, n: int = 10, key_format: str = "utf-8", value_format: str = "utf-8"):
    """
    Show first N key-value pairs from RocksDB

    Args:
        db_path: Path to RocksDB database
        n: Number of pairs to show (default: 10)
        key_format: "utf-8", "hex", or "raw"
        value_format: "utf-8", "hex", or "raw"
    """
    with open_db(db_path, mode="r", profile="read") as db:
        print(f"First {n} key-value pairs:")
        print("─" * 100)

        count = 0
        for key, value in range_scan(db, b""):
            if count >= n:
                break

            formatted_key = format_data(key, key_format)
            formatted_value = format_data(value, value_format)

            print(f"[{count + 1:2d}] Key:   {formatted_key}")
            print(f"     Value: {formatted_value}")
            print()

            count += 1

        if count == 0:
            print("Database is empty")


def db_tail(db_path: str, n: int = 10, key_format: str = "utf-8", value_format: str = "utf-8"):
    """
    Show last N key-value pairs from RocksDB

    Args:
        db_path: Path to RocksDB database
        n: Number of pairs to show (default: 10)
        key_format: "utf-8", "hex", or "raw"
        value_format: "utf-8", "hex", or "raw"
    """
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        print("Scanning database for tail keys...")

        # Collect all keys (this will be slow for large DBs)
        all_pairs = list(scan_all(db))

        if not all_pairs:
            print("Database is empty")
            return

        print(f"Last {n} key-value pairs:")
        print("─" * 100)

        # Get last N pairs
        tail_pairs = all_pairs[-n:] if len(all_pairs) >= n else all_pairs

        for i, (key, value) in enumerate(tail_pairs):
            formatted_key = format_data(key, key_format)
            formatted_value = format_data(value, value_format)

            print(f"[{i + 1:2d}] Key:   {formatted_key}")
            print(f"     Value: {formatted_value}")
            print()


def db_peek(db_path: str, start_key: bytes = b"", n: int = 10,
            key_format: str = "utf-8", value_format: str = "utf-8"):
    """
    Show N key-value pairs starting from a specific key

    Args:
        db_path: Path to RocksDB database
        start_key: Key to start from (empty bytes = beginning)
        n: Number of pairs to show
        key_format: "utf-8", "hex", or "raw"
        value_format: "utf-8", "hex", or "raw"
    """
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        start_str = format_data(start_key, "hex") if start_key else "beginning"
        print(f"{n} key-value pairs starting from {start_str}:")
        print("─" * 100)

        count = 0
        for key, value in range_scan(db, start_key):
            if count >= n:
                break

            formatted_key = format_data(key, key_format)
            formatted_value = format_data(value, value_format)

            print(f"[{count + 1:2d}] Key:   {formatted_key}")
            print(f"     Value: {formatted_value}")
            print()

            count += 1


def db_peek_prefix(db_path: str, prefix: bytes, n: int = 10,
                   key_format: str = "utf-8", value_format: str = "utf-8"):
    """
    Show N key-value pairs with a specific prefix

    Args:
        db_path: Path to RocksDB database
        prefix: Prefix to search for
        n: Number of pairs to show
        key_format: "utf-8", "hex", or "raw"
        value_format: "utf-8", "hex", or "raw"
    """
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        prefix_str = format_data(prefix, "hex")
        print(f"{n} key-value pairs with prefix {prefix_str}:")
        print("─" * 100)

        count = 0
        for key, value in prefix_scan(db, prefix):
            if count >= n:
                break

            formatted_key = format_data(key, key_format)
            formatted_value = format_data(value, value_format)

            print(f"[{count + 1:2d}] Key:   {formatted_key}")
            print(f"     Value: {formatted_value}")
            print()

            count += 1

        if count == 0:
            print(f"No keys found with prefix {prefix_str}")


def db_get(db_path: str, key: bytes, value_format: str = "summary") -> Optional[bytes]:
    """
    Get a specific key's value

    Args:
        db_path: Path to RocksDB database
        key: Key to look up (as bytes)
        value_format: Format for displaying value ("summary", "packed", "hex", "raw")

    Returns:
        Formatted value string, or None if not found
    """
    with open_db(db_path, mode="r", profile="read") as db:
        try:
            result = db.get(key)
            if result is not None:
                return format_data(result, value_format)
            else:
                return None
        except Exception as e:
            return f"Error: {e}"

def db_sample(db_path: str, every_nth: int = 1000, n: int = 10,
              key_format: str = "utf-8", value_format: str = "utf-8"):
    """
    Sample every Nth key-value pair (useful for large databases)

    Args:
        db_path: Path to RocksDB database
        every_nth: Sample every Nth key (default: 1000)
        n: Number of samples to show
        key_format: "utf-8", "hex", or "raw"
        value_format: "utf-8", "hex", or "raw"
    """
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        print(f"Sampling every {every_nth} keys (showing {n} samples):")
        print("─" * 100)

        count = 0
        samples_shown = 0

        for key, value in range_scan(db, b""):
            if count % every_nth == 0:
                if samples_shown >= n:
                    break

                formatted_key = format_data(key, key_format)
                formatted_value = format_data(value, value_format)

                print(f"[{samples_shown + 1:2d}] Position ~{count:,}")
                print(f"     Key:   {formatted_key}")
                print(f"     Value: {formatted_value}")
                print()

                samples_shown += 1

            count += 1

# Example usage in notebook:
# 
# # Basic usage
# db_head("/path/to/your/db")
# db_tail("/path/to/your/db", 5)
# 
# # Hex format for binary data  
# db_head("/path/to/your/db", 3, key_format="hex", value_format="hex")
# 
# # Start from specific key (as bytes)
# db_peek("/path/to/your/db", start_key=b"some_prefix", n=5)
# 
# # Search by prefix
# db_peek_prefix("/path/to/your/db", prefix=b"ngram_", n=10)
# 
# # Sample large database
# db_sample("/path/to/your/db", every_nth=10000, n=5)
# 
# # Get specific value
# value = db_get("/path/to/your/db", b"my_key")
# if value:
#     print(f"Found: {value}")
# else:
#     print("Key not found")