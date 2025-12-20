from ngramprep.common_db.api import open_db, range_scan, prefix_scan, scan_all
from typing import Optional, Union
import struct
import re


def parse_davies_key(key_str: str) -> bytes:
    """
    Parse a human-readable Davies key string like "[fic][2000] quick" into bytes.

    Args:
        key_str: String in format "[genre][year] text", "[genre][year]", etc.

    Returns:
        Encoded key bytes (4-byte genre + 8-byte year + text)
    """
    # Match pattern: [genre][year] optional_text
    match = re.match(r'\[([a-z]+)\]\[(\d+)\]\s*(.*)', key_str)
    if match:
        genre = match.group(1)
        year = int(match.group(2))
        text = match.group(3)

        # Encode genre as 4-byte fixed-width field
        genre_bytes = genre.encode('utf-8')[:4].ljust(4, b'\x00')
        # Encode year as 8-byte little-endian unsigned long
        year_bytes = struct.pack('<Q', year)
        # Encode text
        text_bytes = text.encode('utf-8') if text else b''

        return genre_bytes + year_bytes + text_bytes
    else:
        # Not Davies format, return None to signal fallback
        return None


def parse_pivot_key(key_str: str) -> bytes:
    """
    Parse a human-readable pivot key string like "[2000] quick" or "[2000]" into bytes.

    Args:
        key_str: String in format "[year] ngram", "[year]", or just "ngram"

    Returns:
        Encoded key bytes (4-byte year + ngram, or just 4-byte year if no ngram)
    """
    # Match pattern: [year] optional_ngram
    match = re.match(r'\[(\d+)\]\s*(.*)', key_str)
    if match:
        year = int(match.group(1))
        ngram = match.group(2)
        if ngram:  # Has ngram text after year
            return struct.pack('>I', year) + ngram.encode('utf-8')
        else:  # Just [year] - return year prefix only
            return struct.pack('>I', year)
    else:
        # No year specified, assume year 0
        return struct.pack('>I', 0) + key_str.encode('utf-8')


def normalize_key(key: Union[str, bytes]) -> bytes:
    """
    Normalize key input to bytes, handling both strings and bytes.

    Args:
        key: Either bytes or a string (will parse as Davies or pivot key)

    Returns:
        Key as bytes
    """
    if isinstance(key, str):
        # Try Davies format first ([genre][year] text)
        davies_key = parse_davies_key(key)
        if davies_key is not None:
            return davies_key
        # Fall back to pivot format ([year] text)
        return parse_pivot_key(key)
    return key


def format_key(data: bytes, encoding: str) -> str:
    """Format key bytes for display"""
    if encoding == "hex":
        return data.hex()
    elif encoding == "raw":
        return repr(data)
    elif encoding == "pivot":
        # Decode pivoted DB key: 4 bytes year + ngram text
        try:
            if len(data) >= 4:
                year = struct.unpack('>I', data[:4])[0]
                ngram = data[4:]
                try:
                    ngram_str = ngram.decode('utf-8')
                    return f"[{year}] {ngram_str}"
                except UnicodeDecodeError:
                    return f"[{year}] <binary: {ngram.hex()[:40]}{'...' if len(ngram) > 20 else ''}>"
            else:
                return f"<invalid pivot key: {len(data)} bytes>"
        except struct.error as e:
            return f"<pack error: {e}>"
    elif encoding == "davies":
        # Decode Davies key: 4 bytes genre + 8 bytes year + sentence text
        try:
            if len(data) >= 12:
                genre_bytes = data[:4]
                genre = genre_bytes.rstrip(b'\x00').decode('utf-8')
                year = struct.unpack('<Q', data[4:12])[0]
                sentence = data[12:]
                try:
                    sentence_str = sentence.decode('utf-8')
                    return f"[{genre}][{year}] {sentence_str}"
                except UnicodeDecodeError:
                    return f"[{genre}][{year}] <binary: {sentence.hex()[:40]}{'...' if len(sentence) > 20 else ''}>"
            else:
                return f"<invalid davies key: {len(data)} bytes>"
        except (struct.error, UnicodeDecodeError) as e:
            return f"<decode error: {e}>"
    elif encoding == "auto":
        # Auto-detect format
        # Try Davies format first (4-byte genre + 8-byte year + text)
        if len(data) >= 12:
            try:
                genre_bytes = data[:4]
                genre = genre_bytes.rstrip(b'\x00').decode('utf-8')
                year = struct.unpack('<Q', data[4:12])[0]
                # Check if genre looks like text and year is reasonable
                if genre.isalpha() and 1000 <= year <= 2100:
                    return format_key(data, "davies")
            except:
                pass
        # Try pivot format (4-byte year + text)
        if len(data) >= 4:
            try:
                year = struct.unpack('>I', data[:4])[0]
                if 1000 <= year <= 2100:  # Reasonable year range for Google Books
                    return format_key(data, "pivot")
            except:
                pass
        # Fall through to utf-8
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return f"<binary: {data.hex()[:40]}{'...' if len(data) > 20 else ''}>"
    else:  # utf-8
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            return f"<binary: {data.hex()[:40]}{'...' if len(data) > 20 else ''}>"


def format_value(data: bytes, encoding: str) -> str:
    """Format value bytes for display"""
    if encoding == "hex":
        return data.hex()
    elif encoding == "raw":
        return repr(data)
    elif encoding == "pivot":
        # Decode pivoted DB format: 16 bytes = 2 × uint64 (occurrences, documents)
        try:
            if len(data) == 16:
                occurrences, documents = struct.unpack('<QQ', data)
                return f"{occurrences:,} occurrences in {documents:,} documents"
            else:
                return f"<invalid pivot format: {len(data)} bytes, expected 16>"
        except struct.error as e:
            return f"<pack error: {e}>"
    elif encoding == "auto":
        # Auto-detect: if 16 bytes, assume pivot format; if multiple of 24, assume packed24
        if len(data) == 16:
            return format_value(data, "pivot")
        elif len(data) % 24 == 0:
            return format_value(data, "summary")
        else:
            return f"<unknown format: {len(data)} bytes>"
    elif encoding == "summary":
        # Aggregate across all years/bins to show totals (packed24 format)
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
                    return f"Total: {total_count:,} occurrences in {total_volumes:,} volumes ({min_year}-{max_year}, {num_records} bins)"
                else:
                    return "No data"
            else:
                return f"<non-24-byte-aligned: {len(data)} bytes>"
        except struct.error as e:
            return f"<pack error: {e}>"
    elif encoding == "packed":
        # Decode <QQQ format (3 uint64s) - show last records
        try:
            if len(data) % 24 == 0:
                num_records = len(data) // 24
                records = []
                start_idx = max(0, num_records - 10)
                for i in range(start_idx, num_records):
                    offset = i * 24
                    vals = struct.unpack('<QQQ', data[offset:offset + 24])
                    records.append(f"({vals[0]}, {vals[1]}, {vals[2]})")
                prefix = f"[{num_records} records] "
                if num_records > 10:
                    prefix += f"... +{start_idx} earlier, "
                indent = 12
                lines = []
                current_line = prefix
                for rec in records:
                    test_line = current_line + rec + ", "
                    if len(test_line) > 100 and current_line != prefix:
                        lines.append(current_line.rstrip(", "))
                        current_line = " " * indent + rec + ", "
                    else:
                        current_line = test_line
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


def db_head(db_path: str, n: int = 10, key_format: str = "auto", value_format: str = "auto"):
    """Show first N key-value pairs from RocksDB"""
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        print(f"First {n} key-value pairs:")
        print("─" * 100)
        count = 0
        for key, value in range_scan(db, b""):
            if count >= n:
                break
            formatted_key = format_key(key, key_format)
            formatted_value = format_value(value, value_format)
            print(f"[{count + 1:2d}] Key:   {formatted_key}")
            print(f"     Value: {formatted_value}")
            print()
            count += 1
        if count == 0:
            print("Database is empty")


def db_tail(db_path: str, n: int = 10, key_format: str = "auto", value_format: str = "auto"):
    """Show last N key-value pairs from RocksDB"""
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        print("Scanning database for tail keys...")
        all_pairs = list(scan_all(db))
        if not all_pairs:
            print("Database is empty")
            return
        print(f"Last {n} key-value pairs:")
        print("─" * 100)
        tail_pairs = all_pairs[-n:] if len(all_pairs) >= n else all_pairs
        for i, (key, value) in enumerate(tail_pairs):
            formatted_key = format_key(key, key_format)
            formatted_value = format_value(value, value_format)
            print(f"[{i + 1:2d}] Key:   {formatted_key}")
            print(f"     Value: {formatted_value}")
            print()


def db_peek(db_path: str, start_key: Union[str, bytes] = b"", n: int = 10,
            key_format: str = "auto", value_format: str = "auto"):
    """
    Show N key-value pairs starting from a specific key.

    Args:
        db_path: Path to database
        start_key: Start key as bytes or string (e.g., "[2000] quick" for pivoted DB)
        n: Number of pairs to show
        key_format: Display format for keys
        value_format: Display format for values
    """
    # Normalize key to bytes
    start_key_bytes = normalize_key(start_key) if start_key else b""

    with open_db(db_path, mode="r", profile="read:packed24") as db:
        start_str = format_key(start_key_bytes, "hex") if start_key_bytes else "beginning"
        print(f"{n} key-value pairs starting from {start_str}:")
        print("─" * 100)
        count = 0
        for key, value in range_scan(db, start_key_bytes):
            if count >= n:
                break
            formatted_key = format_key(key, key_format)
            formatted_value = format_value(value, value_format)
            print(f"[{count + 1:2d}] Key:   {formatted_key}")
            print(f"     Value: {formatted_value}")
            print()
            count += 1


def db_peek_prefix(db_path: str, prefix: Union[str, bytes], n: int = 10,
                   key_format: str = "auto", value_format: str = "auto"):
    """
    Show N key-value pairs with a specific prefix.

    Args:
        db_path: Path to database
        prefix: Prefix as bytes or string (e.g., "[2000]" for year 2000 in pivoted DB)
        n: Number of pairs to show
        key_format: Display format for keys
        value_format: Display format for values
    """
    # Normalize prefix to bytes
    prefix_bytes = normalize_key(prefix)

    with open_db(db_path, mode="r", profile="read:packed24") as db:
        prefix_str = format_key(prefix_bytes, "hex")
        print(f"{n} key-value pairs with prefix {prefix_str}:")
        print("─" * 100)
        count = 0
        for key, value in prefix_scan(db, prefix_bytes):
            if count >= n:
                break
            formatted_key = format_key(key, key_format)
            formatted_value = format_value(value, value_format)
            print(f"[{count + 1:2d}] Key:   {formatted_key}")
            print(f"     Value: {formatted_value}")
            print()
            count += 1
        if count == 0:
            print(f"No keys found with prefix {prefix_str}")


def db_get(db_path: str, key: Union[str, bytes], value_format: str = "auto") -> Optional[str]:
    """
    Get a specific key's value.

    Args:
        db_path: Path to database
        key: Key as bytes or string (e.g., "[2000] quick" for pivoted DB)
        value_format: Display format for value

    Returns:
        Formatted value string, or None if not found
    """
    # Normalize key to bytes
    key_bytes = normalize_key(key)

    with open_db(db_path, mode="r", profile="read:packed24") as db:
        try:
            result = db.get(key_bytes)
            if result is not None:
                return format_value(result, value_format)
            else:
                return None
        except Exception as e:
            return f"Error: {e}"


def db_sample(db_path: str, every_nth: int = 1000, n: int = 10,
              key_format: str = "auto", value_format: str = "auto"):
    """Sample every Nth key-value pair"""
    with open_db(db_path, mode="r", profile="read:packed24") as db:
        print(f"Sampling every {every_nth} keys (showing {n} samples):")
        print("─" * 100)
        count = 0
        samples_shown = 0
        for key, value in range_scan(db, b""):
            if count % every_nth == 0:
                if samples_shown >= n:
                    break
                formatted_key = format_key(key, key_format)
                formatted_value = format_value(value, value_format)
                print(f"[{samples_shown + 1:2d}] Position ~{count:,}")
                print(f"     Key:   {formatted_key}")
                print(f"     Value: {formatted_value}")
                print()
                samples_shown += 1
            count += 1
