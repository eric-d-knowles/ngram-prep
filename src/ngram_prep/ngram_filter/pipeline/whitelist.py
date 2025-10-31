# ngram_filter/pipeline/whitelist.py
from __future__ import annotations

import heapq
import struct
from collections import Counter
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Iterator, List, Tuple, Union, Generator, Optional

try:
    import numpy as np  # optional fast path
except Exception:
    np = None  # type: ignore[assignment]

try:
    import enchant
except ImportError:
    enchant = None  # type: ignore[assignment]

import rocks_shim as rs  # <-- rocks-shim only
from ngram_prep.common_db.api import open_db  # read profile via rocks-shim

FMT = "<QQQ"                 # (year, match_count, volume_count), little-endian u64s
TUPLE_SIZE = struct.calcsize(FMT)
METADATA_PREFIX = b"__"      # skip metadata keys like "__..."
DECODING = "utf-8"           # decoding for keys when decode=True

__all__ = [
    "write_whitelist",
    "load_whitelist",
]


@contextmanager
def _db_from(db_or_path: Union[str, Path, "rs.DB"]):
    """
    Yield an open rocks-shim DB. If a path is provided, open with read-optimized
    profile when available (profile="read").
    """
    if hasattr(db_or_path, "iterator") and callable(getattr(db_or_path, "iterator")):
        # Assume it's a rocks-shim DB-like object
        yield db_or_path  # type: ignore[misc]
    else:
        path = str(Path(db_or_path))
        # Use your rocks-shim context manager. If "read" is unsupported, shim can ignore it.
        with open_db(path, mode="r", profile="read") as db:
            yield db


def _iter_db_items(db: "rs.DB") -> Iterator[Tuple[bytes, bytes]]:
    """
    Yield (key_bytes, value_bytes) for all non-metadata entries using rocks-shim iterator.
    rocks-shim iterator interface: it.seek(lower), it.valid(), it.key(), it.value(), it.next()
    """
    it = db.iterator()
    # Seek to start: empty lower bound (b"") is the minimal byte prefix
    it.seek(b"")
    while it.valid():
        k = it.key()
        if not k.startswith(METADATA_PREFIX):
            yield k, it.value()
        it.next()


def _total_matches_struct(value_bytes: bytes) -> int:
    """Sum match_count using struct.iter_unpack (portable path)."""
    if len(value_bytes) < TUPLE_SIZE:
        return 0
    usable = (len(value_bytes) // TUPLE_SIZE) * TUPLE_SIZE
    tot = 0
    for (_year, match, _vol) in struct.iter_unpack(FMT, value_bytes[:usable]):
        tot += match
    return tot


def _total_matches_numpy_optimized(value_bytes: bytes) -> int:
    """Optimized NumPy version with better memory handling."""
    if len(value_bytes) < TUPLE_SIZE:
        return 0

    # Create view directly, no intermediate array
    n_tuples = len(value_bytes) // TUPLE_SIZE
    arr = np.frombuffer(value_bytes, dtype=np.uint64, count=n_tuples * 3)

    # Direct slice sum - no intermediate array creation
    return int(arr[1::3].sum())


def _total_matches(value_bytes: bytes) -> int:
    """Sum match counts with optimized NumPy path."""
    if np is not None:
        try:
            return _total_matches_numpy_optimized(value_bytes)
        except Exception:
            # Safety net: if numpy dtype/view fails for any reason, fall back
            pass
    return _total_matches_struct(value_bytes)


def _maybe_decode(b: bytes, decode: bool) -> Union[str, bytes]:
    return b.decode(DECODING, "replace") if decode else b


def _create_spell_checker(language: str = "en_US") -> Optional["enchant.Dict"]:
    """Create a spell checker for the given language."""
    if enchant is None:
        return None
    try:
        return enchant.Dict(language)
    except Exception:
        return None


def _is_correctly_spelled(word: str, spell_checker: Optional["enchant.Dict"]) -> bool:
    """Check if a word is correctly spelled."""
    if spell_checker is None:
        return True  # No spell checking available, accept all words

    # Skip non-alphabetic tokens (numbers, symbols, etc.)
    if not word.isalpha():
        return False

    return spell_checker.check(word)


def _iter_ranked_items(db_or_path: Union[str, Path, "rs.DB"], *, decode: bool = True) -> Generator[Tuple[Union[str, bytes], int], None, None]:
    """Generator that yields (key, total) sorted by total descending."""
    with _db_from(db_or_path) as db:
        # Use sorted() on the generator - more memory efficient than building intermediate list
        items = ((_maybe_decode(k, decode), _total_matches(v))
                 for k, v in _iter_db_items(db))
        yield from sorted(items, key=lambda kv: kv[1], reverse=True)


def _rank_all_counter(
        db_or_path: Union[str, Path, "rs.DB"],
        *,
        decode: bool = True,
        spell_check: bool = False,
        spell_check_language: str = "en_US",
) -> List[Tuple[Union[str, bytes], int]]:
    """Use Counter for better performance on frequency operations.

    Args:
        db_or_path: Database or path to database
        decode: Whether to decode bytes to strings
        spell_check: If True, only include correctly spelled words
        spell_check_language: Language for spell checking (default: en_US)
    """
    spell_checker = _create_spell_checker(spell_check_language) if spell_check else None
    counter = Counter()

    with _db_from(db_or_path) as db:
        for k, v in _iter_db_items(db):
            # Apply spell check filter if enabled
            if spell_check:
                word_str = k.decode(DECODING, "replace")
                if not _is_correctly_spelled(word_str, spell_checker):
                    continue

            counter[_maybe_decode(k, decode)] = _total_matches(v)
    return counter.most_common()


def _top_k_optimized(
        db_or_path: Union[str, Path, "rs.DB"],
        k: int,
        *,
        decode: bool = True,
        spell_check: bool = False,
        spell_check_language: str = "en_US",
) -> List[Tuple[Union[str, bytes], int]]:
    """Streaming top-K via min-heap (RAM ~ O(K)) with optimizations.

    Args:
        db_or_path: Database or path to database
        k: Number of top items to return
        decode: Whether to decode bytes to strings
        spell_check: If True, only include correctly spelled words in top K
        spell_check_language: Language for spell checking (default: en_US)
    """
    if k <= 0:
        return []

    spell_checker = _create_spell_checker(spell_check_language) if spell_check else None

    heap: List[Tuple[int, int, bytes]] = []  # (total, counter, key_bytes)
    counter = 0

    with _db_from(db_or_path) as db:
        for kbytes, vbytes in _iter_db_items(db):
            tot = _total_matches(vbytes)

            # Apply spell check filter if enabled
            if spell_check:
                word_str = kbytes.decode(DECODING, "replace")
                if not _is_correctly_spelled(word_str, spell_checker):
                    continue

            if len(heap) < k:
                heapq.heappush(heap, (tot, counter, kbytes))
                counter += 1
            elif tot > heap[0][0]:
                heapq.heapreplace(heap, (tot, counter, kbytes))
                counter += 1

    # Sort in descending order by total, then decode
    result = [(_maybe_decode(kb, decode), tot) for (tot, _cnt, kb) in
              sorted(heap, key=lambda x: x[0], reverse=True)]

    return result


def write_whitelist_streaming(
        db_or_path: Union[str, Path, "rs.DB"],
        dest: Union[str, Path],
        *,
        top: int | None = None,
        decode: bool = True,
        sep: str = "\t",
        spell_check: bool = False,
        spell_check_language: str = "en_US",
) -> Path:
    """
    Write a plain TXT file of tokens ranked by total frequency (desc) using streaming.
    Each line: <token><sep><total_matches>
    Memory usage: O(1) for unlimited output, O(K) for top-K

    Args:
        db_or_path: Database or path to database
        dest: Output file path
        top: Optional limit to top N tokens
        decode: Whether to decode bytes to strings
        sep: Separator between token and count
        spell_check: If True, only include correctly spelled words
        spell_check_language: Language for spell checking (default: en_US)
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="") as f:
        if top is not None:
            # Use optimized heap-based top-K for memory efficiency
            items = _top_k_optimized(
                db_or_path, top, decode=decode,
                spell_check=spell_check,
                spell_check_language=spell_check_language
            )
        else:
            # Use streaming approach with Counter for better performance
            items = _rank_all_counter(
                db_or_path, decode=decode,
                spell_check=spell_check,
                spell_check_language=spell_check_language
            )

        for token, total in items:
            if isinstance(token, bytes):
                token_str = token.decode("utf-8", "backslashreplace")
            else:
                token_str = token
            f.write(f"{token_str}{sep}{total}\n")

    tmp.replace(dest)
    return dest.resolve()


def write_whitelist(
        db_or_path: Union[str, Path, "rs.DB"],
        dest: Union[str, Path],
        *,
        top: int | None = None,
        decode: bool = True,
        sep: str = "\t",
        spell_check: bool = False,
        spell_check_language: str = "en_US",
) -> Path:
    """
    Write a plain TXT file of tokens ranked by total frequency (desc).
    Each line: <token><sep><total_matches>

    This is the main API - uses streaming implementation for better performance.

    Args:
        db_or_path: Database or path to database
        dest: Output file path
        top: Optional limit to top N tokens
        decode: Whether to decode bytes to strings
        sep: Separator between token and count
        spell_check: If True, only include correctly spelled words
        spell_check_language: Language for spell checking (default: en_US)
    """
    return write_whitelist_streaming(
        db_or_path, dest, top=top, decode=decode, sep=sep,
        spell_check=spell_check, spell_check_language=spell_check_language
    )


def load_whitelist_optimized(
        whitelist_path: Union[str, Path],
        *,
        top_n: int | None = None,
        min_count: int = 1,
        encoding: str = "utf-8",
        sep: str = "\t",
) -> set[bytes]:
    """
    Optimized whitelist loading using set comprehension and early termination.

    Args:
        whitelist_path: Path to whitelist file (token<sep>count format)
        top_n: Optional limit to top N most frequent tokens
        min_count: Minimum frequency threshold (tokens below this are excluded)
        encoding: File encoding (default: utf-8)
        sep: Separator between token and count (default: tab)

    Returns:
        Set of bytes tokens for use in process_tokens(whitelist=...)
    """
    whitelist_path = Path(whitelist_path)
    if not whitelist_path.exists():
        raise FileNotFoundError(f"Whitelist file not found: {whitelist_path}")

    def parse_line(line: str) -> bytes | None:
        """Parse a line and return token bytes if valid, None otherwise."""
        line = line.rstrip("\r\n")
        if not line:
            return None

        parts = line.split(sep, 1)
        if len(parts) != 2:
            return None

        token_str, count_str = parts
        try:
            frequency = int(count_str)
        except ValueError:
            return None

        # Apply frequency threshold
        if frequency < min_count:
            return None

        # Convert to bytes (matching your pipeline's data type)
        return token_str.encode(encoding, "surrogatepass")

    with open(whitelist_path, "r", encoding=encoding) as f:
        if top_n is not None:
            # Use islice for memory-efficient top-N processing
            tokens = {token for token in
                      (parse_line(line) for line in islice(f, top_n))
                      if token is not None}
        else:
            # Process entire file with set comprehension
            tokens = {token for token in
                      (parse_line(line) for line in f)
                      if token is not None}

    return tokens


def load_whitelist(
        whitelist_path: Union[str, Path],
        *,
        top_n: int | None = None,
        min_count: int = 1,
        encoding: str = "utf-8",
        sep: str = "\t",
) -> set[bytes]:
    """
    Load whitelist from TSV file created by write_whitelist().

    This is the main API - uses optimized implementation.
    """
    return load_whitelist_optimized(
        whitelist_path,
        top_n=top_n,
        min_count=min_count,
        encoding=encoding,
        sep=sep
    )