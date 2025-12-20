# daviesprep/davies_filter/whitelist.py
"""Whitelist generation and loading for Davies filtered corpus."""

from __future__ import annotations

import heapq
import struct
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Iterator, List, Tuple, Union, Generator, Optional

from tqdm import tqdm

try:
    import numpy as np  # optional fast path
except Exception:
    np = None  # type: ignore[assignment]

try:
    import enchant
except ImportError:
    enchant = None  # type: ignore[assignment]

import rocks_shim as rs
from ngramprep.common_db.api import open_db

FMT_PACKED24 = "<QQQ"        # (year, count, volume_count), 24 bytes - packed24 format
FMT_PIVOT = "<QQ"            # (count, volume_count), 16 bytes - pivoted format
TUPLE_SIZE_PACKED24 = 24
TUPLE_SIZE_PIVOT = 16
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
    profile when available (profile="read:packed24" for Davies data).
    """
    if hasattr(db_or_path, "iterator") and callable(getattr(db_or_path, "iterator")):
        # Assume it's a rocks-shim DB-like object
        yield db_or_path  # type: ignore[misc]
    else:
        path = str(Path(db_or_path))
        with open_db(path, mode="r", profile="read:packed24") as db:
            yield db


def _iter_db_items(db: "rs.DB") -> Iterator[Tuple[bytes, bytes]]:
    """
    Yield (key_bytes, value_bytes) for all non-metadata entries using rocks-shim iterator.
    """
    it = db.iterator()
    it.seek(b"")
    while it.valid():
        k = it.key()
        if not k.startswith(METADATA_PREFIX):
            yield k, it.value()
        it.next()


def _extract_tokens_from_key(key_bytes: bytes, track_genre: bool = False) -> List[bytes]:
    """
    Extract individual tokens from a Davies database key.

    Args:
        key_bytes: Raw key from database
        track_genre: If True, key format is [genre:4][year:8][sentence]
                    If False, key format is [year:4][sentence] (pivoted format, year is 4 bytes big-endian)

    Returns:
        List of token bytes (space-separated words from sentence)
    """
    if track_genre:
        # Genre (4 bytes) + Year (8 bytes little-endian) + Sentence
        if len(key_bytes) < 12:
            return []
        sentence = key_bytes[12:]
    else:
        # Year (4 bytes big-endian) + Sentence (pivoted format)
        if len(key_bytes) < 4:
            return []
        sentence = key_bytes[4:]

    # Split sentence into tokens (space-separated)
    return sentence.split(b' ')


def _total_occurrences_struct(value_bytes: bytes) -> int:
    """Sum occurrence counts using struct.iter_unpack (portable path).

    Supports both formats:
    - 16 bytes: (count, volume_count) - pivoted format
    - 24 bytes: (year, count, volume_count) - packed24 format
    """
    if len(value_bytes) == TUPLE_SIZE_PIVOT:
        # Pivoted format: 16 bytes = (count, volume_count)
        count, _vol = struct.unpack(FMT_PIVOT, value_bytes)
        return count
    elif len(value_bytes) >= TUPLE_SIZE_PACKED24 and len(value_bytes) % TUPLE_SIZE_PACKED24 == 0:
        # Packed24 format: multiple 24-byte records
        tot = 0
        for (_year, count, _vol) in struct.iter_unpack(FMT_PACKED24, value_bytes):
            tot += count
        return tot
    else:
        return 0


def _total_occurrences_numpy_optimized(value_bytes: bytes) -> int:
    """Optimized NumPy version with better memory handling."""
    if len(value_bytes) < TUPLE_SIZE:
        return 0


def _total_occurrences_numpy_optimized(value_bytes: bytes) -> int:
    """Optimized NumPy version with better memory handling.

    Supports both formats:
    - 16 bytes: (count, volume_count) - pivoted format
    - 24 bytes: (year, count, volume_count) - packed24 format
    """
    if len(value_bytes) == TUPLE_SIZE_PIVOT:
        # Pivoted format: 16 bytes = (count, volume_count)
        arr = np.frombuffer(value_bytes, dtype=np.uint64, count=2)
        return int(arr[0])  # count is first element
    elif len(value_bytes) >= TUPLE_SIZE_PACKED24 and len(value_bytes) % TUPLE_SIZE_PACKED24 == 0:
        # Packed24 format
        n_tuples = len(value_bytes) // TUPLE_SIZE_PACKED24
        arr = np.frombuffer(value_bytes, dtype=np.uint64, count=n_tuples * 3)
        # count is at indices 1, 4, 7, 10, ... = 1::3
        return int(arr[1::3].sum())
    else:
        return 0


def _total_occurrences(value_bytes: bytes) -> int:
    """Sum occurrence counts with optimized NumPy path."""
    if np is not None:
        try:
            return _total_occurrences_numpy_optimized(value_bytes)
        except Exception:
            # Safety net: if numpy dtype/view fails for any reason, fall back
            pass
    return _total_occurrences_struct(value_bytes)


def _total_occurrences_in_range(key_bytes: bytes, value_bytes: bytes, year_range: Tuple[int, int], track_genre: bool) -> int:
    """Get occurrence count if the sentence year is within the specified range.

    For Davies corpora, each key represents a single sentence in a single year.
    Year is extracted from the key, not the value.

    Args:
        key_bytes: Key bytes from database (contains year)
        value_bytes: Value bytes from database (contains count)
        year_range: (start_year, end_year) inclusive range
        track_genre: If True, key format is [genre:4][year:8][tokens]; else [year:4][tokens]

    Returns:
        Occurrence count if year is in range, 0 otherwise
    """
    start_year, end_year = year_range

    # Extract year from key
    # Key format: [genre (4 bytes)][year (8 bytes little-endian)][tokens] if track_genre
    #          or [year (4 bytes big-endian)][tokens] if not track_genre
    if track_genre:
        # Genre-aware format: skip 4-byte genre, read 8-byte year (little-endian)
        if len(key_bytes) < 12:
            return 0
        year = struct.unpack('<Q', key_bytes[4:12])[0]
    else:
        # Year-only format: 4-byte year (big-endian)
        if len(key_bytes) < 4:
            return 0
        year = struct.unpack('>I', key_bytes[0:4])[0]

    # Check if year is in range
    if not (start_year <= year <= end_year):
        return 0

    # Extract count from value
    # Value is 24 bytes: (year, occurrences, documents)
    if len(value_bytes) == 24:
        _year, occurrences, _documents = struct.unpack('<QQQ', value_bytes)
        return occurrences
    elif len(value_bytes) == 16:
        # Old format fallback: (occurrences, documents)
        occurrences, _documents = struct.unpack('<QQ', value_bytes)
        return occurrences
    else:
        return 0


def _maybe_decode(b: bytes, decode: bool) -> Union[str, bytes]:
    return b.decode(DECODING, "replace") if decode else b


def _create_spell_checker(language: str = "en_US") -> Optional["enchant.Dict"]:
    """Create a spell checker for the given language.

    Returns None if enchant is not available (spell checking will be disabled).
    Raises an exception if enchant is available but the language is not.
    """
    if enchant is None:
        return None
    try:
        return enchant.Dict(language)
    except enchant.errors.DictNotFoundError as e:
        raise ValueError(
            f"Spell checking language '{language}' not found. "
            f"Install it with: enchant.broker.list_dicts() to see available languages."
        ) from e
    except Exception as e:
        # For other enchant errors, raise with more context
        raise ValueError(
            f"Failed to create spell checker for language '{language}': {e}"
        ) from e


def _is_correctly_spelled(word: str, spell_checker: Optional["enchant.Dict"], alpha_only: bool = False) -> bool:
    """Check if a word is correctly spelled.

    Args:
        word: Token to check
        spell_checker: Enchant dictionary for spell checking (None disables spell checking)
        alpha_only: If True, reject non-alphabetic tokens (matching alpha_only filter setting)

    Returns:
        True if token should be included in whitelist, False otherwise
    """
    if spell_checker is None:
        return True  # No spell checking available, accept all words

    # If alpha_only filter was used, reject non-alphabetic tokens
    # (they should have been filtered out, so shouldn't be in whitelist)
    if alpha_only and not word.isalpha():
        return False

    # For pure alphabetic words, check spelling
    if word.isalpha():
        return spell_checker.check(word)

    # For mixed tokens when alpha_only=False, accept without spell checking
    return True


def _check_year_coverage(value_bytes: bytes, year_range: tuple[int, int]) -> bool:
    """Check if a sentence appears in all years in the specified range.

    Args:
        value_bytes: Packed records (16 or 24 bytes per record)
        year_range: (start_year, end_year) inclusive range

    Returns:
        True if sentence appears in all years within range, False otherwise

    Note: Year range filtering only works with packed24 format (24-byte values).
          For pivoted format (16-byte values), year info is in the key, not value.
    """
    # Pivoted format doesn't have year in value - can't filter by year range
    if len(value_bytes) == TUPLE_SIZE_PIVOT:
        return False

    # Packed24 format
    if len(value_bytes) < TUPLE_SIZE_PACKED24:
        return False

    start_year, end_year = year_range
    required_years = set(range(start_year, end_year + 1))
    years_present = set()

    for i in range(0, len(value_bytes), TUPLE_SIZE_PACKED24):
        if i + 8 > len(value_bytes):
            break
        year = struct.unpack('<Q', value_bytes[i:i+8])[0]
        years_present.add(year)
        # Early exit if we've found all required years
        if required_years.issubset(years_present):
            return True

    return required_years.issubset(years_present)


def _process_batch_for_frequencies(
        batch: List[Tuple[bytes, bytes]],
        track_genre: bool,
        genre_focus: Optional[List[str]],
        spell_check: bool,
        spell_check_language: str,
        alpha_only: bool,
        year_range: Optional[Tuple[int, int]],
        always_include: Optional[set[bytes]],
) -> Tuple[Counter, Optional[defaultdict]]:
    """Process a batch of sentences and count token frequencies.

    Args:
        batch: List of (key, value) tuples from database
        track_genre: If True, keys include genre prefix
        genre_focus: If set, only process these genres
        spell_check: Whether to filter by spell checking
        spell_check_language: Language code for spell checking
        alpha_only: If True, reject non-alphabetic tokens in spell checking
        year_range: Optional year range filter
        always_include: Tokens to always include

    Returns:
        (token_occurrences, token_year_sets) where token_year_sets is None if no year_range
    """
    spell_checker = _create_spell_checker(spell_check_language) if spell_check else None

    if year_range is not None:
        token_year_sets = defaultdict(set)
        token_occurrences = Counter()
        years_in_corpus = set()
    else:
        token_occurrences = Counter()
        token_year_sets = None
        years_in_corpus = None

    for k, v in batch:
        # Extract year and optionally genre from key
        if track_genre:
            # Genre-aware format: [genre:4][year:8][tokens]
            if len(k) < 12:
                continue
            genre = k[:4].rstrip(b'\x00').decode('utf-8')
            year = struct.unpack('<Q', k[4:12])[0]

            # Skip if genre filtering is enabled and this genre is not in focus
            if genre_focus is not None and genre not in genre_focus:
                continue
        else:
            # Year-only format: [year:4][tokens]
            if len(k) < 4:
                continue
            year = struct.unpack('>I', k[0:4])[0]

        # Extract tokens from the sentence key
        tokens = _extract_tokens_from_key(k, track_genre=track_genre)

        # Count occurrences for this sentence
        try:
            if year_range is not None:
                # Count only occurrences within the year range
                occurrences = _total_occurrences_in_range(k, v, year_range, track_genre)
                # Skip if no occurrences in range (year outside range)
                if occurrences == 0:
                    continue
            else:
                # No year range filter, count all occurrences
                occurrences = _total_occurrences(v)
                year = None
        except Exception:
            continue

        # Add each token's occurrences to counter
        for token in tokens:
            # Skip UNK tokens
            if token == b'<UNK>':
                continue

            # Apply spell check filter if enabled (but bypass for always_include tokens)
            if spell_check and (always_include is None or token not in always_include):
                word_str = token.decode(DECODING, "replace")
                if not _is_correctly_spelled(word_str, spell_checker, alpha_only):
                    continue

            if year_range is not None:
                # Track which years this token appears in
                token_year_sets[token].add(year)
                token_occurrences[token] += occurrences
                years_in_corpus.add(year)
            else:
                token_occurrences[token] += occurrences

    # Return year sets data if year_range is specified
    if year_range is not None:
        return token_occurrences, (token_year_sets, years_in_corpus)
    else:
        return token_occurrences, None


def _build_token_frequencies(
        db_or_path: Union[str, Path, "rs.DB"],
        *,
        track_genre: bool = False,
        genre_focus: Optional[List[str]] = None,
        spell_check: bool = False,
        spell_check_language: str = "en_US",
        alpha_only: bool = False,
        year_range: Optional[tuple[int, int]] = None,
        always_include: Optional[set[bytes]] = None,
        workers: int = 1,
        batch_size: int = 50_000,
) -> Counter:
    """Build frequency counter for all tokens in the Davies database.

    Args:
        db_or_path: Database or path to database
        track_genre: Whether keys include genre prefix
        genre_focus: If set, only process these genres (e.g., ["fic", "mag"])
        spell_check: If True, only include correctly spelled words
        spell_check_language: Language for spell checking (default: en_US)
        alpha_only: If True, reject non-alphabetic tokens when spell checking
        year_range: Optional (start_year, end_year) tuple - only include tokens that appear in ALL years in range
        always_include: Optional set of token bytes to always include, bypassing spell check filter
        workers: Number of parallel workers (default: 1 for sequential)
        batch_size: Number of sentences per batch for parallel processing

    Returns:
        Counter mapping token bytes to total occurrence counts
    """
    # Warn if spell checking was requested but enchant is unavailable
    if spell_check:
        spell_checker = _create_spell_checker(spell_check_language)
        if spell_checker is None:
            import logging
            logging.warning(
                "Spell checking was requested but enchant library is not available. "
                "All words will be included regardless of spelling. "
                "Install enchant C library to enable spell checking."
            )

    # Read all data into batches
    batches: List[List[Tuple[bytes, bytes]]] = []
    current_batch: List[Tuple[bytes, bytes]] = []

    with _db_from(db_or_path) as db:
        print("Scanning database...")
        for k, v in _iter_db_items(db):
            current_batch.append((k, v))
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

    total_sentences = sum(len(batch) for batch in batches)
    print(f"Found {total_sentences:,} sentences in {len(batches)} batches")

    if year_range is not None:
        print("Detecting years present in corpus...")

    # Process batches in parallel
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Aggregate results from workers
        token_occurrences = Counter()
        token_year_sets = defaultdict(set) if year_range is not None else None
        years_in_corpus = set() if year_range is not None else None

        with ProcessPoolExecutor(max_workers=workers) as executor:
            with tqdm(
                total=len(batches),
                desc="Building token frequencies",
                unit="batches",
                ncols=100,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ) as pbar:
                # Submit all batches
                futures = {
                    executor.submit(
                        _process_batch_for_frequencies,
                        batch,
                        track_genre,
                        genre_focus,
                        spell_check,
                        spell_check_language,
                        alpha_only,
                        year_range,
                        always_include
                    ): batch_id
                    for batch_id, batch in enumerate(batches)
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        batch_occurrences, year_data = future.result()

                        # Merge token occurrences
                        token_occurrences.update(batch_occurrences)

                        # Merge year data if tracking years
                        if year_range is not None and year_data is not None:
                            batch_year_sets, batch_years = year_data
                            for token, years in batch_year_sets.items():
                                token_year_sets[token].update(years)
                            years_in_corpus.update(batch_years)
                    except Exception as e:
                        print(f"\nError processing batch: {e}")
                    finally:
                        pbar.update(1)
    else:
        # Sequential processing (workers=1)
        token_occurrences = Counter()
        token_year_sets = defaultdict(set) if year_range is not None else None
        years_in_corpus = set() if year_range is not None else None

        with tqdm(
            total=len(batches),
            desc="Building token frequencies",
            unit="batches",
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as pbar:
            for batch in batches:
                batch_occurrences, year_data = _process_batch_for_frequencies(
                    batch, track_genre, genre_focus, spell_check, spell_check_language, alpha_only, year_range, always_include
                )

                # Merge token occurrences
                token_occurrences.update(batch_occurrences)

                # Merge year data if tracking years
                if year_range is not None and year_data is not None:
                    batch_year_sets, batch_years = year_data
                    for token, years in batch_year_sets.items():
                        token_year_sets[token].update(years)
                    years_in_corpus.update(batch_years)

                pbar.update(1)

    # If year_range specified, filter out tokens that don't appear in ALL years
    if year_range is not None:
        # Determine which years in the range are actually present in the corpus
        start_year, end_year = year_range
        required_years = years_in_corpus & set(range(start_year, end_year + 1))

        if required_years:
            sorted_years = sorted(required_years)
            print(f"\nYears present in corpus within range: {len(required_years)} years")
            print(f"  Range: {min(sorted_years)} to {max(sorted_years)}")
            if len(sorted_years) <= 20:
                print(f"  Years: {sorted_years}")
            else:
                print(f"  Sample: {sorted_years[:5]} ... {sorted_years[-5:]}")

        print(f"\nFiltering tokens by year coverage (must appear in all {len(required_years)} years)...")
        counter = Counter()
        tokens_before = len(token_occurrences)
        for token, total_count in token_occurrences.items():
            if required_years.issubset(token_year_sets[token]):
                counter[token] = total_count
        tokens_after = len(counter)
        print(f"Tokens before year filter: {tokens_before:,}")
        print(f"Tokens after year filter:  {tokens_after:,}")
        print(f"Tokens removed:            {tokens_before - tokens_after:,}")
        return counter
    else:
        return token_occurrences


def write_whitelist(
        db_or_path: Union[str, Path, "rs.DB"],
        dest: Union[str, Path],
        *,
        top: int | None = None,
        track_genre: bool = False,
        genre_focus: Optional[List[str]] = None,
        spell_check: bool = False,
        spell_check_language: str = "en_US",
        alpha_only: bool = False,
        year_range: Optional[tuple[int, int]] = None,
        always_include: Optional[set[bytes]] = None,
        workers: int = 1,
        batch_size: int = 50_000,
) -> Path:
    """
    Write a plain TXT file of tokens ranked by total frequency (desc).
    Each line: <token><tab><total_occurrences>

    Args:
        db_or_path: Database or path to Davies filtered database
        dest: Output file path
        top: Optional limit to top N tokens
        track_genre: Whether database keys include genre prefix (4 bytes)
        genre_focus: If set, only process these genres (e.g., ["fic", "mag"])
        spell_check: If True, only include correctly spelled words
        spell_check_language: Language for spell checking (default: en_US)
        alpha_only: If True, reject non-alphabetic tokens (should match filter setting used)
        year_range: Optional (start_year, end_year) tuple - only include tokens from sentences present in all years
        always_include: Optional set of token bytes to always include in whitelist,
                       regardless of frequency (e.g., {b"working-class", b"nuclear"})
        workers: Number of parallel workers for frequency counting (default: 1 for sequential)
        batch_size: Number of sentences per batch for parallel processing

    Returns:
        Resolved path to the created whitelist file
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    # Build token frequency counter
    counter = _build_token_frequencies(
        db_or_path,
        track_genre=track_genre,
        genre_focus=genre_focus,
        spell_check=spell_check,
        spell_check_language=spell_check_language,
        alpha_only=alpha_only,
        year_range=year_range,
        always_include=always_include,
        workers=workers,
        batch_size=batch_size,
    )

    print(f"\nRanking {len(counter):,} unique tokens...")

    # Add always_include tokens to counter if not already present
    # (this should rarely happen now that spell check bypasses them, but keep as safety net)
    if always_include:
        added_count = 0
        for token in always_include:
            if token not in counter:
                # Add with count of 0 (will be sorted to bottom, but still included)
                counter[token] = 0
                added_count += 1
        if added_count > 0:
            print(f"Added {added_count} always_include tokens that were not found in corpus")

    # Get most common tokens
    if top is not None:
        items = counter.most_common(top)

        # Ensure always_include tokens are included even if they didn't make top N
        if always_include:
            included_tokens = {token for token, _ in items}
            missing_tokens = always_include - included_tokens
            if missing_tokens:
                # Add missing always_include tokens at the end
                for token in missing_tokens:
                    items.append((token, counter.get(token, 0)))
                print(f"Added {len(missing_tokens)} always_include tokens that didn't make top {top}")

        print(f"Selected top {top:,} tokens (+ {len(missing_tokens) if always_include and missing_tokens else 0} always_include)")
    else:
        items = counter.most_common()
        print(f"Writing all {len(items):,} tokens")

    # Write to file
    print(f"Writing whitelist to {dest}...")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        for token, total in items:
            if isinstance(token, bytes):
                token_str = token.decode("utf-8", "backslashreplace")
            else:
                token_str = token
            f.write(f"{token_str}\t{total}\n")

    tmp.replace(dest)
    print(f"Whitelist written successfully: {dest.resolve()}")
    return dest.resolve()


def load_whitelist(
        whitelist_path: Union[str, Path],
        *,
        top_n: int | None = None,
        min_count: int = 1,
        encoding: str = "utf-8",
) -> set[bytes]:
    """
    Load whitelist from TSV file created by write_whitelist().

    Args:
        whitelist_path: Path to whitelist file (token<tab>count format)
        top_n: Optional limit to top N most frequent tokens
        min_count: Minimum frequency threshold (tokens below this are excluded)
        encoding: File encoding (default: utf-8)

    Returns:
        Set of bytes tokens for use in process_sentence(whitelist=...)
    """
    whitelist_path = Path(whitelist_path)
    if not whitelist_path.exists():
        raise FileNotFoundError(f"Whitelist file not found: {whitelist_path}")

    def parse_line(line: str) -> bytes | None:
        """Parse a line and return token bytes if valid, None otherwise."""
        line = line.rstrip("\r\n")
        if not line:
            return None

        parts = line.split("\t", 1)
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

        # Convert to bytes (matching the pipeline's data type)
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
