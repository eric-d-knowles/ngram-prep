"""Parser for Google Ngrams data format."""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)

__all__ = ["YearFreq", "NgramRecord", "parse_line"]


def _combine_bigrams_in_ngram(ngram_text: str, combined_bigrams: set) -> str:
    """
    Replace consecutive token pairs matching combined_bigrams with hyphenated versions.

    Scans the n-gram tokens from left to right, looking for consecutive pairs
    that match entries in combined_bigrams. When found, replaces the pair with
    a hyphenated version. For POS-tagged data, strips tags before matching but
    preserves them in the output.

    Args:
        ngram_text: Space-separated n-gram tokens (e.g., "the working class in america")
                   or POS-tagged tokens (e.g., "the_DET working_VERB class_NOUN in_ADP america_PROPN")
        combined_bigrams: Set of bigrams to combine (e.g., {"working class"})

    Returns:
        Modified n-gram text with bigrams hyphenated (e.g., "the working-class in america")
        or with POS tags (e.g., "the_DET working-class_VERB in_ADP america_PROPN")

    Examples:
        >>> _combine_bigrams_in_ngram("the working class in america", {"working class"})
        'the working-class in america'
        >>> _combine_bigrams_in_ngram("lower class working class", {"lower class", "working class"})
        'lower-class working-class'
        >>> _combine_bigrams_in_ngram("the_DET working_VERB class_NOUN", {"working class"})
        'the_DET working-class_VERB'
    """
    tokens = ngram_text.split()

    # Scan for consecutive pairs matching combined_bigrams
    i = 0
    result_tokens = []

    while i < len(tokens):
        # Check if we have at least 2 tokens remaining
        if i + 1 < len(tokens):
            # Strip POS tags for matching (if present)
            # POS tags are indicated by underscore suffix (e.g., "working_VERB")
            token1 = tokens[i]
            token2 = tokens[i+1]

            # Extract base forms (without POS tags) for matching
            base1 = token1.rsplit('_', 1)[0] if '_' in token1 else token1
            base2 = token2.rsplit('_', 1)[0] if '_' in token2 else token2

            # Form potential bigram from base forms
            bigram = f"{base1} {base2}"

            # If bigram matches, combine and skip next token
            if bigram in combined_bigrams:
                # Preserve POS tag from first token if present
                if '_' in token1:
                    result_tokens.append(f"{base1}-{base2}_{token1.rsplit('_', 1)[1]}")
                else:
                    result_tokens.append(f"{base1}-{base2}")
                i += 2  # Skip both tokens
                continue

        # No match, keep token as-is
        result_tokens.append(tokens[i])
        i += 1

    return " ".join(result_tokens)


class YearFreq(TypedDict):
    """Year-level frequency data for an n-gram."""
    year: int
    frequency: int
    document_count: int


class NgramRecord(TypedDict):
    """Complete n-gram record with all year frequencies."""
    frequencies: List[YearFreq]


def parse_line(
        line: str,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
        combined_bigrams: Optional[set] = None,
) -> Tuple[Optional[str], Optional[NgramRecord]]:
    """
    Parse Google Ngrams line format into structured data.

    Format: "ngram\\tYEAR,FREQ,DOC\\tYEAR,FREQ,DOC..."

    Args:
        line: Raw line from ngrams file
        filter_pred: Optional predicate to filter n-grams by text
                     (applied before parsing frequency data)
        combined_bigrams: Optional set of bigrams to combine with hyphens
                         (e.g., {"working class", "middle class"})
                         When found within an n-gram, consecutive tokens matching
                         a bigram will be replaced with a hyphenated version,
                         reducing the n-gram size by 1.

    Returns:
        (ngram_text, record) if valid and passes filter, else (None, None)

    Notes:
        - Returns (None, None) for empty/malformed lines or filtered n-grams
        - Skips malformed frequency tuples silently
        - Requires at least one valid frequency tuple to succeed
        - If combined_bigrams is provided, matching consecutive token pairs
          within the n-gram will be hyphenated (e.g., "the working class in america"
          becomes "the working-class in america")

    Examples:
        >>> parse_line("hello\\t2000,100,50\\t2001,150,60")
        ('hello', {'frequencies': [
            {'year': 2000, 'frequency': 100, 'document_count': 50},
            {'year': 2001, 'frequency': 150, 'document_count': 60}
        ]})
        >>> parse_line("the working class in america\\t2000,100,50", combined_bigrams={"working class"})
        ('the working-class in america', {'frequencies': [{'year': 2000, 'frequency': 100, 'document_count': 50}]})
    """
    s = line.strip()
    if not s:
        return None, None

    # Split into n-gram text and frequency data
    parts = s.split("\t", 1)
    if len(parts) != 2:
        return None, None

    ngram_text, freq_blob = parts

    # Apply combined bigrams transformation: collapse matching consecutive tokens
    if combined_bigrams is not None:
        ngram_text = _combine_bigrams_in_ngram(ngram_text, combined_bigrams)

    # Apply filter before parsing frequency data (optimization)
    if filter_pred is not None and not filter_pred(ngram_text):
        return None, None

    frequencies: List[YearFreq] = []

    # Parse frequency tuples: "year,frequency,document_count"
    for entry in freq_blob.split("\t"):
        parts = entry.split(",")
        if len(parts) != 3:
            continue

        try:
            frequencies.append({
                "year": int(parts[0]),
                "frequency": int(parts[1]),
                "document_count": int(parts[2]),
            })
        except ValueError:
            # Skip malformed numeric values
            continue

    # Only return if we got at least one valid frequency
    if not frequencies:
        return None, None

    return ngram_text, {"frequencies": frequencies}