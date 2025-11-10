from __future__ import annotations
from functools import partial
from typing import Callable

DEFAULT_TAGS: tuple[str, ...] = (
    "NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET",
    "ADP", "NUM", "CONJ", "X", ".",
)


def _is_tagged(tok: str, suffixes: tuple[str, ...]) -> bool:
    return tok.endswith(suffixes)


def _pred_all_nonempty(line: str) -> bool:
    # Preserve existing behavior for "all": any non-empty, non-whitespace line
    return bool(line) and not line.isspace()


def _pred_tagged(line: str, *, suffixes: tuple[str, ...]) -> bool:
    toks = line.split(" ")
    return toks != [] and all(_is_tagged(t, suffixes) for t in toks)


def _pred_untagged(line: str, *, suffixes: tuple[str, ...]) -> bool:
    toks = line.split(" ")
    return toks != [] and all(not _is_tagged(t, suffixes) for t in toks)


def make_ngram_type_predicate(
    ngram_type: str,
    tags: tuple[str, ...] = DEFAULT_TAGS,
) -> Callable[[str], bool]:
    """
    Return predicate(line) -> bool for 'tagged' | 'untagged' | 'all'.

    Assumptions
    -----------
    - Tokens are separated by single spaces.
    - A token is 'tagged' iff it ends with '_<TAG>' and <TAG> ∈ tags.
      (Base may be any string, including underscores; e.g., '__NOUN' is tagged.)
      Examples:
        '__NOUN'      -> tagged
        'cat_VERB'    -> tagged
        '_NOUN_'      -> NOT tagged (tag not at token end)
    """
    suffixes = tuple(f"_{t}" for t in tags)

    if ngram_type == "all":
        return _pred_all_nonempty

    if ngram_type == "tagged":
        # functools.partial is picklable when the underlying function and args are picklable
        return partial(_pred_tagged, suffixes=suffixes)

    if ngram_type == "untagged":
        return partial(_pred_untagged, suffixes=suffixes)

    raise ValueError("ngram_type must be 'tagged', 'untagged', or 'all'.")