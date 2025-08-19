# ngram_prep/utils/filters.py
from __future__ import annotations
from typing import Callable

DEFAULT_TAGS: tuple[str, ...] = (
    "NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET",
    "ADP", "NUM", "CONJ", "X", ".",
)


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

    def is_tagged(tok: str) -> bool:
        return tok.endswith(suffixes)

    if ngram_type == "all":
        return lambda line: bool(line) and not line.isspace()

    if ngram_type == "tagged":
        return lambda line: (
            (toks := line.split(" ")) != [] and all(is_tagged(t) for t in toks)
        )

    if ngram_type == "untagged":
        return lambda line: (
            (toks := line.split(" ")) != [] and
            all(not is_tagged(t) for t in toks)
        )

    raise ValueError("ngram_type must be 'tagged', 'untagged', or 'all'.")
