"""Sentence tokenization for Davies corpus text."""
from __future__ import annotations

from typing import Iterator, List, Optional, Set
import re

__all__ = [
    "tokenize_sentences",
    "simple_sentence_tokenizer",
    "combine_bigrams_in_tokens",
]


def combine_bigrams_in_tokens(tokens: List[str], combined_bigrams: Set[str]) -> List[str]:
    """
    Replace consecutive token pairs matching combined_bigrams with hyphenated versions.

    Scans the token list from left to right, looking for consecutive pairs
    that match entries in combined_bigrams (case-insensitive). When found,
    replaces the pair with a hyphenated version.

    Args:
        tokens: List of tokens (e.g., ["the", "working", "class", "family"])
        combined_bigrams: Set of bigrams to combine (e.g., {"working class", "middle class"})
                         Note: Bigrams in the set should be lowercase with spaces

    Returns:
        Modified token list with bigrams hyphenated (e.g., ["the", "working-class", "family"])

    Examples:
        >>> combine_bigrams_in_tokens(["the", "working", "class"], {"working class"})
        ['the', 'working-class']
        >>> combine_bigrams_in_tokens(["lower", "class", "working", "class"], {"lower class", "working class"})
        ['lower-class', 'working-class']
    """
    if not combined_bigrams or len(tokens) < 2:
        return tokens

    result = []
    i = 0

    while i < len(tokens):
        # Check if we have at least 2 tokens remaining
        if i + 1 < len(tokens):
            # Form potential bigram (lowercase for matching)
            bigram = f"{tokens[i].lower()} {tokens[i+1].lower()}"

            # If bigram matches, combine with hyphen
            if bigram in combined_bigrams:
                # Preserve original casing in first token
                result.append(f"{tokens[i]}-{tokens[i+1]}")
                i += 2  # Skip both tokens
                continue

        # No match, keep token as-is
        result.append(tokens[i])
        i += 1

    return result


def simple_sentence_tokenizer(
    text: str,
    combined_bigrams: Optional[Set[str]] = None
) -> Iterator[List[str]]:
    """
    Simple sentence tokenizer for Davies corpus text.

    Splits on sentence-ending punctuation (., !, ?) and tokenizes
    each sentence into words.

    Args:
        text: Raw text to tokenize
        combined_bigrams: Optional set of bigrams to combine with hyphens
                         (e.g., {"working class", "middle class"})

    Yields:
        Lists of tokens (words) for each sentence

    Example:
        >>> text = "Hello world. This is a test!"
        >>> list(simple_sentence_tokenizer(text))
        [['Hello', 'world'], ['This', 'is', 'a', 'test']]
        >>> text = "The working class family. The middle class home."
        >>> list(simple_sentence_tokenizer(text, {"working class", "middle class"}))
        [['The', 'working-class', 'family'], ['The', 'middle-class', 'home']]
    """
    # Split into sentences on sentence-ending punctuation
    # Keep the punctuation for now
    sentences = re.split(r'([.!?]+)', text)

    # Combine sentences with their punctuation
    current_sentence = ""
    for i, part in enumerate(sentences):
        if re.match(r'[.!?]+', part):
            # This is punctuation, add to current sentence
            current_sentence += part
            # Yield the sentence
            if current_sentence.strip():
                tokens = tokenize_sentence(current_sentence, combined_bigrams=combined_bigrams)
                if len(tokens) >= 2:  # Require at least 2 tokens
                    yield tokens
            current_sentence = ""
        else:
            # This is text, accumulate
            current_sentence += part

    # Don't forget the last sentence if it doesn't end with punctuation
    if current_sentence.strip():
        tokens = tokenize_sentence(current_sentence, combined_bigrams=combined_bigrams)
        if len(tokens) >= 2:
            yield tokens


def tokenize_sentence(
    sentence: str,
    combined_bigrams: Optional[Set[str]] = None
) -> List[str]:
    """
    Tokenize a single sentence into words.

    Uses simple whitespace splitting with some cleanup.
    Filters out COHA markup symbols like @ and other standalone punctuation.

    Args:
        sentence: Sentence text
        combined_bigrams: Optional set of bigrams to combine with hyphens
                         (e.g., {"working class", "middle class"})

    Returns:
        List of tokens

    Example:
        >>> tokenize_sentence("Hello, world!")
        ['Hello', 'world']
        >>> tokenize_sentence("The working class family", {"working class"})
        ['The', 'working-class', 'family']
    """
    # Simple whitespace tokenization
    # Remove leading/trailing whitespace and split
    tokens = sentence.strip().split()

    # Remove empty tokens and clean up
    tokens = [t for t in tokens if t]

    # Remove standalone punctuation and COHA markup symbols (@)
    tokens = [t for t in tokens if not re.match(r'^[.!?,;:"\'\-@]+$', t)]

    # Apply bigram combination if specified
    if combined_bigrams:
        tokens = combine_bigrams_in_tokens(tokens, combined_bigrams)

    return tokens


def tokenize_sentences(
    text: str,
    min_tokens: int = 2,
    combined_bigrams: Optional[Set[str]] = None
) -> Iterator[List[str]]:
    """
    Tokenize text into sentences with word tokens.

    This is the main entry point for tokenization.

    Args:
        text: Raw text to tokenize
        min_tokens: Minimum number of tokens per sentence
        combined_bigrams: Optional set of bigrams to combine with hyphens
                         (e.g., {"working class", "middle class"})

    Yields:
        Lists of tokens for each sentence

    Example:
        >>> text = "The cat sat. On the mat!"
        >>> list(tokenize_sentences(text))
        [['The', 'cat', 'sat'], ['On', 'the', 'mat']]
        >>> text = "The working class family. The middle class home."
        >>> list(tokenize_sentences(text, combined_bigrams={"working class", "middle class"}))
        [['The', 'working-class', 'family'], ['The', 'middle-class', 'home']]
    """
    for tokens in simple_sentence_tokenizer(text, combined_bigrams=combined_bigrams):
        if len(tokens) >= min_tokens:
            yield tokens
