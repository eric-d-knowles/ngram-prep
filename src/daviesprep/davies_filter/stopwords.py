"""Helper utilities for loading stopwords with language tracking."""

from typing import Tuple, Set

__all__ = ["load_stopwords"]


def load_stopwords(language: str) -> Tuple[Set[str], str]:
    """
    Load stopwords for a language using spaCy and return both the set and language code.

    Args:
        language: Language code for spaCy (e.g., "en", "ru", "de", etc.)

    Returns:
        Tuple of (stopword_set, language_code)

    Raises:
        ImportError: If spaCy is not installed
        ValueError: If the language is not supported by spaCy

    Example:
        >>> stop_set, stop_lang = load_stopwords("en")
        >>> filter_config = FilterConfig(
        ...     stop_set=stop_set,
        ...     stop_words_language=stop_lang,
        ...     ...
        ... )
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is required for stopword filtering. "
            "Install it with: pip install spacy"
        )

    try:
        nlp = spacy.blank(language)
        stopword_set = set(nlp.Defaults.stop_words)
    except Exception as e:
        raise ValueError(
            f"Unsupported stopword language: '{language}'. "
            f"spaCy may not support this language."
        ) from e

    return stopword_set, language
