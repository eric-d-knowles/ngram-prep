"""Helper utilities for loading stopwords with language tracking."""

from typing import Tuple, Set

__all__ = ["load_stopwords"]


def load_stopwords(language: str) -> Tuple[Set[str], str]:
    """
    Load stopwords for a language and return both the set and language name.
    
    This ensures a single source of truth for the stopword language,
    avoiding the need to specify the language twice.
    
    Args:
        language: Language code for stop-words package. Supported languages:
                 "arabic" (ar), "bulgarian" (bg), "catalan" (ca), "czech" (cz),
                 "danish" (da), "german" (de), "english" (en), "spanish" (es),
                 "finnish" (fi), "french" (fr), "hindi" (hi), "hungarian" (hu),
                 "indonesian" (id), "italian" (it), "norwegian" (nb), "dutch" (nl),
                 "polish" (pl), "portuguese" (pt), "romanian" (ro), "russian" (ru),
                 "slovak" (sk), "swedish" (sv), "turkish" (tr), "ukrainian" (uk),
                 "vietnamese" (vi)
    
    Returns:
        Tuple of (stopword_set, language_name)
        
    Raises:
        ValueError: If the language is not supported
        
    Example:
        >>> stop_set, stop_lang = load_stopwords("russian")
        >>> filter_config = FilterConfig(
        ...     stop_set=stop_set,
        ...     stop_words_language=stop_lang,
        ...     ...
        ... )
    """
    try:
        from stop_words import get_stop_words, LANGUAGE_MAPPING
    except ImportError:
        raise ImportError(
            "stop-words package is required for stopword filtering. "
            "Install it with: pip install stop-words"
        )
    
    try:
        stopword_set = set(get_stop_words(language))
    except Exception as e:
        # Get list of supported languages
        supported = sorted(LANGUAGE_MAPPING.keys())
        raise ValueError(
            f"Unsupported stopword language: '{language}'. "
            f"Supported languages: {', '.join(supported)}"
        ) from e
    
    return stopword_set, language
