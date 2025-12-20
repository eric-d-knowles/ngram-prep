"""
Lemmatizer wrapper for spaCy.

Provides a simple .lemmatize(word, pos) interface compatible with the Cython filter pipeline.
"""
import logging
import spacy
from spacy.tokens import Doc
from functools import lru_cache


# Default spaCy models by language
DEFAULT_MODELS = {
    "en": "en_core_web_sm",      # English
    "de": "de_core_news_sm",     # German (ger)
    "es": "es_core_news_sm",     # Spanish (spa)
    "fr": "fr_core_news_sm",     # French (fre)
    "it": "it_core_news_sm",     # Italian (ita)
    "ru": "ru_core_news_sm",     # Russian (rus)
    "zh": "zh_core_web_sm",      # Chinese (chi-sim)
    "he": "he_core_news_sm",     # Hebrew (heb)
}


class SpacyLemmatizer:
    """
    Wrapper around spaCy's lemmatizer that provides a simple interface.

    Usage:
        # English (default)
        lemmatizer = SpacyLemmatizer()
        result = lemmatizer.lemmatize("running", pos="VERB")  # returns "run"

        # Other languages
        lemmatizer = SpacyLemmatizer(language="de")
        lemmatizer = SpacyLemmatizer(model="de_core_news_md")  # Custom model
    """

    def __init__(self, language=None, model=None):
        """
        Initialize the spaCy lemmatizer.

        Args:
            language: ISO 639-1 language code (e.g., "en", "de", "fr"). Defaults to "en".
            model: Explicit spaCy model name. Overrides language parameter if provided.
        """
        if model is None:
            lang = language or "en"
            if lang not in DEFAULT_MODELS:
                raise ValueError(
                    f"Language '{lang}' not supported. "
                    f"Supported languages: {', '.join(DEFAULT_MODELS.keys())}. "
                    f"Or specify a custom model with the 'model' parameter."
                )
            model = DEFAULT_MODELS[lang]

        self.nlp = spacy.load(model, disable=["parser", "ner", "textcat"])

        # Set up hybrid lemmatization: lookup + rule-based fallback
        # We keep the original rule-based lemmatizer and add a lookup one

        # Store reference to the original (rule-based) lemmatizer
        self.rule_lemmatizer = self.nlp.get_pipe("lemmatizer")

        # Remove it and add lookup-based lemmatizer as primary
        self.nlp.remove_pipe("lemmatizer")
        self.nlp.add_pipe("lemmatizer", name="lemmatizer_lookup", config={"mode": "lookup"})

        # Suppress spaCy's INFO messages during initialization
        spacy_logger = logging.getLogger("spacy")
        original_level = spacy_logger.level
        spacy_logger.setLevel(logging.WARNING)

        self.nlp.initialize()

        # Restore original logging level
        spacy_logger.setLevel(original_level)

    def lemmatize(self, word: str, pos: str = "NOUN") -> str:
        """
        Lemmatize a single word given its POS tag.

        Uses lookup table only to avoid morphological truncation errors.
        If word is not in lookup table, returns the word unchanged.

        Args:
            word: The word to lemmatize
            pos: spaCy POS tag (NOUN, VERB, ADJ, ADV, etc.)

        Returns:
            The lemmatized form of the word, or the original word if not in lookup

        Note: This creates a Doc for each word, which has overhead.
        For better performance, consider using a caching wrapper or
        pre-lemmatizing a vocabulary.
        """
        # Check if word is in lookup table
        lookup_table = self.nlp.get_pipe("lemmatizer_lookup").lookups.get_table("lemma_lookup")

        if word in lookup_table:
            # Word is in lookup table, use its result
            doc = Doc(self.nlp.vocab, words=[word])
            doc = self.nlp.get_pipe("lemmatizer_lookup")(doc)
            return doc[0].lemma_

        # Word not in lookup table, return unchanged to avoid morphological truncation
        # (e.g., "business" → "busines", "success" → "succes")
        return word


class CachedSpacyLemmatizer(SpacyLemmatizer):
    """
    Cached version of SpacyLemmatizer for better performance.

    Uses LRU cache to avoid re-lemmatizing the same (word, pos) pairs.
    Recommended for processing large corpora with repeated tokens.

    Usage:
        lemmatizer = CachedSpacyLemmatizer(cache_size=100000)
        result = lemmatizer.lemmatize("running", pos="VERB")  # cached
    """

    def __init__(self, language=None, model=None, cache_size=100000):
        """
        Initialize cached lemmatizer.

        Args:
            language: ISO 639-1 language code
            model: Explicit spaCy model name
            cache_size: Maximum number of (word, pos) pairs to cache
        """
        super().__init__(language=language, model=model)

        # Create cached version of _lemmatize_internal
        self._cached_lemmatize = lru_cache(maxsize=cache_size)(self._lemmatize_internal)

    def _lemmatize_internal(self, word: str, pos: str) -> str:
        """Internal lemmatization (gets cached)."""
        # Check if word is in lookup table
        lookup_table = self.nlp.get_pipe("lemmatizer_lookup").lookups.get_table("lemma_lookup")

        if word in lookup_table:
            # Word is in lookup table, use its result
            doc = Doc(self.nlp.vocab, words=[word])
            doc = self.nlp.get_pipe("lemmatizer_lookup")(doc)
            return doc[0].lemma_

        # Word not in lookup table, return unchanged to avoid morphological truncation
        return word

    def lemmatize(self, word: str, pos: str = "NOUN") -> str:
        """Lemmatize with caching."""
        return self._cached_lemmatize(word, pos)

    def cache_info(self):
        """Return cache statistics."""
        return self._cached_lemmatize.cache_info()

    def clear_cache(self):
        """Clear the lemmatization cache."""
        self._cached_lemmatize.cache_clear()
