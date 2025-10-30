"""
Lemmatizer wrapper for spaCy.

Provides a simple .lemmatize(word, pos) interface compatible with the Cython filter pipeline.
"""
import spacy
from spacy.tokens import Doc
from functools import lru_cache


# Default spaCy models by language
DEFAULT_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "el": "el_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
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
        self.nlp.initialize()

    def lemmatize(self, word: str, pos: str = "NOUN") -> str:
        """
        Lemmatize a single word given its POS tag.

        Uses a hybrid approach:
        1. Try lookup table first (avoids "other" → "oth" errors)
        2. If word unchanged, fall back to rule-based lemmatization

        Args:
            word: The word to lemmatize
            pos: spaCy POS tag (NOUN, VERB, ADJ, ADV, etc.)

        Returns:
            The lemmatized form of the word

        Note: This creates a Doc for each word, which has overhead.
        For better performance, consider using a caching wrapper or
        pre-lemmatizing a vocabulary.
        """
        # Try lookup first
        doc = Doc(self.nlp.vocab, words=[word])
        doc = self.nlp.get_pipe("lemmatizer_lookup")(doc)
        lookup_lemma = doc[0].lemma_

        # If lookup found a lemma (changed from original), use it
        if lookup_lemma != word:
            return lookup_lemma

        # Otherwise, fall back to rule-based lemmatization with POS tag
        doc = Doc(self.nlp.vocab, words=[word], pos=[pos])
        # Set morphology to avoid incorrect suffix stripping on edge cases
        if pos == "ADJ":
            doc[0].set_morph("Degree=Pos")
        elif pos == "VERB":
            doc[0].set_morph("VerbForm=Inf")
        doc = self.rule_lemmatizer(doc)
        return doc[0].lemma_


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
        # Try lookup first
        doc = Doc(self.nlp.vocab, words=[word])
        doc = self.nlp.get_pipe("lemmatizer_lookup")(doc)
        lookup_lemma = doc[0].lemma_

        # If lookup found a lemma, use it
        if lookup_lemma != word:
            return lookup_lemma

        # Fall back to rule-based with morphology
        doc = Doc(self.nlp.vocab, words=[word], pos=[pos])
        if pos == "ADJ":
            doc[0].set_morph("Degree=Pos")
        elif pos == "VERB":
            doc[0].set_morph("VerbForm=Inf")
        doc = self.rule_lemmatizer(doc)
        return doc[0].lemma_

    def lemmatize(self, word: str, pos: str = "NOUN") -> str:
        """Lemmatize with caching."""
        return self._cached_lemmatize(word, pos)

    def cache_info(self):
        """Return cache statistics."""
        return self._cached_lemmatize.cache_info()

    def clear_cache(self):
        """Clear the lemmatization cache."""
        self._cached_lemmatize.cache_clear()
