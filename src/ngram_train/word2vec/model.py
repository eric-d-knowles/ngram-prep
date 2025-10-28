"""Word2Vec model training and data loading."""

import random
from itertools import repeat
from math import log, floor

from gensim.models import Word2Vec

from ngram_prep.ngram_pivot.stream import stream_year_ngrams

__all__ = [
    "SentencesIterable",
    "train_word2vec",
]


def calculate_weight(freq, base=10):
    """
    Calculate the weight of an n-gram using logarithmic scaling.

    Args:
        freq (int): Raw frequency of the n-gram.
        base (float): Logarithm base for scaling.

    Returns:
        int: Scaled weight.
    """
    return max(1, floor(log(freq + 1, base)))


def create_unk_filter(unk_mode='reject'):
    """
    Create a filter function for handling <UNK> tokens in ngrams.

    Args:
        unk_mode (str): How to handle <UNK> tokens. One of:
            - 'reject': Discard entire n-gram if it contains any <UNK> (default)
            - 'strip': Remove <UNK> tokens, keep if ≥2 tokens remain (handled in iterator)
            - 'retain': Keep n-grams as-is, including <UNK> tokens

    Returns:
        callable or None: Filter function for stream_year_ngrams, or None if no filtering.
    """
    if unk_mode == 'reject':
        def filter_fn(ngram_bytes):
            return b'<UNK>' not in ngram_bytes
        return filter_fn
    elif unk_mode == 'strip':
        # Strip mode filtering happens in the iterator, not here
        # We need to pass all n-grams through so they can be processed
        return None
    elif unk_mode == 'retain':
        return None  # No filtering
    else:
        raise ValueError(f"Invalid unk_mode: '{unk_mode}'. Must be 'reject', 'strip', or 'retain'.")


class SentencesIterable:
    """
    An iterable wrapper for sentences generated from RocksDB pivoted ngram database.
    Supports streaming, loading into memory, and optional shuffling.
    """

    def __init__(self, db_path, year, weight_by="freq", log_base=10,
                 unk_mode='reject', load_into_memory=False, shuffle=False, random_seed=42):
        """
        Initialize the iterable.

        Args:
            db_path (str): Path to RocksDB database.
            year (int): Year to load data for.
            weight_by (str): Weighting strategy ("freq", "doc_freq", or "none").
            log_base (int): Base for logarithmic weighting.
            unk_mode (str): How to handle <UNK> tokens. One of:
                - 'reject': Discard entire n-gram if it contains any <UNK> (default)
                - 'strip': Remove <UNK> tokens, keep if ≥2 tokens remain
                - 'retain': Keep n-grams as-is, including <UNK> tokens
            load_into_memory (bool): If True, load all ngrams into memory.
                                    If False, stream from database each iteration.
            shuffle (bool): If True, shuffle ngrams before each epoch.
                           Only applies when load_into_memory=True.
            random_seed (int): Random seed for shuffling (only used if shuffle=True).
        """
        self.db_path = db_path
        self.year = year
        self.weight_by = weight_by
        self.log_base = log_base
        self.unk_mode = unk_mode
        self.ngram_filter = create_unk_filter(unk_mode)
        self.load_into_memory = load_into_memory
        self.shuffle = shuffle
        self.random_seed = random_seed

        # Validate parameters
        if shuffle and not load_into_memory:
            raise ValueError("shuffle=True requires load_into_memory=True")

        # Cache for loaded data (only used if load_into_memory=True)
        self._ngrams = None
        self._total_examples = None
        self._total_words = None

    def _load_ngrams(self):
        """Load all ngrams into memory with weighting applied."""
        if self._ngrams is not None:
            return  # Already loaded

        ngrams = []
        total_examples = 0
        total_words = 0

        ngram_stream = stream_year_ngrams(
            self.db_path,
            self.year,
            ngram_filter=self.ngram_filter
        )

        # No individual progress bars - just load silently
        for ngram_bytes, occurrences, documents in ngram_stream:
            ngram_str = ngram_bytes.decode('utf-8', errors='replace')
            ngram_tokens = ngram_str.split()

            # Handle strip mode: remove <UNK> tokens and check minimum length
            if self.unk_mode == 'strip':
                ngram_tokens = [token for token in ngram_tokens if token != '<UNK>']
                if len(ngram_tokens) < 2:
                    continue  # Skip unigrams and empty n-grams

            # Apply weighting strategy
            if self.weight_by == "freq":
                weight = calculate_weight(occurrences, base=self.log_base)
            elif self.weight_by == "doc_freq":
                weight = calculate_weight(documents, base=self.log_base)
            else:
                weight = 1

            # Add weighted copies
            for _ in range(weight):
                ngrams.append(ngram_tokens)

            total_examples += weight
            total_words += len(ngram_tokens) * weight

        self._ngrams = ngrams
        self._total_examples = total_examples
        self._total_words = total_words

    def __iter__(self):
        """
        Iterate over ngrams.

        Three modes:
        1. Stream from DB (load_into_memory=False): Consecutive repetition of weighted ngrams
        2. Load without shuffle (load_into_memory=True, shuffle=False): Fixed order, no I/O after load
        3. Load and shuffle (load_into_memory=True, shuffle=True): Randomized order each epoch
        """
        if self.load_into_memory:
            # Load data into memory if not already loaded
            self._load_ngrams()

            if self.shuffle:
                # Shuffle before each epoch
                shuffled = self._ngrams.copy()  # Shallow copy (just pointers)
                random.seed(self.random_seed)
                random.shuffle(shuffled)
                # No progress bar during training epochs - gensim shows its own
                for ngram_tokens in shuffled:
                    yield ngram_tokens
            else:
                # Iterate in original loaded order
                # No progress bar during training epochs - gensim shows its own
                for ngram_tokens in self._ngrams:
                    yield ngram_tokens
        else:
            # Streaming mode (original behavior)
            ngram_stream = stream_year_ngrams(
                self.db_path,
                self.year,
                ngram_filter=self.ngram_filter
            )

            # No progress bar during streaming - gensim shows its own training progress
            for ngram_bytes, occurrences, documents in ngram_stream:
                ngram_str = ngram_bytes.decode('utf-8', errors='replace')
                ngram_tokens = ngram_str.split()

                # Handle strip mode: remove <UNK> tokens and check minimum length
                if self.unk_mode == 'strip':
                    ngram_tokens = [token for token in ngram_tokens if token != '<UNK>']
                    if len(ngram_tokens) < 2:
                        continue  # Skip unigrams and empty n-grams

                # Apply weighting strategy
                if self.weight_by == "freq":
                    weight = calculate_weight(occurrences, base=self.log_base)
                    yield from repeat(ngram_tokens, weight)
                elif self.weight_by == "doc_freq":
                    weight = calculate_weight(documents, base=self.log_base)
                    yield from repeat(ngram_tokens, weight)
                else:
                    yield ngram_tokens

    @property
    def total_examples(self):
        """Get total number of examples (for Word2Vec training)."""
        if self.load_into_memory:
            if self._ngrams is None:
                self._load_ngrams()
            return self._total_examples
        else:
            return None  # Not available in streaming mode

    @property
    def total_words(self):
        """Get total number of words (for Word2Vec training)."""
        if self.load_into_memory:
            if self._ngrams is None:
                self._load_ngrams()
            return self._total_words
        else:
            return None  # Not available in streaming mode


def train_word2vec(
        db_path,
        year,
        weight_by,
        vector_size,
        window,
        min_count,
        sg,
        workers,
        epochs,
        unk_mode='reject',
        load_into_memory=False,
        shuffle=False,
        random_seed=42,
        **kwargs
):
    """
    Train a Word2Vec model on the given sentences from RocksDB.

    Args:
        db_path (str): Path to the pivoted RocksDB containing ngrams.
        year (int): Year to train on.
        weight_by (str): Weighting strategy ("freq", "doc_freq", or "none").
        vector_size (int): Size of word vectors.
        window (int): Context window size.
        min_count (int): Minimum frequency of words to include.
        sg (int): Training algorithm (1=skip-gram, 0=CBOW).
        workers (int): Number of worker threads.
        epochs (int): Number of training epochs.
        unk_mode (str): How to handle <UNK> tokens. One of:
            - 'reject': Discard entire n-gram if it contains any <UNK> (default)
            - 'strip': Remove <UNK> tokens, keep if ≥2 tokens remain
            - 'retain': Keep n-grams as-is, including <UNK> tokens
        load_into_memory (bool): If True, load ngrams into memory before training.
        shuffle (bool): If True, shuffle ngrams before each epoch (requires load_into_memory=True).
        random_seed (int): Random seed for shuffling.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    sentences = SentencesIterable(
        db_path, year, weight_by=weight_by, log_base=10, unk_mode=unk_mode,
        load_into_memory=load_into_memory, shuffle=shuffle, random_seed=random_seed
    )

    if load_into_memory:
        # Build vocabulary separately for better control
        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=workers,
            **kwargs
        )

        model.build_vocab(sentences)

        # Note: Data is now loaded in memory (triggered by build_vocab iteration)
        # Vocab stats logged by gensim

        # Train with proper corpus statistics
        model.train(
            sentences,
            total_examples=sentences.total_examples,
            total_words=sentences.total_words,
            epochs=epochs
        )

        return model
    else:
        # Streaming mode: use original single-call approach
        return Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=workers,
            epochs=epochs,
            **kwargs
        )
