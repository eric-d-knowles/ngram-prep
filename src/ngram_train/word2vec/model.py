"""Word2Vec model training and data loading."""

import random
import time
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
    Streams data from the database each epoch.
    """

    def __init__(self, db_path, year, weight_by="freq", log_base=10, unk_mode='reject', debug_sample=0, debug_interval=0):
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
            debug_sample (int): If > 0, print first N sentences for debugging
            debug_interval (int): If > 0, print one sample every N seconds (overrides debug_sample)
        """
        self.db_path = db_path
        self.year = year
        self.weight_by = weight_by
        self.log_base = log_base
        self.unk_mode = unk_mode
        self.debug_sample = debug_sample
        self.debug_interval = debug_interval
        self.ngram_filter = create_unk_filter(unk_mode)

    def __iter__(self):
        """Stream ngrams from database, applying weighting on the fly."""
        ngram_stream = stream_year_ngrams(
            self.db_path,
            self.year,
            ngram_filter=self.ngram_filter
        )

        debug_count = 0
        last_debug_time = time.time() if self.debug_interval > 0 else None

        for ngram_bytes, occurrences, documents in ngram_stream:
            ngram_str = ngram_bytes.decode('utf-8', errors='replace')
            ngram_tokens = ngram_str.split()

            # Handle strip mode: remove <UNK> tokens and check minimum length
            if self.unk_mode == 'strip':
                original_tokens = ngram_tokens.copy()
                ngram_tokens = [token for token in ngram_tokens if token != '<UNK>']
                if len(ngram_tokens) < 2:
                    continue  # Skip unigrams and empty n-grams

                # Debug logging - interval mode (time-based)
                if self.debug_interval > 0:
                    current_time = time.time()
                    if original_tokens != ngram_tokens and (current_time - last_debug_time) >= self.debug_interval:
                        print(f"[STRIP] {original_tokens} -> {ngram_tokens}")
                        last_debug_time = current_time
                # Debug logging - sample mode (first N)
                elif self.debug_sample > 0 and debug_count < self.debug_sample:
                    if original_tokens != ngram_tokens:
                        print(f"[STRIP] {original_tokens} -> {ngram_tokens}")
                    debug_count += 1

            # Apply weighting strategy
            if self.weight_by == "freq":
                weight = calculate_weight(occurrences, base=self.log_base)
                yield from repeat(ngram_tokens, weight)
            elif self.weight_by == "doc_freq":
                weight = calculate_weight(documents, base=self.log_base)
                yield from repeat(ngram_tokens, weight)
            else:
                yield ngram_tokens


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
        debug_sample=0,
        debug_interval=0,
        **kwargs
):
    """
    Train a Word2Vec model on streaming sentences from RocksDB.

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
        debug_sample (int): If > 0, print first N sentences for debugging
        debug_interval (int): If > 0, print one sample every N seconds (overrides debug_sample)

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    sentences = SentencesIterable(
        db_path=db_path,
        year=year,
        weight_by=weight_by,
        log_base=10,
        unk_mode=unk_mode,
        debug_sample=debug_sample,
        debug_interval=debug_interval
    )

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
