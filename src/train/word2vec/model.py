"""Word2Vec model training and data loading."""

import os
import random
import tempfile
import time
from itertools import repeat
from math import log, floor

from gensim.models import Word2Vec
from setproctitle import setproctitle

from ngramprep.ngram_pivot.stream import stream_year_ngrams

__all__ = [
    "SentencesIterable",
    "train_word2vec",
    "create_corpus_file",
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
    Streams data from the database each epoch, or caches in memory for faster repeated access.
    """

    def __init__(self, db_path, year, weight_by="freq", log_base=10, unk_mode='reject',
                 debug_sample=0, debug_interval=0, cache_in_memory=False):
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
            cache_in_memory (bool): If True, load entire corpus into memory on first iteration.
                                   Subsequent epochs reuse cached data. Default: False (stream mode).
        """
        self.db_path = db_path
        self.year = year
        self.weight_by = weight_by
        self.log_base = log_base
        self.unk_mode = unk_mode
        self.debug_sample = debug_sample
        self.debug_interval = debug_interval
        self.cache_in_memory = cache_in_memory
        self.ngram_filter = create_unk_filter(unk_mode)
        self._cached_sentences = None  # Will hold cached corpus if cache_in_memory=True

    def __iter__(self):
        """
        Stream ngrams from database, applying weighting on the fly.
        If cache_in_memory=True, loads entire corpus on first iteration and reuses it.
        """
        # If caching is enabled and we already have cached data, yield from cache
        if self.cache_in_memory and self._cached_sentences is not None:
            yield from self._cached_sentences
            return

        # Otherwise, stream from database (and optionally build cache)
        ngram_stream = stream_year_ngrams(
            self.db_path,
            self.year,
            ngram_filter=self.ngram_filter
        )

        debug_count = 0
        last_debug_time = time.time() if self.debug_interval > 0 else None

        # If caching, collect sentences in a list
        if self.cache_in_memory:
            cache_list = []

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
                if self.cache_in_memory:
                    # Cache the weighted sentences
                    for _ in range(weight):
                        cache_list.append(ngram_tokens)
                else:
                    yield from repeat(ngram_tokens, weight)
            elif self.weight_by == "doc_freq":
                weight = calculate_weight(documents, base=self.log_base)
                if self.cache_in_memory:
                    for _ in range(weight):
                        cache_list.append(ngram_tokens)
                else:
                    yield from repeat(ngram_tokens, weight)
            else:
                if self.cache_in_memory:
                    cache_list.append(ngram_tokens)
                else:
                    yield ngram_tokens

        # If caching, store the cache and yield from it
        if self.cache_in_memory:
            self._cached_sentences = cache_list
            yield from self._cached_sentences


def create_corpus_file(db_path, year, weight_by, unk_mode='reject',
                       temp_dir=None, log_base=10):
    """
    Create a corpus file from RocksDB for a specific year.

    This file can be shared across multiple Word2Vec training runs for the same year,
    reducing memory pressure and disk I/O.

    Args:
        db_path (str): Path to the pivoted RocksDB containing ngrams.
        year (int): Year to extract corpus for.
        weight_by (str): Weighting strategy ("freq", "doc_freq", or "none").
        unk_mode (str): How to handle <UNK> tokens ('reject', 'strip', or 'retain').
        temp_dir (str): Directory for corpus file (e.g., '/scratch'). If None, uses system default.
        log_base (int): Base for logarithmic weighting.

    Returns:
        str: Path to created corpus file.
    """
    import logging

    # Set process title for monitoring
    setproctitle(f"ngt:writer_y{year}_wb{weight_by}")

    sentences = SentencesIterable(
        db_path=db_path,
        year=year,
        weight_by=weight_by,
        log_base=log_base,
        unk_mode=unk_mode,
        cache_in_memory=False,
        debug_sample=0,
        debug_interval=0
    )

    # Create persistent temp file (caller responsible for cleanup)
    temp_fd, temp_file_path = tempfile.mkstemp(
        suffix='.txt',
        prefix=f'w2v_corpus_y{year}_wb{weight_by}_',
        dir=temp_dir,
        text=True
    )

    # Write corpus to file in LineSentence format
    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(' '.join(sentence) + '\n')

    logging.getLogger("gensim.models.word2vec").info(
        f"Shared corpus file created: {temp_file_path}"
    )

    return temp_file_path


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
        cache_corpus=False,
        use_corpus_file=True,
        corpus_file_path=None,
        temp_dir=None,
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
        cache_corpus (bool): If True, load entire corpus into memory for faster training
                            across multiple epochs. Only applies when use_corpus_file=False.
                            Default: False (stream from disk each epoch).
        use_corpus_file (bool): If True, use corpus_file parameter for training. Enables better
                               multi-core scaling (32+ workers) by bypassing GIL. If False, use
                               iterator-based approach (optimal for 8-12 workers). Default: True.
        corpus_file_path (str): Optional path to pre-created corpus file. If provided, this file
                               is used directly (shared mode). If None and use_corpus_file=True,
                               creates a temporary corpus file (isolated mode). Default: None.
        temp_dir (str): Optional directory for temporary corpus file (only used if corpus_file_path
                       is None). Useful for HPC scratch space (e.g., '/scratch'). Default: None.
        debug_sample (int): If > 0, print first N sentences for debugging
        debug_interval (int): If > 0, print one sample every N seconds (overrides debug_sample)

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    import logging

    # Ensure gensim logging is verbose (in case it gets reset)
    logging.getLogger("gensim").setLevel(logging.DEBUG)
    logging.getLogger("gensim.models.word2vec").setLevel(logging.DEBUG)
    logging.getLogger("gensim.models.base_any2vec").setLevel(logging.DEBUG)

    sentences = SentencesIterable(
        db_path=db_path,
        year=year,
        weight_by=weight_by,
        log_base=10,
        unk_mode=unk_mode,
        cache_in_memory=cache_corpus if not use_corpus_file else False,
        debug_sample=debug_sample,
        debug_interval=debug_interval
    )

    if use_corpus_file:
        # Corpus file mode: use pre-created file or create temp file
        if corpus_file_path:
            # Shared mode: use pre-created corpus file (no cleanup needed)
            logging.getLogger("gensim.models.word2vec").info(
                f"Using shared corpus file: {corpus_file_path}"
            )
            model = Word2Vec(
                corpus_file=corpus_file_path,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                sg=sg,
                workers=workers,
                epochs=epochs,
                compute_loss=True,
                **kwargs
            )
        else:
            # Isolated mode: create and clean up temporary corpus file
            temp_file_path = None
            try:
                # Create temporary file in specified directory
                temp_fd, temp_file_path = tempfile.mkstemp(
                    suffix='.txt',
                    prefix=f'w2v_corpus_y{year}_',
                    dir=temp_dir,
                    text=True
                )

                # Write corpus to file in LineSentence format
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    for sentence in sentences:
                        f.write(' '.join(sentence) + '\n')

                logging.getLogger("gensim.models.word2vec").info(
                    f"Corpus written to temporary file: {temp_file_path}"
                )

                # Train using corpus_file parameter
                model = Word2Vec(
                    corpus_file=temp_file_path,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    sg=sg,
                    workers=workers,
                    epochs=epochs,
                    compute_loss=True,
                    **kwargs
                )
            finally:
                # Clean up temp file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        logging.getLogger("gensim.models.word2vec").info(
                            f"Temporary corpus file removed: {temp_file_path}"
                        )
                    except Exception as e:
                        logging.getLogger("gensim.models.word2vec").warning(
                            f"Failed to remove temporary file {temp_file_path}: {e}"
                        )
    else:
        # Iterator mode: traditional approach (optimal for 8-12 workers)
        model = Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=workers,
            epochs=epochs,
            compute_loss=True,
            **kwargs
        )

    return model
