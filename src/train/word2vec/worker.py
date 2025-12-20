"""Worker functions for parallel Word2Vec model training."""

import logging
import os

try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = None

from .model import train_word2vec

__all__ = ["train_model", "configure_logging"]


def configure_logging(log_dir, filename):
    """
    Configure and return a logger for a child process, adding Gensim's logs.

    Args:
        log_dir (str): Directory to store log files.
        filename (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, filename)
    logger_name = os.path.splitext(filename)[0]

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Configure root logger for this worker process
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Clear any existing handlers from parent process
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Configure all gensim loggers
    for gensim_module in ["gensim", "gensim.models.word2vec", "gensim.models.base_any2vec", "gensim.utils"]:
        gensim_logger = logging.getLogger(gensim_module)
        gensim_logger.setLevel(logging.INFO)

    return logger


def train_model(year, db_path, model_dir, log_dir, weight_by, vector_size,
                window, min_count, approach, epochs, workers, unk_mode='reject',
                cache_corpus=False, use_corpus_file=True, corpus_file_path=None,
                temp_dir=None, debug_sample=0, debug_interval=0):
    """
    Train a Word2Vec model for a specific year from RocksDB.

    Args:
        year (int): Year to train on.
        db_path (str): Path to the pivoted RocksDB.
        model_dir (str): Directory to save trained models.
        log_dir (str): Directory to save log files.
        weight_by (str): Weighting strategy ("freq", "doc_freq", or "none").
        vector_size (int): Size of word vectors.
        window (int): Context window size.
        min_count (int): Minimum frequency of words to include.
        approach (str): Training approach ("skip-gram" or "CBOW").
        epochs (int): Number of training epochs.
        workers (int): Number of worker threads.
        unk_mode (str): How to handle <UNK> tokens. One of:
            - 'reject': Discard entire n-gram if it contains any <UNK> (default)
            - 'strip': Remove <UNK> tokens, keep if ≥2 tokens remain
            - 'retain': Keep n-grams as-is, including <UNK> tokens
        cache_corpus (bool): If True, load corpus into memory for faster multi-epoch training.
                            Only applies when use_corpus_file=False.
        use_corpus_file (bool): If True, use corpus_file parameter for training (enables
                               better multi-core scaling). Default: True.
        corpus_file_path (str): Optional path to pre-created corpus file (shared mode).
        temp_dir (str): Optional directory for temporary corpus file (for HPC scratch space).
        debug_sample (int): If > 0, print first N sentences for debugging
        debug_interval (int): If > 0, print one sample every N seconds (overrides debug_sample)
    """
    sg = 1 if approach == 'skip-gram' else 0

    name_string = (
        f"y{year}_wb{weight_by}_vs{vector_size:03d}_w{window:03d}_"
        f"mc{min_count:03d}_sg{sg}_e{epochs:03d}"
    )

    # Set process title for monitoring
    if setproctitle is not None:
        try:
            setproctitle(f"ngt:{name_string}")
        except Exception:
            pass  # Silently continue if setproctitle fails

    logger = configure_logging(
        log_dir,
        filename=f"w2v_{name_string}.log"
    )

    # Check if database exists
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return

    os.makedirs(model_dir, exist_ok=True)

    try:
        logger.info(
            f"Processing year {year} with parameters: "
            f"vector_size={vector_size}, window={window}, "
            f"min_count={min_count}, sg={sg}, epochs={epochs}, "
            f"unk_mode={unk_mode}, cache_corpus={cache_corpus}..."
        )

        model = train_word2vec(
            db_path=db_path,
            year=year,
            weight_by=weight_by,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=workers,
            unk_mode=unk_mode,
            cache_corpus=cache_corpus,
            use_corpus_file=use_corpus_file,
            corpus_file_path=corpus_file_path,
            temp_dir=temp_dir,
            debug_sample=debug_sample,
            debug_interval=debug_interval
        )

        model_filename = f"w2v_{name_string}.kv"
        model_save_path = os.path.join(model_dir, model_filename)
        model.wv.save(model_save_path)

        logger.info(f"Model for year {year} saved to {model_save_path}.")
    except Exception as e:
        logger.error(f"Error training model for year {year}: {e}", exc_info=True)
        raise  # Re-raise exception so main process knows the task failed
