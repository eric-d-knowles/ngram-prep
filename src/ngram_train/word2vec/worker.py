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

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Attach to gensim logger
    gensim_logger = logging.getLogger("gensim")
    gensim_logger.handlers.clear()
    gensim_logger.setLevel(logging.INFO)
    gensim_logger.addHandler(file_handler)

    return logger


def train_model(year, db_path, model_dir, log_dir, weight_by, vector_size,
                window, min_count, approach, epochs, workers, allow_unk=True,
                max_unk_count=None, load_into_memory=False, shuffle=False,
                random_seed=42):
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
        allow_unk (bool): Whether to allow ngrams with <UNK> tokens.
        max_unk_count (int): Maximum number of <UNK> tokens per ngram.
        load_into_memory (bool): If True, load ngrams into memory before training.
        shuffle (bool): If True, shuffle ngrams before each epoch.
        random_seed (int): Random seed for shuffling.
    """
    # Set process title for monitoring
    if setproctitle is not None:
        try:
            setproctitle(f"ngt:y{year}_vs{vector_size}_w{window}")
        except Exception:
            pass  # Silently continue if setproctitle fails

    sg = 1 if approach == 'skip-gram' else 0

    name_string = (
        f"y{year}_wb{weight_by}_vs{vector_size}_w{window}_"
        f"mc{min_count}_sg{sg}_e{epochs}"
    )

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
            f"allow_unk={allow_unk}, max_unk_count={max_unk_count}, "
            f"load_into_memory={load_into_memory}, shuffle={shuffle}..."
        )

        if load_into_memory:
            logger.info(f"Loading year {year} into memory...")

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
            allow_unk=allow_unk,
            max_unk_count=max_unk_count,
            load_into_memory=load_into_memory,
            shuffle=shuffle,
            random_seed=random_seed
        )

        model_filename = f"w2v_{name_string}.kv"
        model_save_path = os.path.join(model_dir, model_filename)
        model.wv.save(model_save_path)

        logger.info(f"Model for year {year} saved to {model_save_path}.")
    except Exception as e:
        logger.error(f"Error training model for year {year}: {e}", exc_info=True)
