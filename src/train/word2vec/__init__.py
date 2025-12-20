"""Word2Vec training module for Google N-grams."""

from .train import train_models, build_word2vec_models, transfer_models
from .model import SentencesIterable, train_word2vec, create_corpus_file
from .worker import train_model, configure_logging
from .config import ensure_iterable, construct_model_path, set_info
from .display import print_training_header, print_completion_banner, LINE_WIDTH
from .w2v_model import W2VModel
from .evaluate import evaluate_models, evaluate_word2vec_models
from .align import normalize_and_align_models
from .visualize import plot_evaluation_results
from .regression_analysis import run_regression_analysis, plot_regression_results, get_model_summary

__all__ = [
    # Main entry points
    "train_models",
    "build_word2vec_models",
    "transfer_models",
    "evaluate_models",
    "evaluate_word2vec_models",
    "normalize_and_align_models",
    "plot_evaluation_results",
    # Regression analysis
    "run_regression_analysis",
    "plot_regression_results",
    "get_model_summary",
    # Model training
    "SentencesIterable",
    "train_word2vec",
    "create_corpus_file",
    # Model analysis
    "W2VModel",
    # Worker functions
    "train_model",
    "configure_logging",
    # Configuration utilities
    "ensure_iterable",
    "construct_model_path",
    "set_info",
    # Display utilities
    "print_training_header",
    "print_completion_banner",
    "LINE_WIDTH",
]
