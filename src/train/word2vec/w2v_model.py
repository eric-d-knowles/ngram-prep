"""
W2VModel class for handling trained Word2Vec models.

Provides methods for intrinsic evaluation, normalization, vocabulary filtering,
and alignment using orthogonal Procrustes transforms.
"""

import os
import random

import numpy as np
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes

__all__ = ["W2VModel"]


class W2VModel:
    """
    A class for handling Word2Vec models stored as .kv files, with methods for
    intrinsic evaluation, normalization, vocabulary filtering, and alignment
    using orthogonal Procrustes transforms.
    """

    def __init__(self, model_path):
        """
        Initialize the W2VModel instance by loading the Word2Vec .kv file.

        Args:
            model_path (str): Path to the .kv file containing the Word2Vec model.

        Raises:
            FileNotFoundError: If the provided model_path does not exist.
            ValueError: If the file is not a valid .kv file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not model_path.endswith(".kv"):
            raise ValueError("The model file must be a .kv file.")

        self.model = KeyedVectors.load(model_path, mmap="r")
        self.vocab = set(self.model.index_to_key)
        self.vector_size = self.model.vector_size

    def evaluate(self, task, dataset_path):
        """
        Evaluate the model on a specified task (e.g., similarity or analogy).

        Args:
            task (str): The evaluation task ('similarity' or 'analogy').
            dataset_path (str): Path to the dataset file.

        Returns:
            float or dict: Evaluation results:
                - Similarity: Returns Spearman correlation as a float.
                - Analogy: Returns a dictionary of results (correct, total, accuracy).

        Raises:
            ValueError: If the task is not supported or the dataset is missing.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if task == "similarity":
            results = self.model.evaluate_word_pairs(dataset_path)
            return results[1][0]  # Spearman correlation

        elif task == "analogy":
            results = self.model.evaluate_word_analogies(dataset_path)
            return results[0]  # Analogy accuracy

        else:
            raise ValueError("Unsupported task. Choose 'similarity' or 'analogy'.")

    def normalize(self):
        """
        Normalize vectors in the model to unit length (L2 normalization).

        Returns:
            W2VModel: The instance itself, for method chaining.
        """
        norms = np.linalg.norm(self.model.vectors, axis=1, keepdims=True)
        self.model.vectors = self.model.vectors.copy() / norms
        return self

    def extract_vocab(self):
        """
        Extract the model's vocabulary.

        Returns:
            set: The vocabulary of the model as a set of words.
        """
        return self.vocab

    def filter_vocab(self, reference_vocab):
        """
        Filter the model's vocabulary to include only words in the reference vocabulary.

        Args:
            reference_vocab (set): A set of words representing the reference vocabulary.

        Returns:
            W2VModel: The instance itself, for method chaining.

        Raises:
            ValueError: If the reference vocabulary is not a set.
        """
        if not isinstance(reference_vocab, set):
            raise ValueError("reference_vocab must be a set of words.")

        shared_vocab = self.vocab.intersection(reference_vocab)
        self.filtered_vectors = {word: self.model[word] for word in shared_vocab}
        self.filtered_vocab = shared_vocab
        return self

    def align_to(self, reference_model, weights=None):
        """
        Align this model to a reference model using orthogonal Procrustes.

        Args:
            reference_model (W2VModel): The reference W2VModel instance to align to.
            weights (dict, optional): Mapping of word -> weight. If provided, performs
                weighted Procrustes where higher-weighted words contribute more to the
                rotation estimate. Words absent from the dict receive weight 1.0.

        Returns:
            W2VModel: The instance itself, for method chaining.

        Raises:
            ValueError: If the filtered vocabularies are empty or mismatched.
        """
        shared_vocab = self.filtered_vocab.intersection(reference_model.filtered_vocab)

        if not shared_vocab:
            raise ValueError("No shared vocabulary between the models.")

        # Sort for reproducibility
        shared_vocab_list = sorted(shared_vocab)

        X = np.vstack([reference_model.filtered_vectors[w] for w in shared_vocab_list])
        Y = np.vstack([self.filtered_vectors[w] for w in shared_vocab_list])

        if weights is not None:
            # Weighted Procrustes: scale rows by sqrt(weight) before SVD.
            # Words absent from the weights dict fall back to weight 1.0.
            w = np.array([weights.get(word, 1.0) for word in shared_vocab_list])
            w = np.sqrt(w / w.sum())
            X = X * w[:, np.newaxis]
            Y = Y * w[:, np.newaxis]

        R, _ = orthogonal_procrustes(Y, X)

        # Apply rotation to all filtered vectors, not just shared vocab.
        # Words outside shared_vocab still need to be rotated into the
        # same space — they just didn't contribute to estimating R.
        for word in self.filtered_vectors:
            self.filtered_vectors[word] = np.dot(self.filtered_vectors[word], R)

        return self

    def is_normalized(self, tolerance=1e-6):
        """
        Check if all word vectors in the model are L2 normalized.

        Args:
            tolerance (float): Allowed deviation from norm 1 due to floating-point precision.

        Returns:
            bool: True if all vectors are normalized, False otherwise.
        """
        norms = np.linalg.norm(self.model.vectors, axis=1)
        return np.all(np.abs(norms - 1) < tolerance)

    def is_aligned_with(self, reference_model, tolerance=1e-6):
        """
        Check if this model is already aligned with a reference model using Procrustes.

        Args:
            reference_model (W2VModel): The reference W2VModel instance.
            tolerance (float): Allowed deviation from identity matrix for Procrustes check.

        Returns:
            bool: True if the models appear to be aligned, False otherwise.
        """
        shared_vocab = self.filter
