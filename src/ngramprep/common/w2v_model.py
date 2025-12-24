import os
from gensim.models import KeyedVectors
import numpy as np
import random
from scipy.linalg import orthogonal_procrustes


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
        # Normalize vectors to unit length
        # Note: In modern gensim, init_sims() is deprecated. We normalize manually.
        import numpy as np
        norms = np.linalg.norm(self.model.vectors, axis=1, keepdims=True)
        self.model.vectors = self.model.vectors / norms
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
            weights (dict or None): Optional stability weights for words.
                - If None: Unweighted Procrustes (all shared vocab words contribute equally)
                - If dict: Weighted Procrustes (words weighted by their stability scores)

        Returns:
            W2VModel: The instance itself, for method chaining.

        Raises:
            ValueError: If the filtered vocabularies are empty or mismatched.
        """
        shared_vocab = self.filtered_vocab.intersection(reference_model.filtered_vocab)

        if not shared_vocab:
            raise ValueError("No shared vocabulary between the models.")

        # Determine whether to use weighted alignment
        if weights is None:
            # Unweighted: use all shared vocabulary
            alignment_vocab = shared_vocab
            use_weights = False
        elif isinstance(weights, dict):
            # Weighted: use words that have weights (should be all shared vocab)
            alignment_vocab = shared_vocab.intersection(set(weights.keys()))
            use_weights = True
        else:
            raise ValueError("weights must be None or a dict")

        if not alignment_vocab:
            raise ValueError("No words available for alignment")

        # Create aligned matrices
        alignment_vocab_list = list(alignment_vocab)
        X = np.vstack([reference_model.filtered_vectors[word] for word in alignment_vocab_list])
        Y = np.vstack([self.filtered_vectors[word] for word in alignment_vocab_list])

        # Perform orthogonal Procrustes alignment (weighted or unweighted)
        if use_weights:
            # Extract weights for alignment words
            weight_values = np.array([weights[word] for word in alignment_vocab_list])
            # Create diagonal weight matrix
            W = np.diag(weight_values)
            # Weighted Procrustes: R = argmin ||W(YR - X)||^2
            # Solution: R from SVD of Y^T W^T W X
            A = Y.T @ W @ W @ X
            U, _, Vt = np.linalg.svd(A)
            R = U @ Vt
        else:
            # Standard unweighted Procrustes
            R, _ = orthogonal_procrustes(Y, X)

        # Apply the transformation to ALL filtered vectors
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
        shared_vocab = self.filtered_vocab.intersection(reference_model.filtered_vocab)

        if not shared_vocab:
            raise ValueError("No shared vocabulary between the models to check alignment.")

        X = np.vstack([reference_model.filtered_vectors[word] for word in shared_vocab])
        Y = np.vstack([self.filtered_vectors[word] for word in shared_vocab])

        R, _ = orthogonal_procrustes(Y, X)

        # Check if R is approximately an identity matrix
        identity_matrix = np.eye(R.shape[0])
        return np.all(np.abs(R - identity_matrix) < tolerance)

    def evaluate_alignment(self, reference_model, tolerance=1e-3):
        """
        Evaluate the alignment quality between this model and a reference model.

        Args:
            reference_model (W2VModel): The reference W2VModel instance.
            tolerance (float): Allowed deviation from identity matrix for Procrustes check.

        Returns:
            dict: A dictionary containing various alignment diagnostics.
        """
        if not isinstance(reference_model, W2VModel):
            raise TypeError("reference_model must be an instance of W2VModel.")

        # Check normalization
        is_norm_self = self.is_normalized()
        is_norm_ref = reference_model.is_normalized()

        # Extract shared vocabulary
        shared_vocab = self.filtered_vocab.intersection(reference_model.filtered_vocab)
        vocab_match = self.filtered_vocab == reference_model.filtered_vocab

        # Prepare matrices for Procrustes
        X = np.vstack([reference_model.filtered_vectors[word] for word in shared_vocab])
        Y = np.vstack([self.filtered_vectors[word] for word in shared_vocab])

        # Compute Procrustes alignment matrix
        R, _ = orthogonal_procrustes(Y, X)
        identity_matrix = np.eye(R.shape[0])
        alignment_deviation = np.linalg.norm(R - identity_matrix)

        # Interpret deviation results
        if alignment_deviation < 1e-4:
            deviation_message = "✅ Alignment deviation is minimal. Alignment likely successful."
        elif alignment_deviation < 1e-2:
            deviation_message = "⚠️ Alignment deviation is small but nonzero. Check vocabulary consistency."
        else:
            deviation_message = "❌ Warning: Alignment deviation is significant. Possible alignment failure."

        # Final assessment
        aligned = alignment_deviation < tolerance

        # Print diagnostic information
        print("\n---------------- Normalization and Alignment Evaluation ------------------")
        print(f"Model1 normalized: {is_norm_self}")
        print(f"Model2 normalized: {is_norm_ref}")
        print(f"Shared vocabulary size: {len(shared_vocab)}")
        print(f"Filtered vocabularies match: {vocab_match}")
        print(f"Shape of X (anchor model vectors): {X.shape}")
        print(f"Shape of Y (target model vectors): {Y.shape}")
        print(f"Alignment deviation from identity: {alignment_deviation:.6f}")
        print(deviation_message)
        print(f"Models are aligned (threshold {tolerance}): {aligned}")
        print("--------------------------------------------------------------------------\n")

        # Return detailed results as a dictionary
        return {
            "is_normalized_self": is_norm_self,
            "is_normalized_ref": is_norm_ref,
            "shared_vocab_size": len(shared_vocab),
            "vocab_match": vocab_match,
            "matrix_shape_X": X.shape,
            "matrix_shape_Y": Y.shape,
            "alignment_deviation": alignment_deviation,
            "alignment_message": deviation_message,
            "is_aligned": aligned
        }

    def compare_words_cosim(self, word1, word2):
        """
        Compute the cosine similarity between two words in a given model.

        Args:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            float: Cosine similarity score between the two words.

        Raises:
            KeyError: If either word is not in the vocabulary.
        """
        if word1 not in self.vocab or word2 not in self.vocab:
            raise KeyError(f"One or both words ('{word1}', '{word2}') are not in the vocabulary.")

        return self.model.similarity(word1, word2)

    def compare_models_cosim(self, reference_model, word=None):
        """
        Compute the mean cosine similarity with a reference model across shared words.

        Args:
            reference_model (W2VModel): The anchor model (reference).

        Returns:
            float: The mean cosine similarity of shared words, or None if no shared words exist.
        """
        if word:
            if not (word in self.vocab and word in reference_model.vocab):
                print(f"⚠️ Warning: Word '{word}' not found in both models.")
                return None, None, None

            similarities = np.dot(self.model[word], reference_model.model[word])
            common_words = 1

        else:
            common_words = self.vocab.intersection(reference_model.vocab)
            if not common_words:
                print("⚠️ Warning: No shared words between models.")
                return None, None, None

            similarities = [np.dot(self.model[word], reference_model.model[word]) for word in common_words]
            common_words = len(common_words)

        return (np.mean(similarities), np.std(similarities), common_words)

    def mean_cosine_similarity_to_all(self, word, excluded_words=None):
        """
        Compute the mean cosine similarity of a given word with every other word in the vocabulary.

        Args:
            word (str): The word for which to compute the mean similarity.
            excluded_words (list or set): Words to exclude from similarity calculations.

        Returns:
            float: Mean cosine similarity score of the word with all other words in the vocabulary.

        Raises:
            KeyError: If the word is not in the vocabulary.
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' is not in the vocabulary.")

        total_similarity = 0
        count = 0

        excluded_words = set(excluded_words) if excluded_words else set()

        for other_word in self.vocab:
            if other_word == word or other_word in excluded_words:
                continue  # Skip self-similarity
            total_similarity += self.compare_words_cosim(word, other_word)
            count += 1

        return total_similarity / count if count > 0 else 0

    def compute_weat(self, targ1, targ2, attr1, attr2, num_permutations=10000, return_std=False, return_associations=False, labels=None):
        """
        Compute WEAT effect size, p-value, and optionally return standard deviation from permutations.
        Fully follows Caliskan et al.'s method.

        Args:
            targ1 (list): First set of target words
            targ2 (list): Second set of target words
            attr1 (list): First set of attribute words
            attr2 (list): Second set of attribute words
            num_permutations (int): Number of permutations for significance testing
            return_std (bool): If True, returns standard deviation from permutations
            return_associations (bool): If True, returns the 4 component mean associations
            labels (dict): Optional labels for targets and attributes. Format:
                {'target1': 'Label1', 'target2': 'Label2', 'attribute1': 'Label3', 'attribute2': 'Label4'}
                If None, uses default keys ('target1_attribute1', etc.)

        Returns:
            tuple: Format depends on parameters:
                - Basic: (effect_size, p_value)
                - With return_std: (effect_size, p_value, std_dev)
                - With return_associations: (effect_size, p_value, associations_dict)
                - With both: (effect_size, p_value, std_dev, associations_dict)
        """
        missing_words = [word for word in (targ1 + targ2 + attr1 + attr2) if word not in self.vocab]
        if missing_words:
            print(f"⚠️ Warning: The following words are missing from the model and will be ignored: {missing_words}")

        def mean_similarity(target_word, attribute_words):
            """Compute mean cosine similarity between a target word and a set of attribute words"""
            sims = [self.model.similarity(target_word, attr) for attr in attribute_words if attr in self.vocab]
            return np.mean(sims) if sims else 0.0

        def s(target_word, attr1_words, attr2_words):
            """Compute association difference for a single target word (Equation 2 in Caliskan et al.)"""
            return mean_similarity(target_word, attr1_words) - mean_similarity(target_word, attr2_words)

        # Compute test statistic using per-word associations (Equation 3 in Caliskan et al.)
        # Sum of s() over target1 minus sum of s() over target2
        targ1_filtered = [w for w in targ1 if w in self.vocab]
        targ2_filtered = [w for w in targ2 if w in self.vocab]
        attr1_filtered = [w for w in attr1 if w in self.vocab]
        attr2_filtered = [w for w in attr2 if w in self.vocab]

        s_vals_targ1 = [s(x, attr1_filtered, attr2_filtered) for x in targ1_filtered]
        s_vals_targ2 = [s(y, attr1_filtered, attr2_filtered) for y in targ2_filtered]

        # Compute pooled standard deviation across all s() values (Equation 4 in Caliskan et al.)
        all_s_vals = s_vals_targ1 + s_vals_targ2
        pooled_std = np.std(all_s_vals, ddof=1)

        if pooled_std == 0:
            print("⚠️ Warning: No variation in association scores. Returning NaN for WEAT effect size.")
            if return_associations:
                # Create association keys using labels if provided
                if labels:
                    t1, t2, a1, a2 = labels['target1'], labels['target2'], labels['attribute1'], labels['attribute2']
                    associations = {
                        f'{t1}→{a1}': np.nan,
                        f'{t1}→{a2}': np.nan,
                        f'{t2}→{a1}': np.nan,
                        f'{t2}→{a2}': np.nan
                    }
                else:
                    associations = {
                        'target1_attribute1': np.nan,
                        'target1_attribute2': np.nan,
                        'target2_attribute1': np.nan,
                        'target2_attribute2': np.nan
                    }
                if return_std:
                    return (np.nan, None, None, associations)
                else:
                    return (np.nan, None, associations)
            return (np.nan, None, None) if return_std else (np.nan, None)

        # Compute observed test statistic and WEAT effect size
        # Test statistic is the sum (for permutation test)
        # Effect size uses means (following Caliskan et al.)
        observed_test_statistic = np.sum(s_vals_targ1) - np.sum(s_vals_targ2)
        mean_diff = np.mean(s_vals_targ1) - np.mean(s_vals_targ2)
        weat_effect_size = mean_diff / pooled_std

        # Compute component associations if requested
        if return_associations:
            # Compute mean associations for each target-attribute pair
            targ1_attr1_sims = [mean_similarity(t, attr1_filtered) for t in targ1_filtered]
            targ1_attr2_sims = [mean_similarity(t, attr2_filtered) for t in targ1_filtered]
            targ2_attr1_sims = [mean_similarity(t, attr1_filtered) for t in targ2_filtered]
            targ2_attr2_sims = [mean_similarity(t, attr2_filtered) for t in targ2_filtered]

            # Create association keys using labels if provided
            if labels:
                t1, t2, a1, a2 = labels['target1'], labels['target2'], labels['attribute1'], labels['attribute2']
                associations = {
                    f'{t1}→{a1}': np.mean(targ1_attr1_sims) if targ1_attr1_sims else np.nan,
                    f'{t1}→{a2}': np.mean(targ1_attr2_sims) if targ1_attr2_sims else np.nan,
                    f'{t2}→{a1}': np.mean(targ2_attr1_sims) if targ2_attr1_sims else np.nan,
                    f'{t2}→{a2}': np.mean(targ2_attr2_sims) if targ2_attr2_sims else np.nan
                }
            else:
                associations = {
                    'target1_attribute1': np.mean(targ1_attr1_sims) if targ1_attr1_sims else np.nan,
                    'target1_attribute2': np.mean(targ1_attr2_sims) if targ1_attr2_sims else np.nan,
                    'target2_attribute1': np.mean(targ2_attr1_sims) if targ2_attr1_sims else np.nan,
                    'target2_attribute2': np.mean(targ2_attr2_sims) if targ2_attr2_sims else np.nan
                }

        if num_permutations == 0:
            if return_associations:
                if return_std:
                    return (weat_effect_size, None, pooled_std, associations)
                else:
                    return (weat_effect_size, None, associations)
            return (weat_effect_size, None, pooled_std) if return_std else (weat_effect_size, None)

        # Permutation Test (Shuffle only target words `X` and `Y`)
        combined_targets = targ1_filtered + targ2_filtered
        n = len(targ1_filtered)
        permuted_test_statistics = []

        for _ in range(num_permutations):
            # Shuffle only the target words (not attributes)
            perm_targ1 = random.sample(combined_targets, n)
            perm_targ2 = [w for w in combined_targets if w not in perm_targ1]

            # Compute per-word association scores for permuted targets
            perm_s_vals_targ1 = [s(x, attr1_filtered, attr2_filtered) for x in perm_targ1]
            perm_s_vals_targ2 = [s(y, attr1_filtered, attr2_filtered) for y in perm_targ2]

            # Compute test statistic for this permutation
            perm_test_statistic = np.sum(perm_s_vals_targ1) - np.sum(perm_s_vals_targ2)
            permuted_test_statistics.append(perm_test_statistic)

        # Compute p-value (one-sided test) by comparing test statistics
        # Following Caliskan et al. (2017): Pr[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
        p_value = np.mean(np.array(permuted_test_statistics) > observed_test_statistic)

        # Compute standard deviation of the permuted test statistics (confidence interval estimate)
        std_dev = np.std(permuted_test_statistics, ddof=1) if return_std else None

        # Return results based on flags
        if return_associations:
            if return_std:
                return (weat_effect_size, p_value, std_dev, associations)
            else:
                return (weat_effect_size, p_value, associations)
        else:
            return (weat_effect_size, p_value, std_dev) if return_std else (weat_effect_size, p_value)

    def save(self, output_path):
        """
        Save the filtered and aligned model to the specified path.

        Args:
            output_path (str): Path to save the aligned .kv model.

        Raises:
            ValueError: If no filtered vectors are available to save.
        """
        if not hasattr(self, "filtered_vectors") or not self.filtered_vectors:
            raise ValueError("No filtered vectors available to save.")

        aligned_model = KeyedVectors(vector_size=self.vector_size)
        aligned_model.add_vectors(
            list(self.filtered_vectors.keys()), list(self.filtered_vectors.values())
        )
        aligned_model.save(output_path)
