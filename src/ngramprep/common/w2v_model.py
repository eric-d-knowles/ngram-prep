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
            alignment_vocab = shared_vocab
            use_weights = False
        elif isinstance(weights, dict):
            alignment_vocab = shared_vocab.intersection(set(weights.keys()))
            use_weights = True
        else:
            raise ValueError("weights must be None or a dict")

        if not alignment_vocab:
            raise ValueError("No words available for alignment")

        # Sort for reproducibility
        alignment_vocab_list = sorted(alignment_vocab)
        X = np.vstack([reference_model.filtered_vectors[word] for word in alignment_vocab_list])
        Y = np.vstack([self.filtered_vectors[word] for word in alignment_vocab_list])

        if use_weights:
            # Weighted Procrustes via sqrt-scaling: equivalent to the diagonal
            # weight matrix formulation but avoids materializing a 29k x 29k
            # matrix, keeping memory at O(vocab x dims) instead of O(vocab^2).
            w = np.array([weights[word] for word in alignment_vocab_list])
            w = np.sqrt(w / w.sum())
            X = X * w[:, np.newaxis]
            Y = Y * w[:, np.newaxis]

        R, _ = orthogonal_procrustes(Y, X)

        # Vectorized rotation — stack all filtered vectors into a matrix,
        # apply R in a single NumPy operation, then unpack back to dict.
        # Far faster than iterating word-by-word in Python.
        words = list(self.filtered_vectors.keys())
        matrix = np.vstack([self.filtered_vectors[w] for w in words])
        rotated = matrix @ R
        self.filtered_vectors = {w: rotated[i] for i, w in enumerate(words)}

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

        is_norm_self = self.is_normalized()
        is_norm_ref = reference_model.is_normalized()

        shared_vocab = self.filtered_vocab.intersection(reference_model.filtered_vocab)
        vocab_match = self.filtered_vocab == reference_model.filtered_vocab

        X = np.vstack([reference_model.filtered_vectors[word] for word in shared_vocab])
        Y = np.vstack([self.filtered_vectors[word] for word in shared_vocab])

        R, _ = orthogonal_procrustes(Y, X)
        identity_matrix = np.eye(R.shape[0])
        alignment_deviation = np.linalg.norm(R - identity_matrix)

        if alignment_deviation < 1e-4:
            deviation_message = "✅ Alignment deviation is minimal. Alignment likely successful."
        elif alignment_deviation < 1e-2:
            deviation_message = "⚠️ Alignment deviation is small but nonzero. Check vocabulary consistency."
        else:
            deviation_message = "❌ Warning: Alignment deviation is significant. Possible alignment failure."

        aligned = alignment_deviation < tolerance

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

        targ1_filtered = [w for w in targ1 if w in self.vocab]
        targ2_filtered = [w for w in targ2 if w in self.vocab]
        attr1_filtered = [w for w in attr1 if w in self.vocab]
        attr2_filtered = [w for w in attr2 if w in self.vocab]

        s_vals_targ1 = [s(x, attr1_filtered, attr2_filtered) for x in targ1_filtered]
        s_vals_targ2 = [s(y, attr1_filtered, attr2_filtered) for y in targ2_filtered]

        all_s_vals = s_vals_targ1 + s_vals_targ2
        pooled_std = np.std(all_s_vals, ddof=1)

        if pooled_std == 0:
            print("⚠️ Warning: No variation in association scores. Returning NaN for WEAT effect size.")
            if return_associations:
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

        observed_test_statistic = np.sum(s_vals_targ1) - np.sum(s_vals_targ2)
        mean_diff = np.mean(s_vals_targ1) - np.mean(s_vals_targ2)
        weat_effect_size = mean_diff / pooled_std

        if return_associations:
            targ1_attr1_sims = [mean_similarity(t, attr1_filtered) for t in targ1_filtered]
            targ1_attr2_sims = [mean_similarity(t, attr2_filtered) for t in targ1_filtered]
            targ2_attr1_sims = [mean_similarity(t, attr1_filtered) for t in targ2_filtered]
            targ2_attr2_sims = [mean_similarity(t, attr2_filtered) for t in targ2_filtered]

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

        combined_targets = targ1_filtered + targ2_filtered
        n = len(targ1_filtered)
        permuted_test_statistics = []

        for _ in range(num_permutations):
            perm_targ1 = random.sample(combined_targets, n)
            perm_targ2 = [w for w in combined_targets if w not in perm_targ1]

            perm_s_vals_targ1 = [s(x, attr1_filtered, attr2_filtered) for x in perm_targ1]
            perm_s_vals_targ2 = [s(y, attr1_filtered, attr2_filtered) for y in perm_targ2]

            perm_test_statistic = np.sum(perm_s_vals_targ1) - np.sum(perm_s_vals_targ2)
            permuted_test_statistics.append(perm_test_statistic)

        p_value = np.mean(np.array(permuted_test_statistics) > observed_test_statistic)
        std_dev = np.std(permuted_test_statistics, ddof=1) if return_std else None

        if return_associations:
            if return_std:
                return (weat_effect_size, p_value, std_dev, associations)
            else:
                return (weat_effect_size, p_value, associations)
        else:
            return (weat_effect_size, p_value, std_dev) if return_std else (weat_effect_size, p_value)

    def compute_pca_dimension(self, token_contrasts, ensure_sign_positive=None, n_components_diagnostic=None):
        """
        Compute a semantic dimension via PCA on contrast pair difference vectors.

        Args:
            token_contrasts (list of tuples): List of (token1, token2) pairs defining contrasts.
                Example: [('he', 'she'), ('him', 'her'), ('man', 'woman'), ...]
            ensure_sign_positive (bool, list, or None): Controls sign orientation of PC1.
                - If True: infer positive tokens from pair order (second element of each pair)
                - If list of str: use specified tokens as positive pole
                - If None/False: no sign consistency enforcement (arbitrary orientation)
            n_components_diagnostic (int, optional): If provided, fit this many components for diagnostics.

        Returns:
            dict: dimension, variance_explained, component_loadings, pca_object,
                  all_variance_explained.

        Raises:
            ValueError: If contrast pairs contain words not in vocabulary.
        """
        from sklearn.decomposition import PCA

        if isinstance(token_contrasts, list):
            pairs = token_contrasts
        elif isinstance(token_contrasts, dict):
            pairs = list(token_contrasts.values()) if token_contrasts else []
        else:
            raise TypeError("token_contrasts must be a list of tuples or dict of pairs")

        difference_vectors = []
        valid_pairs = []
        missing_pairs = []

        for pair in pairs:
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                token1, token2 = pair
                if token1 in self.vocab and token2 in self.vocab:
                    diff = self.model[token2] - self.model[token1]
                    difference_vectors.append(diff)
                    valid_pairs.append(pair)
                else:
                    missing_pairs.append(pair)
            else:
                raise ValueError(f"Invalid pair format: {pair}. Expected (token1, token2) tuple.")

        if missing_pairs:
            print(f"⚠️ Warning: The following pairs contain words not in vocabulary and will be skipped:")
            for pair in missing_pairs:
                print(f"  {pair}")

        if not valid_pairs:
            raise ValueError("No valid contrast pairs found in vocabulary.")

        X = np.array(difference_vectors)
        n_components = n_components_diagnostic if n_components_diagnostic else 1
        n_components = min(n_components, X.shape[0], X.shape[1])

        pca = PCA(n_components=n_components)
        pca.fit(X)

        dimension = pca.components_[0]
        variance_explained = pca.explained_variance_ratio_[0]
        all_variance_explained = pca.explained_variance_ratio_ if n_components_diagnostic else None

        if ensure_sign_positive:
            if isinstance(ensure_sign_positive, bool):
                positive_tokens = [pair[1] for pair in valid_pairs]
            else:
                positive_tokens = ensure_sign_positive

            positive_projections = []
            positive_tokens_found = []
            for token in positive_tokens:
                if token in self.vocab:
                    proj = np.dot(self.model[token], dimension)
                    positive_projections.append(proj)
                    positive_tokens_found.append((token, proj))

            if positive_projections:
                mean_proj = np.mean(positive_projections)
                if mean_proj < 0:
                    dimension = -dimension
                    positive_tokens_found = [(t, -p) for t, p in positive_tokens_found]

        component_loadings = {}
        for pair in valid_pairs:
            diff_vec = self.model[pair[1]] - self.model[pair[0]]
            diff_norm = diff_vec / (np.linalg.norm(diff_vec) + 1e-10)
            dim_norm = dimension / (np.linalg.norm(dimension) + 1e-10)
            cosine_sim = np.dot(diff_norm, dim_norm)
            component_loadings[f"{pair[0]}→{pair[1]}"] = cosine_sim

        return {
            'dimension': dimension,
            'variance_explained': variance_explained,
            'component_loadings': component_loadings,
            'pca_object': pca,
            'all_variance_explained': all_variance_explained
        }

    def compute_meandiff_dimension(self, token_contrasts, verbose=False):
        """
        Compute a semantic dimension via mean of contrast pair difference vectors.

        Args:
            token_contrasts (list of tuples): List of (token1, token2) pairs defining contrasts.
                Example: [('he', 'she'), ('him', 'her'), ('man', 'woman'), ...]
            verbose (bool): If True, prints summary to console. Default False.

        Returns:
            dict: dimension, component_loadings, n_pairs.

        Raises:
            ValueError: If no valid contrast pairs found in vocabulary.
        """
        if isinstance(token_contrasts, list):
            pairs = token_contrasts
        elif isinstance(token_contrasts, dict):
            pairs = list(token_contrasts.values()) if token_contrasts else []
        else:
            raise TypeError("token_contrasts must be a list of tuples or dict of pairs")

        difference_vectors = []
        valid_pairs = []
        missing_pairs = []

        for pair in pairs:
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                token1, token2 = pair
                if token1 in self.vocab and token2 in self.vocab:
                    diff = self.model[token2] - self.model[token1]
                    difference_vectors.append(diff)
                    valid_pairs.append(pair)
                else:
                    missing_pairs.append(pair)
            else:
                raise ValueError(f"Invalid pair format: {pair}. Expected (token1, token2) tuple.")

        if missing_pairs:
            print(f"⚠️ Warning: The following pairs contain words not in vocabulary and will be skipped:")
            for pair in missing_pairs:
                print(f"  {pair}")

        if not valid_pairs:
            raise ValueError("No valid contrast pairs found in vocabulary.")

        mean_diff = np.mean(difference_vectors, axis=0)
        dimension = mean_diff / (np.linalg.norm(mean_diff) + 1e-10)

        component_loadings = {}
        for pair in valid_pairs:
            diff_vec = self.model[pair[1]] - self.model[pair[0]]
            diff_norm = diff_vec / (np.linalg.norm(diff_vec) + 1e-10)
            cosine_sim = np.dot(diff_norm, dimension)
            component_loadings[f"{pair[0]}→{pair[1]}"] = cosine_sim

        if verbose:
            print("\n" + "═" * 100)
            print("MEAN-DIFFERENCE DIMENSION")
            print("═" * 100)
            print(f"Valid pairs: {len(valid_pairs)}\n")
            print("Pair Loadings")
            print("─" * 100)
            print(f"{'Pair':<30} {'Loading':>15}")
            print("─" * 100)
            for pair, loading in sorted(component_loadings.items(), key=lambda x: x[1], reverse=True):
                print(f"{pair:<30} {loading:>15.4f}")

        return {
            'dimension': dimension,
            'component_loadings': component_loadings,
            'n_pairs': len(valid_pairs)
        }

    def project_onto_dimension(self, word, dimension):
        """
        Project a word onto a semantic dimension via cosine similarity.

        Args:
            word (str): The word to project.
            dimension (np.ndarray): The dimension vector.

        Returns:
            float: Cosine similarity between the word vector and dimension (in range [-1, 1]).

        Raises:
            ValueError: If word is not in vocabulary.
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in model vocabulary.")

        word_vec = self.model[word]
        projection = np.dot(word_vec, dimension) / (np.linalg.norm(word_vec) * np.linalg.norm(dimension) + 1e-10)
        return projection

    def compare_dimension_methods(self, token_contrasts, test_words=None, ensure_sign_positive=True,
                                   n_components_diagnostic=None, verbose=True):
        """
        Compare mean-difference and PCA methods for extracting semantic dimensions.

        Args:
            token_contrasts (list of tuples): List of (token1, token2) pairs defining contrasts.
            test_words (list of str, optional): Words to project onto both dimensions for comparison.
            ensure_sign_positive (bool or list, optional): Sign control for PCA.
            n_components_diagnostic (int, optional): Number of PCA components for diagnostics.
            verbose (bool): If True, prints comparison summary to console.

        Returns:
            dict: meandiff_result, pca_result, angle_degrees, cosine_similarity,
                  agreement, pair_loadings, word_projections.
        """
        meandiff_result = self.compute_meandiff_dimension(token_contrasts)

        if n_components_diagnostic is None:
            n_components_diagnostic = min(meandiff_result['n_pairs'], 10)

        pca_result = self.compute_pca_dimension(
            token_contrasts,
            ensure_sign_positive=ensure_sign_positive,
            n_components_diagnostic=n_components_diagnostic
        )

        meandiff_dim = meandiff_result['dimension']
        pca_dim = pca_result['dimension']

        cos_sim = float(np.dot(meandiff_dim, pca_dim) /
                       (np.linalg.norm(meandiff_dim) * np.linalg.norm(pca_dim) + 1e-10))
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1, 1))))

        if angle_deg < 10:
            agreement = 'nearly identical'
        elif angle_deg < 30:
            agreement = 'similar'
        else:
            agreement = 'different'

        import pandas as pd
        pair_loadings_data = []
        for pair_name in meandiff_result['component_loadings'].keys():
            pair_loadings_data.append({
                'pair': pair_name,
                'meandiff_loading': meandiff_result['component_loadings'][pair_name],
                'pca_loading': pca_result['component_loadings'][pair_name],
                'loading_diff': abs(meandiff_result['component_loadings'][pair_name] -
                                   pca_result['component_loadings'][pair_name])
            })
        pair_loadings_df = pd.DataFrame(pair_loadings_data)
        pair_loadings_df = pair_loadings_df.sort_values('meandiff_loading', ascending=False)

        word_projections_df = None
        if test_words is None:
            test_words = sorted(set([t for pair in token_contrasts
                                    for t in pair if isinstance(pair, (tuple, list))]))

        if test_words:
            word_proj_data = []
            for word in test_words:
                if word in self.vocab:
                    proj_md = self.project_onto_dimension(word, meandiff_dim)
                    proj_pca = self.project_onto_dimension(word, pca_dim)
                    word_proj_data.append({
                        'word': word,
                        'meandiff_proj': proj_md,
                        'pca_proj': proj_pca,
                        'proj_diff': abs(proj_md - proj_pca)
                    })
            if word_proj_data:
                word_projections_df = pd.DataFrame(word_proj_data)
                word_projections_df = word_projections_df.sort_values('meandiff_proj', ascending=False)

        if verbose:
            print("\n" + "═" * 100)
            print("DIMENSION METHOD COMPARISON")
            print("═" * 100)
            print(f"Contrast pairs:          {meandiff_result['n_pairs']}")
            print(f"Angle between dimensions: {angle_deg:.2f}°")
            print(f"Cosine similarity:       {cos_sim:+.4f}")
            print(f"Assessment:              Dimensions are {agreement}")
            print(f"PCA variance explained:  {pca_result['variance_explained']*100:.1f}%")

            print("\n" + "─" * 100)
            print("PAIR LOADINGS COMPARISON")
            print("─" * 100)
            print(f"{'Pair':<30} {'MeanDiff':>15} {'PCA':>15} {'|Diff|':>15}")
            print("─" * 100)
            for _, row in pair_loadings_df.iterrows():
                print(f"{row['pair']:<30} {row['meandiff_loading']:>15.4f} "
                      f"{row['pca_loading']:>15.4f} {row['loading_diff']:>15.4f}")

            if word_projections_df is not None and len(word_projections_df) > 0:
                print("\n" + "─" * 100)
                print("WORD PROJECTIONS COMPARISON")
                print("─" * 100)
                print(f"{'Word':<20} {'MeanDiff':>20} {'PCA':>20} {'|Diff|':>15}")
                print("─" * 100)
                for _, row in word_projections_df.iterrows():
                    print(f"{row['word']:<20} {row['meandiff_proj']:>20.4f} "
                          f"{row['pca_proj']:>20.4f} {row['proj_diff']:>15.4f}")

        return {
            'meandiff_result': meandiff_result,
            'pca_result': pca_result,
            'angle_degrees': angle_deg,
            'cosine_similarity': cos_sim,
            'agreement': agreement,
            'pair_loadings': pair_loadings_df,
            'word_projections': word_projections_df
        }

    @staticmethod
    def plot_pca_diagnostics(pca_result, figsize=(12, 5), display_dpi=160):
        """
        Plot diagnostic visualizations for PCA dimension analysis.

        Args:
            pca_result (dict): Output from compute_pca_dimension() with n_components_diagnostic.
            figsize (tuple): Figure size (width, height) in inches.
            display_dpi (int): DPI for rendering.

        Returns:
            matplotlib.figure.Figure: The diagnostic figure.

        Raises:
            ValueError: If pca_result doesn't contain all_variance_explained.
        """
        import matplotlib.pyplot as plt

        all_var = pca_result.get('all_variance_explained')
        if all_var is None:
            raise ValueError(
                "PCA diagnostics not available. Re-run compute_pca_dimension() with "
                "n_components_diagnostic=<number of components> to enable diagnostics."
            )

        text_scale = np.sqrt((figsize[0] * figsize[1]) / (12 * 6))

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=display_dpi)
        fig.suptitle('PCA Dimension Diagnostics', fontsize=int(16*text_scale), fontweight='bold')

        ax = axes[0]
        n_components = len(all_var)
        ax.bar(range(1, n_components + 1), all_var, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Component', fontsize=int(13*text_scale))
        ax.set_ylabel('Variance Explained', fontsize=int(13*text_scale))
        ax.set_title('Scree Plot', fontsize=int(14*text_scale))
        ax.set_xticks(range(1, min(n_components + 1, 11)))
        ax.tick_params(axis='both', which='major', labelsize=int(10*text_scale))
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1]
        cumsum = np.cumsum(all_var)
        ax.plot(range(1, n_components + 1), cumsum, 'o-', linewidth=2.5,
                markersize=8, color='darkgreen', label='Cumulative')
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='80% threshold')
        ax.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='90% threshold')
        ax.set_xlabel('Number of Components', fontsize=int(13*text_scale))
        ax.set_ylabel('Cumulative Variance Explained', fontsize=int(13*text_scale))
        ax.set_title('Cumulative Variance', fontsize=int(14*text_scale))
        ax.set_xticks(range(1, min(n_components + 1, 11)))
        ax.set_ylim([0, 1.05])
        ax.tick_params(axis='both', which='major', labelsize=int(10*text_scale))
        ax.legend(fontsize=int(11*text_scale), loc='lower right')
        ax.grid(alpha=0.3)

        fig.tight_layout(h_pad=5.0, w_pad=3.0)

        return fig

    @staticmethod
    def print_pca_variance_summary(pca_result, n_show=5):
        """
        Print a text summary of PCA variance breakdown.

        Args:
            pca_result (dict): Output from compute_pca_dimension() with n_components_diagnostic.
            n_show (int): Number of top components to display in summary table.

        Raises:
            ValueError: If pca_result doesn't contain all_variance_explained.
        """
        all_var = pca_result.get('all_variance_explained')
        if all_var is None:
            raise ValueError(
                "PCA diagnostics not available. Re-run compute_pca_dimension() with "
                "n_components_diagnostic=<number of components> to enable diagnostics."
            )

        cumsum = np.cumsum(all_var)
        n_show = min(n_show, len(all_var))

        print("\n" + "═" * 100)
        print("PCA VARIANCE BREAKDOWN")
        print("═" * 100)
        print(f"{'Component':<20} {'Variance':>20} {'Cumulative':>20} {'% of Total':>20}")
        print("─" * 100)

        for i in range(n_show):
            var_pct = all_var[i] * 100
            cum_pct = cumsum[i] * 100
            print(f"PC{i+1:<18} {all_var[i]:>20.4f}  {cumsum[i]:>20.4f}  {var_pct:>18.2f}%")

        if len(all_var) > n_show:
            remaining_var = 1.0 - cumsum[n_show-1]
            print(f"{'Others':<20} {remaining_var:>20.4f}  {1.0:>20.4f}  {remaining_var*100:>18.2f}%")

        print("═" * 100)
        print(f"\nPC1 explains {all_var[0]*100:.2f}% of total variance")

        n_80 = np.argmax(cumsum >= 0.80) + 1
        n_90 = np.argmax(cumsum >= 0.90) + 1
        n_95 = np.argmax(cumsum >= 0.95) + 1

        print(f"  → {n_80} components for 80% variance")
        print(f"  → {n_90} components for 90% variance")
        if n_95 <= len(all_var):
            print(f"  → {n_95} components for 95% variance")

    @staticmethod
    def print_meandiff_summary(meandiff_result):
        """
        Print a text summary of mean-difference dimension quality and alignment.

        Args:
            meandiff_result (dict): Output from compute_meandiff_dimension().

        Raises:
            ValueError: If meandiff_result doesn't contain required fields.
        """
        loadings = meandiff_result.get('component_loadings')
        n_pairs = meandiff_result.get('n_pairs')

        if loadings is None or n_pairs is None:
            raise ValueError(
                "component_loadings or n_pairs not found in meandiff_result. "
                "Ensure compute_meandiff_dimension() was called successfully."
            )

        loading_values = np.array(list(loadings.values()))

        mean_loading = np.mean(loading_values)
        std_loading = np.std(loading_values)
        min_loading = np.min(loading_values)
        max_loading = np.max(loading_values)

        cv = std_loading / mean_loading if mean_loading > 0 else np.inf

        if cv < 0.15:
            quality = "EXCELLENT (highly coherent pairs)"
        elif cv < 0.25:
            quality = "GOOD (coherent pairs)"
        elif cv < 0.40:
            quality = "MODERATE (mixed semantic domains)"
        else:
            quality = "POOR (heterogeneous pairs)"

        print("\n" + "═" * 100)
        print("MEAN-DIFFERENCE DIMENSION SUMMARY")
        print("═" * 100)
        print(f"Valid pairs:             {n_pairs}")
        print(f"\nPair Loading Statistics:")
        print("─" * 100)
        print(f"Mean loading:            {mean_loading:>20.4f}")
        print(f"Std deviation:           {std_loading:>20.4f}")
        print(f"Min loading:             {min_loading:>20.4f}")
        print(f"Max loading:             {max_loading:>20.4f}")
        print(f"Loading range:           {max_loading - min_loading:>20.4f}")
        print(f"Coeff. of variation:     {cv:>20.4f}")

        print(f"\n{'─' * 100}")
        print(f"Pair Coherence:          {quality}")
        print(f"{'─' * 100}")

        if std_loading > 0:
            print(f"\nAll {n_pairs} pair loadings are positive by construction.")
            print(f"Loading range of {min_loading:.4f}–{max_loading:.4f} suggests")
            if cv < 0.25:
                print("pairs form a coherent semantic dimension.")
            else:
                print("pairs may span heterogeneous semantic subdomains.")

        print("═" * 100)

    @staticmethod
    def print_component_loadings(pca_result, title="COMPONENT LOADINGS"):
        """
        Print component loadings showing how well each contrast pair aligns with PC1.

        Args:
            pca_result (dict): Output from compute_pca_dimension().
            title (str): Title for the output section.
        """
        loadings = pca_result.get('component_loadings')
        if loadings is None:
            raise ValueError("component_loadings not found in pca_result.")

        print("\n" + "─" * 100)
        print(title)
        print("─" * 100)
        print(f"{'Pair':<35} {'Loading':>20}")
        print("─" * 100)
        for pair, loading in sorted(loadings.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pair:<33} {loading:>20.4f}")

    @staticmethod
    def print_word_projections(model, dimension, test_words=None, title=None):
        """
        Print word projections on a semantic dimension with visual bar representation.

        Args:
            model (W2VModel): The model instance to project words from.
            dimension (np.ndarray): The dimension vector from compute_pca_dimension().
            test_words (list of str, optional): Words to project. If None, uses gender-related words.
            title (str, optional): Custom title for output. If None, uses generic title.

        Returns:
            dict: Dictionary mapping words to their projection values.
        """
        if test_words is None:
            test_words = [
                'she', 'he', 'woman', 'man', 'queen', 'king', 'nurse', 'doctor',
                'mother', 'father', 'princess', 'prince', 'actress', 'actor'
            ]

        if title is None:
            title = "WORD PROJECTIONS ON DIMENSION"

        projections = {}
        for word in test_words:
            try:
                proj = model.project_onto_dimension(word, dimension)
                projections[word] = proj
            except ValueError:
                pass

        print("\n" + "─" * 100)
        print(title)
        print("─" * 100)
        print(f"{'Word':<20} {'Projection':>20} {'Visualization':>55}")
        print("─" * 100)
        for word, proj in sorted(projections.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(abs(proj) * 30)
            if proj > 0:
                bar = '█' * bar_length
            else:
                bar = '░' * bar_length
            print(f"  {word:<18} {proj:>20.4f}  {bar:>50}")

        return projections

    @staticmethod
    def diagnose_pca_sign(pca_result, model, title="PCA SIGN DIAGNOSIS"):
        """
        Diagnose the sign orientation of the PCA dimension by examining positive token projections.

        Args:
            pca_result (dict): Output from compute_pca_dimension().
            model (W2VModel): The model instance (to access vocab and vectors).
            title (str): Title for output.
        """
        dimension = pca_result['dimension']
        loadings = pca_result.get('component_loadings', {})

        print("\n" + "─" * 100)
        print(title)
        print("─" * 100)

        print("\nContrast pair loadings (sorted by value):")
        print("─" * 100)
        print(f"{'Pair':<40} {'Loading':>25}")
        print("─" * 100)
        if loadings:
            for pair, loading in sorted(loadings.items(), key=lambda x: x[1], reverse=True):
                direction = "+" if loading > 0 else "-"
                print(f"  {direction} {pair:<37} {loading:>25.4f}")

        print("\nSign Alignment Analysis:")
        print("─" * 100)

        positive_side_tokens = set()
        negative_side_tokens = set()

        for pair, loading in loadings.items():
            parts = pair.split('→')
            if len(parts) == 2:
                token1, token2 = parts[0].strip(), parts[1].strip()
                if loading > 0:
                    positive_side_tokens.add(token2)
                    negative_side_tokens.add(token1)
                else:
                    positive_side_tokens.add(token1)
                    negative_side_tokens.add(token2)

        if positive_side_tokens:
            print(f"\nTokens on POSITIVE side: {', '.join(sorted(positive_side_tokens))}")
        if negative_side_tokens:
            print(f"Tokens on NEGATIVE side: {', '.join(sorted(negative_side_tokens))}")

        pos_projs = []
        neg_projs = []

        for token in positive_side_tokens:
            if token in model.vocab:
                proj = np.dot(model.model[token], dimension)
                pos_projs.append((token, proj))

        for token in negative_side_tokens:
            if token in model.vocab:
                proj = np.dot(model.model[token], dimension)
                neg_projs.append((token, proj))

        if pos_projs:
            mean_pos = np.mean([p for _, p in pos_projs])
            print(f"\nMean projection (positive side): {mean_pos:+.4f}")
            print(f"  Sample words: {', '.join([t for t, _ in pos_projs[:5]])}")

        if neg_projs:
            mean_neg = np.mean([p for _, p in neg_projs])
            print(f"Mean projection (negative side): {mean_neg:+.4f}")
            print(f"  Sample words: {', '.join([t for t, _ in neg_projs[:5]])}")

        if pos_projs and neg_projs:
            mean_pos = np.mean([p for _, p in pos_projs])
            mean_neg = np.mean([p for _, p in neg_projs])
            if mean_pos < mean_neg:
                print("\n⚠️  WARNING: Sign may be FLIPPED!")
                print(f"   Positive side tokens project to {mean_pos:+.4f}")
                print(f"   Negative side tokens project to {mean_neg:+.4f}")
            else:
                print("\n✓ Sign orientation appears correct")
                print(f"  Positive side at {mean_pos:+.4f}, negative side at {mean_neg:+.4f}")

    def save(self, output_path):
        """
        Save the filtered and aligned model to the specified path.

        Preserves word frequency counts (expandos) from the source model so that
        downstream stability weighting can access count attributes on saved models.

        Args:
            output_path (str): Path to save the aligned .kv model.

        Raises:
            ValueError: If no filtered vectors are available to save.
        """
        if not hasattr(self, "filtered_vectors") or not self.filtered_vectors:
            raise ValueError("No filtered vectors available to save.")

        words = list(self.filtered_vectors.keys())
        aligned_model = KeyedVectors(vector_size=self.vector_size)
        aligned_model.add_vectors(words, list(self.filtered_vectors.values()))

        # Preserve count attributes from source model. These are dropped by
        # add_vectors since it only populates vectors and index_to_key.
        # Without this, get_vecattr(word, 'count') raises KeyError on saved
        # models, breaking frequency-based stability weighting downstream.
        if hasattr(self.model, 'expandos') and 'count' in self.model.expandos:
            for w in words:
                if w in self.model:
                    try:
                        count = self.model.get_vecattr(w, 'count')
                        aligned_model.set_vecattr(w, 'count', count)
                    except KeyError:
                        pass

        aligned_model.save(output_path)