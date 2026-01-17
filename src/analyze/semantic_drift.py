import os
from tqdm.notebook import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from ngramprep.common.w2v_model import W2VModel
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
import pandas as pd

def compute_similarity_to_previous_year(args):
    year, prev_model_path, model_path, word = args

    prev_model = W2VModel(prev_model_path)
    model = W2VModel(model_path)

    similarity_mean, similarity_sd, common_words = model.compare_models_cosim(prev_model, word)

    return year, similarity_mean, similarity_sd, common_words


def compute_distance_from_reference(args):
    """Helper function for multiprocessing: Computes distance from reference year."""
    year, model_path, reference_model_path, reference_year, word = args

    if year == reference_year:
        # Load reference model to get vocab size
        ref_model = W2VModel(reference_model_path)
        return (year, 0.0, 0.0, len(ref_model.vocab))

    try:
        reference_model = W2VModel(reference_model_path)
        model = W2VModel(model_path)
        similarity_mean, similarity_sd, common_words = model.compare_models_cosim(reference_model, word)

        if similarity_mean is not None:
            distance = 1 - similarity_mean
            return (year, distance, similarity_sd, common_words)
        else:
            return (year, None, None, None)
    except Exception as e:
        return (year, None, None, str(e))

def track_local_semantic_change(
    start_year, end_year, model_dir, word=None, year_step=1, plot=True, smooth=False, sigma=2,
    confidence=0.95, error_type="CI", num_workers=None, df=None, regress_on=None, plot_derivative=True
):
    """
    Track year-over-year semantic change (local change between consecutive years).

    Measures how much word embeddings change from one year to the next by computing
    1 - cosine_similarity between consecutive years. When word=None, computes average
    change across all shared vocabulary words.

    Args:
        start_year (int): Starting year of the range.
        end_year (int): Ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        word (str, optional): Specific word to track. If None, tracks all shared words.
        year_step (int): Step size for year increments (default: 1).
        plot (bool): Whether to plot the results.
        smooth (bool): Whether to apply Gaussian smoothing.
        sigma (float): Standard deviation for Gaussian smoothing.
        confidence (float): Confidence level for error bands.
        error_type (str): Either "CI" (confidence intervals) or "SE" (standard error).
        num_workers (int, optional): Number of parallel workers.
        df (pd.DataFrame, optional): DataFrame for regression adjustment.
        regress_on (str, optional): Column name to regress on for adjustment.

    Returns:
        pd.DataFrame: DataFrame with columns ['Drift', 'Error', 'Shared'] indexed by year.
    """
    drift_data = {}
    missing_years = []
    error_years = {}

    model_paths = {}
    for year in range(start_year, end_year + 1, year_step):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = sorted(glob.glob(model_pattern))
        if model_files:
            model_paths[year] = model_files[-1]
        else:
            missing_years.append(year)

    if not model_paths or len(model_paths) < 2:
        print("❌ Not enough valid models found for year-over-year analysis. Exiting.")
        return {}

    years_available = sorted(model_paths.keys())
    args = [(years_available[i], model_paths[years_available[i-1]], model_paths[years_available[i]], word)
            for i in range(1, len(years_available))]

    num_workers = num_workers or min(cpu_count(), len(args))
    with Pool(num_workers) as pool:
        results = pool.map(compute_similarity_to_previous_year, args)

    for result in results:
        year, similarity_mean, similarity_sd, shared_vocab_size = result

        if similarity_mean:
            change_mean = 1 - similarity_mean

            if error_type == "CI":
                error_measure = stats.norm.ppf(1 - (1 - confidence) / 2) * (similarity_sd / np.sqrt(shared_vocab_size))
            elif error_type == "SE":
                error_measure = similarity_sd / np.sqrt(shared_vocab_size) if shared_vocab_size > 1 else 0
            else:
                raise ValueError("Invalid error_type. Choose 'CI' or 'SE'.")

            drift_data[year] = (change_mean, error_measure, shared_vocab_size)

    if missing_years:
        print(f"⚠️ No models found for these years: {missing_years}")

    if error_years:
        print("❌ Errors occurred in the following years:")
        for year, err in error_years.items():
            print(f"  {year}: {err}")

    if not drift_data:
        print("❌ No valid drift scores computed. Exiting.")
        return {}

    df_drift = pd.DataFrame.from_dict(drift_data, orient="index", columns=["Drift", "Error", "Shared"])
    df_drift.index.name = "Year"

    adjusted = False
    if df is not None and regress_on is not None:
        if regress_on not in df.columns:
            print(f"⚠️ Regressor '{regress_on}' not found in the provided DataFrame. Proceeding without adjustment.")
        else:
            df_drift = df_drift.merge(df[[regress_on]], left_index=True, right_index=True, how="left").dropna()
            X = df_drift[[regress_on]].values.reshape(-1, 1)
            y = df_drift["Drift"].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            df_drift["Drift_Adjusted"] = y - y_pred  # Residuals
            adjusted = True

    # Function to plot drift data
    def plot_drift(ax, years, drift, errors, label, title):
        ax.scatter(years, drift, color='blue', alpha=0.2, label=label)
        ax.errorbar(years, drift, yerr=errors, fmt='o', color='blue', alpha=0.3, label="Error bars")

        if smooth:
            smoothed = gaussian_filter1d(drift, sigma=sigma)
            ax.plot(years, smoothed, linestyle='--', color='red', linewidth=2, label=f"Smoothed (σ={sigma})")
            # Ensure window_length is smaller than data size and odd
            window_length = min(11, len(smoothed) if len(smoothed) % 2 == 1 else len(smoothed) - 1)
            polyorder = min(3, window_length - 1)
            derivative = savgol_filter(smoothed, window_length=window_length, polyorder=polyorder, deriv=1, delta=np.mean(np.diff(years)))

            if plot_derivative:
                ax2 = ax.twinx()
                ax2.plot(years, derivative, linestyle='-', color='green', linewidth=1, label="First Derivative")
                ax2.set_ylabel("Rate of Change")
                ax2.set_ylim(-0.005, 0.003)

                ax.legend(loc="upper left")
                ax2.legend(loc="upper right")
            else:
                ax.legend()

        # Add vertical line at start_year for consistency with global plot
        ax.axvline(x=start_year, color='green', linestyle=':', linewidth=2, alpha=0.7)

        # Set x-axis to include start_year even though no data point exists there
        ax.set_xlim(start_year - year_step * 0.5, end_year + year_step * 0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Change Magnitude")
        ax.set_title(title)
        ax.grid(True)

    # Plot Unadjusted Scores
    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        plot_drift(ax1, df_drift.index, df_drift["Drift"], df_drift["Error"], "Unadjusted Drift",
                   f"Year-over-Year Semantic Change {'for ' + word if word else ''}")
        plt.tight_layout()
        plt.show()

    # Plot Adjusted Scores if Regression was Performed
    if adjusted:
        fig, ax2 = plt.subplots(figsize=(10, 5))
        plot_drift(ax2, df_drift.index, df_drift["Drift_Adjusted"], df_drift["Error"], "Adjusted Drift (Residuals)",
                   f"Adjusted Year-over-Year Semantic Change {'for ' + word if word else ''}")
        plt.tight_layout()
        plt.show()

    return df_drift


def track_global_semantic_change(
    start_year, end_year, model_dir, reference_year=None, word=None, year_step=1,
    plot=True, smooth=False, sigma=2, confidence=0.95, error_type="CI", num_workers=None, plot_derivative=True
):
    """
    Track global semantic change (cumulative change from a reference year).

    Measures the scalar distance of word embeddings from a reference year by computing
    1 - cosine_similarity between each year and the reference. When word=None, computes
    average change across all shared vocabulary words.

    Args:
        start_year (int): Starting year of the range.
        end_year (int): Ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        reference_year (int, optional): Reference year to measure from. Defaults to start_year.
        word (str, optional): Specific word to track. If None, tracks all shared words.
        year_step (int): Step size for year increments (default: 1).
        plot (bool): Whether to plot the results.
        smooth (bool): Whether to apply Gaussian smoothing.
        sigma (float): Standard deviation for Gaussian smoothing.
        confidence (float): Confidence level for error bands.
        error_type (str): Either "CI" (confidence intervals) or "SE" (standard error).
        num_workers (int, optional): Number of parallel workers.

    Returns:
        pd.DataFrame: DataFrame with columns ['Distance', 'Error', 'Shared'] indexed by year.
    """
    if reference_year is None:
        reference_year = start_year

    change_data = {}
    missing_years = []

    # Find all model paths
    model_paths = {}
    for year in range(start_year, end_year + 1, year_step):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = sorted(glob.glob(model_pattern))
        if model_files:
            model_paths[year] = model_files[-1]
        else:
            missing_years.append(year)

    if reference_year not in model_paths:
        print(f"❌ Reference year {reference_year} not found. Exiting.")
        return pd.DataFrame()

    if not model_paths:
        print("❌ No valid models found in the specified range. Exiting.")
        return pd.DataFrame()

    reference_model_path = model_paths[reference_year]

    # Prepare multiprocessing arguments
    args = [(year, path, reference_model_path, reference_year, word) for year, path in model_paths.items()]

    num_workers = num_workers or min(cpu_count(), len(args))
    with Pool(num_workers) as pool:
        results = pool.map(compute_distance_from_reference, args)

    # Process results and compute error measures
    for year, distance, sd, shared in results:
        if distance is not None:
            # Compute error measure
            if error_type == "CI":
                error_measure = stats.norm.ppf(1 - (1 - confidence) / 2) * (sd / np.sqrt(shared))
            elif error_type == "SE":
                error_measure = sd / np.sqrt(shared) if shared > 1 else 0
            else:
                raise ValueError("Invalid error_type. Choose 'CI' or 'SE'.")

            change_data[year] = (distance, error_measure, shared)

    if missing_years:
        print(f"⚠️ No models found for these years: {missing_years}")

    if not change_data:
        print("❌ No valid distance scores computed. Exiting.")
        return pd.DataFrame()

    df_change = pd.DataFrame.from_dict(change_data, orient="index", columns=["Distance", "Error", "Shared"])
    df_change.index.name = "Year"

    # Plot results
    if plot:
        # Exclude reference year from plot (it's always 0)
        df_plot = df_change[df_change.index != reference_year]

        years = df_plot.index.values
        distances = df_plot["Distance"].values
        errors = df_plot["Error"].values

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(years, distances, color='blue', alpha=0.2, label="Distance from reference")
        ax.errorbar(years, distances, yerr=errors, fmt='o', color='blue', alpha=0.3, label="Error bars")

        if smooth:
            smoothed = gaussian_filter1d(distances, sigma=sigma)
            ax.plot(years, smoothed, linestyle='--', color='red', linewidth=2, label=f"Smoothed (σ={sigma})")

            # Compute first derivative
            window_length = min(11, len(smoothed) if len(smoothed) % 2 == 1 else len(smoothed) - 1)
            polyorder = min(3, window_length - 1)
            derivative = savgol_filter(smoothed, window_length=window_length, polyorder=polyorder, deriv=1, delta=np.mean(np.diff(years)))

            if plot_derivative:
                ax2 = ax.twinx()
                ax2.plot(years, derivative, linestyle='-', color='green', linewidth=1, label="First Derivative")
                ax2.set_ylabel("Rate of Change")
                ax2.set_ylim(-0.005, 0.003)

                ax.legend(loc="upper left")
                ax2.legend(loc="upper right")
            else:
                ax.legend()
        else:
            ax.legend()

        ax.axvline(x=reference_year, color='green', linestyle=':', linewidth=2, label=f"Reference year ({reference_year})")

        # Set x-axis to include start_year even though reference year data point is excluded from plot
        ax.set_xlim(start_year - year_step * 0.5, end_year + year_step * 0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Distance from Reference")
        ax.set_title(f"Global Semantic Change from {reference_year} {'for ' + word if word else ''}")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return df_change


def track_directional_drift(
    start_year, end_year, model_dir, word=None, year_step=1, plot=True, smooth=False, sigma=2,
    num_workers=None, plot_trajectory=False, trajectory_3d=False, plot_derivative=True
):
    """
    Track directional drift by analyzing change vectors between consecutive years.

    This function preserves directionality information by computing change vectors
    (differences in embeddings) and measuring the cosine similarity between consecutive
    change vectors. This reveals whether changes are consistent (same direction),
    oscillating (opposite directions), or orthogonal (different dimensions).

    Args:
        start_year (int): Starting year of the range.
        end_year (int): Ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        word (str, optional): Specific word to track. If None, computes average across all shared vocabulary.
        year_step (int): Step size for year increments (default: 1).
        plot (bool): Whether to plot the magnitude and direction consistency results.
        smooth (bool): Whether to apply Gaussian smoothing.
        sigma (float): Standard deviation for Gaussian smoothing.
        num_workers (int, optional): Number of parallel workers.
        plot_trajectory (bool): Whether to plot the PCA trajectory showing the path through semantic space.
                               Only available for single word analysis (word != None).
        trajectory_3d (bool): Whether to plot trajectory in 3D (default: False for 2D).

    Returns:
        pd.DataFrame: DataFrame with columns ['Magnitude', 'Direction_Consistency', 'Embedding'] indexed by year.
                     Direction_Consistency shows cosine similarity between consecutive change vectors:
                     ~1.0 = consistent drift, ~-1.0 = oscillation, ~0.0 = orthogonal changes.
                     When word=None, returns average values across all shared vocabulary.
    """

    drift_data = {}
    missing_years = []

    # Find all model paths
    model_paths = {}
    for year in range(start_year, end_year + 1, year_step):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = sorted(glob.glob(model_pattern))
        if model_files:
            model_paths[year] = model_files[-1]
        else:
            missing_years.append(year)

    if len(model_paths) < 2:
        print("❌ Need at least 2 years of models for directional drift analysis. Exiting.")
        return pd.DataFrame()

    years_available = sorted(model_paths.keys())

    # Single word analysis
    if word is not None:
        print(f"Loading embeddings for word '{word}' across {len(years_available)} years...")
        embeddings = {}
        for year in years_available:
            try:
                model = W2VModel(model_paths[year])
                if word in model.vocab:
                    embeddings[year] = model.model[word].copy()
                else:
                    print(f"⚠️ Word '{word}' not found in year {year}")
            except Exception as e:
                print(f"❌ Error loading model for year {year}: {e}")

        if missing_years:
            print(f"⚠️ No models found for these years: {missing_years}")

        if len(embeddings) < 2:
            print("❌ Need embeddings from at least 2 years. Exiting.")
            return pd.DataFrame()

        # Compute change vectors for single word
        years_sorted = sorted(embeddings.keys())
        change_vectors = {}

        for i in range(len(years_sorted) - 1):
            year_curr = years_sorted[i]
            year_next = years_sorted[i + 1]
            change_vector = embeddings[year_next] - embeddings[year_curr]
            change_magnitude = np.linalg.norm(change_vector)
            change_vectors[year_next] = {
                'vector': change_vector,
                'magnitude': change_magnitude,
                'embedding': embeddings[year_next]
            }

    # Vocabulary-wide analysis
    else:
        print(f"Computing average directional drift across shared vocabulary for {len(years_available)} years...")

        # Load all models
        models = {}
        for year in years_available:
            try:
                models[year] = W2VModel(model_paths[year])
            except Exception as e:
                print(f"❌ Error loading model for year {year}: {e}")

        if len(models) < 2:
            print("❌ Need at least 2 years of models. Exiting.")
            return pd.DataFrame()

        years_sorted = sorted(models.keys())
        change_vectors = {}

        # For each consecutive year pair, compute average change vectors
        for i in range(len(years_sorted) - 1):
            year_curr = years_sorted[i]
            year_next = years_sorted[i + 1]

            # Find shared vocabulary
            shared_vocab = models[year_curr].vocab.intersection(models[year_next].vocab)

            if len(shared_vocab) == 0:
                print(f"⚠️ No shared vocabulary between {year_curr} and {year_next}")
                continue

            # Compute change vectors for all shared words
            magnitudes = []
            change_vecs = []

            for w in shared_vocab:
                change_vec = models[year_next].model[w] - models[year_curr].model[w]
                change_vecs.append(change_vec)
                magnitudes.append(np.linalg.norm(change_vec))

            # Store average magnitude and change vectors
            change_vectors[year_next] = {
                'vector': np.mean(change_vecs, axis=0),  # Average change vector
                'magnitude': np.mean(magnitudes),  # Average magnitude
                'embedding': None,  # Not applicable for vocab-wide analysis
                'n_words': len(shared_vocab)
            }

    # Compute direction consistency (cosine similarity between consecutive change vectors)
    change_years = sorted(change_vectors.keys())
    for i in range(len(change_years)):
        year = change_years[i]

        if i == 0:
            # First change vector has no previous to compare to
            drift_data[year] = {
                'Magnitude': change_vectors[year]['magnitude'],
                'Direction_Consistency': np.nan,
                'Embedding': change_vectors[year]['embedding']
            }
        else:
            prev_year = change_years[i - 1]
            prev_vector = change_vectors[prev_year]['vector']
            curr_vector = change_vectors[year]['vector']

            # Compute cosine similarity between change vectors
            prev_norm = np.linalg.norm(prev_vector)
            curr_norm = np.linalg.norm(curr_vector)

            if prev_norm > 0 and curr_norm > 0:
                direction_consistency = np.dot(prev_vector, curr_vector) / (prev_norm * curr_norm)
            else:
                direction_consistency = np.nan

            drift_data[year] = {
                'Magnitude': change_vectors[year]['magnitude'],
                'Direction_Consistency': direction_consistency,
                'Embedding': change_vectors[year]['embedding']
            }

    df_drift = pd.DataFrame.from_dict(drift_data, orient="index")
    df_drift.index.name = "Year"

    # Plot results
    if plot:
        years = df_drift.index.values
        magnitudes = df_drift["Magnitude"].values
        consistencies = df_drift["Direction_Consistency"].values

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot 1: Change magnitude
        ax1.scatter(years, magnitudes, color='blue', alpha=0.5, label="Change magnitude")
        ax1.plot(years, magnitudes, color='blue', alpha=0.3, linestyle='-')

        if smooth and len(magnitudes) > 3:
            smoothed_mag = gaussian_filter1d(magnitudes, sigma=sigma)
            ax1.plot(years, smoothed_mag, linestyle='--', color='red', linewidth=2, label=f"Smoothed (σ={sigma})")

        ax1.set_ylabel("Change Magnitude")
        title = f"Directional Drift Analysis for '{word}'" if word else "Directional Drift Analysis (Vocabulary Average)"
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Direction consistency
        ax2.scatter(years[1:], consistencies[1:], color='green', alpha=0.5, label="Direction consistency")
        ax2.plot(years[1:], consistencies[1:], color='green', alpha=0.3, linestyle='-')

        if smooth and len(consistencies[1:]) > 3:
            smoothed_cons = gaussian_filter1d(consistencies[1:], sigma=sigma)
            ax2.plot(years[1:], smoothed_cons, linestyle='--', color='red', linewidth=2, label=f"Smoothed (σ={sigma})")

        ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Direction Consistency")
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # ✅ PCA Trajectory Plot
        if plot_trajectory:
            if word is None:
                print("⚠️ PCA trajectory plot only available for single word analysis (not vocabulary-wide average).")
            else:
                from sklearn.decomposition import PCA

                # Collect embeddings from DataFrame (sort by year ascending)
                df_sorted = df_drift.sort_index()
                embeddings = np.array([emb for emb in df_sorted['Embedding'].values])
                years_array = df_sorted.index.values

                if trajectory_3d:
                    # 3D trajectory plot
                    from mpl_toolkits.mplot3d import Axes3D

                    pca = PCA(n_components=3)
                    embeddings_pca = pca.fit_transform(embeddings)

                    # Flip PC1 if negatively correlated with time
                    if np.corrcoef(years_array, embeddings_pca[:, 0])[0, 1] < 0:
                        embeddings_pca[:, 0] = -embeddings_pca[:, 0]

                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection='3d')

                    # Plot trajectory line
                    ax.plot(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2],
                           'o-', color='steelblue', linewidth=2, markersize=6, alpha=0.7)

                    # Add year labels
                    for i, year in enumerate(years_array):
                        ax.text(embeddings_pca[i, 0], embeddings_pca[i, 1], embeddings_pca[i, 2],
                               str(year), fontsize=8)

                    # Highlight start and end
                    ax.scatter(embeddings_pca[0, 0], embeddings_pca[0, 1], embeddings_pca[0, 2],
                              color='green', s=200, label=f'Start ({years_array[0]})', zorder=5)
                    ax.scatter(embeddings_pca[-1, 0], embeddings_pca[-1, 1], embeddings_pca[-1, 2],
                              color='red', s=200, label=f'End ({years_array[-1]})', zorder=5)

                    # Calculate return distance
                    return_distance = np.linalg.norm(embeddings_pca[-1] - embeddings_pca[0])

                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
                    ax.set_title(f'3D Semantic Trajectory of "{word}" in PCA Space\nReturn Distance: {return_distance:.4f}')
                    ax.legend()
                    plt.tight_layout()
                    plt.show()

                else:
                    # 2D trajectory plot
                    pca = PCA(n_components=2)
                    embeddings_2d = pca.fit_transform(embeddings)

                    # Flip PC1 if it's negatively correlated with time (for left-to-right chronology)
                    if np.corrcoef(years_array, embeddings_2d[:, 0])[0, 1] < 0:
                        embeddings_2d[:, 0] = -embeddings_2d[:, 0]

                    # Create trajectory plot
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Plot trajectory line
                    ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'o-',
                           color='steelblue', linewidth=2, markersize=6, alpha=0.7)

                    # Add year labels
                    for i, year in enumerate(years_array):
                        ax.annotate(str(year), (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                                   textcoords="offset points", xytext=(5, 5), fontsize=8)

                    # Highlight start and end
                    ax.plot(embeddings_2d[0, 0], embeddings_2d[0, 1], 'go', markersize=12,
                           label=f'Start ({years_array[0]})', zorder=5)
                    ax.plot(embeddings_2d[-1, 0], embeddings_2d[-1, 1], 'ro', markersize=12,
                           label=f'End ({years_array[-1]})', zorder=5)

                    # Calculate and display return distance
                    return_distance = np.linalg.norm(embeddings_2d[-1] - embeddings_2d[0])
                    ax.text(0.02, 0.98, f'Return Distance: {return_distance:.4f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                    ax.set_title(f'Semantic Trajectory of "{word}" in PCA Space')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()

    return df_drift
