"""
Time series projection of words onto semantic dimensions (PCA or mean-diff).

Analogous to WEAT-over-years, this utility:
1) For each year, fits a semantic dimension using either PCA or mean-diff
2) Projects specified words onto that year's own dimension
3) Optionally plots trajectories for quick visual inspection

NOTE: Each year uses its own fitted dimension (not a reference year dimension),
allowing the dimension itself to drift over time as the model and language evolve.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.stats import linregress

from ngramprep.common.w2v_model import W2VModel


def compute_projection_over_years(
    model_dir: Union[str, Path],
    token_contrasts: Sequence[tuple],
    test_words: Sequence[str],
    start_year: int,
    end_year: int,
    year_step: int = 1,
    method: str = "meandiff",
    reference_year: Optional[int] = None,
    ensure_sign_positive: Optional[Union[bool, List[str]]] = True,
    plot: bool = True,
    smooth: bool = False,
    sigma: float = 2,
    plot_regression: bool = False,
    verbose: bool = True,
    baseline_result: Optional[Dict[str, object]] = None,
    baseline_words: Optional[Sequence[str]] = None,
    baseline_agg: str = "median",
    plot_corrected_if_baseline: bool = True,
    **method_kwargs,
) -> Dict[str, object]:
    """
    Project words onto a semantic dimension across yearly models.

    Args:
        model_dir: Directory containing yearly word2vec models (*.kv).
        token_contrasts: List of (token1, token2) pairs defining the dimension.
        test_words: Words to project over time.
        start_year: First year to include (must match available models).
        end_year: Last year to include (inclusive).
        year_step: Step between years (should align with training cadence).
        method: 'pca' or 'meandiff'.
        reference_year: Year to fit the dimension on (defaults to earliest available in requested range).
        ensure_sign_positive: Passed to PCA variant for sign orientation. Ignored for mean-diff.
        plot: Whether to produce a trajectory plot.
        smooth: Apply Gaussian smoothing to trajectories for plotting only.
        sigma: Standard deviation for Gaussian smoothing (same pattern as WEAT series).
        plot_regression: If True, overlay linear regression lines on the plot for each word's trajectory.
        verbose: Print progress information.
        baseline_result: Optional baseline set object returned by compute_baseline_set.
            If provided, baseline-corrected projections are computed and returned. When
            plotting, corrected series are shown if plot_corrected_if_baseline=True.
            Mutually exclusive with baseline_words.
        baseline_words: Optional list of words to use as baseline. If provided, these
            words will be projected onto the dimension for each year and aggregated
            (mean or median) to produce a yearly baseline for correction. Mutually
            exclusive with baseline_result.
        baseline_agg: Aggregation method for baseline_words ('mean' or 'median').
            Only used when baseline_words is provided. Default: 'median'.
        plot_corrected_if_baseline: If True and baseline_result or baseline_words is
            provided, plot baseline-corrected trajectories; otherwise plot raw projections.
        **method_kwargs: Extra kwargs forwarded to dimension computation methods.

    Returns:
        dict with keys:
            'dimension' (None): Set to None since year-specific dimensions are used.
            'reference_year' (int): Original reference_year parameter (for backwards compatibility).
            'method' (str): 'pca' or 'meandiff'.
            'projections' (pd.DataFrame): Index=years, columns=test_words. Projections onto each year's own dimension.
            'component_loadings' (dict): Empty dict (loadings vary per year).
            'yearly_dimensions' (dict): Year -> dimension vector mapping for all years analyzed.
            'missing_years' (list): Years with no matching model files.
            'error_years' (dict): Years that raised errors during processing.
    """

    model_dir = Path(model_dir)

    # Discover available yearly models
    model_files = sorted(model_dir.glob("*.kv"))
    if not model_files:
        raise FileNotFoundError(f"No .kv model files found in {model_dir}")

    available_years = []
    year_to_path = {}
    for f in model_files:
        year = None
        stem_parts = f.stem.split("_")
        for part in stem_parts:
            if part.startswith("y") and len(part) > 1:
                try:
                    year = int(part[1:])
                    break
                except ValueError:
                    pass
        if year is None:
            try:
                year = int(stem_parts[-1])
            except ValueError:
                if verbose:
                    print(f"⚠️ Skipping file with non-standard name: {f.name}")
                continue
        available_years.append(year)
        year_to_path[year] = f

    available_years = sorted(set(available_years))
    if not available_years:
        raise ValueError("No valid year-parsable model filenames found.")

    requested_years = list(range(start_year, end_year + 1, year_step))
    years_to_analyze = sorted([y for y in requested_years if y in year_to_path])
    missing_years = [y for y in requested_years if y not in year_to_path]

    if not years_to_analyze:
        raise ValueError(
            f"No requested years found in models. Requested {requested_years}, available {available_years}"
        )

    # Choose reference year
    if reference_year is None:
        reference_year = years_to_analyze[0]
    elif reference_year not in years_to_analyze:
        raise ValueError(
            f"Reference year {reference_year} not available in requested range. Available: {years_to_analyze}"
        )

    if verbose:
        print(f"📈 Dimension projections: {len(years_to_analyze)} years [{min(years_to_analyze)}-{max(years_to_analyze)}]")
        print(f"   Method: {method}")
        print(f"   Contrast pairs: {len(token_contrasts)}")
        print(f"   Computing year-specific dimensions for each year...")

    projections_data: Dict[int, Dict[str, float]] = {}
    error_years: Dict[int, str] = {}
    yearly_dimensions: Dict[int, np.ndarray] = {}

    # For each year, compute its own dimension and project words onto it
    for year in years_to_analyze:
        model_path = year_to_path.get(year)

        if verbose:
            print(f"   {year}...", end=" ")

        try:
            model = W2VModel(str(model_path))
            
            # Compute dimension for THIS year
            if method.lower() == "pca":
                dimension_result = model.compute_pca_dimension(
                    token_contrasts=token_contrasts,
                    ensure_sign_positive=ensure_sign_positive,
                    **method_kwargs,
                )
            elif method.lower() == "meandiff":
                dimension_result = model.compute_meandiff_dimension(
                    token_contrasts=token_contrasts,
                    **method_kwargs,
                )
            else:
                raise ValueError("method must be 'pca' or 'meandiff'")
            
            dimension = dimension_result["dimension"]
            yearly_dimensions[year] = dimension
            
            # Project words onto THIS year's dimension
            row = {}
            for word in test_words:
                if word in model.vocab:
                    try:
                        row[word] = model.project_onto_dimension(word, dimension)
                    except ValueError:
                        row[word] = np.nan
                else:
                    row[word] = np.nan
            projections_data[year] = row
            if verbose:
                valid = sum(1 for v in row.values() if not pd.isna(v))
                print(f"✓ {valid}/{len(test_words)} words")
        except Exception as exc:  # noqa: BLE001
            error_years[year] = str(exc)
            if verbose:
                print(f"⚠️ error: {exc}")

    if missing_years and verbose:
        print(f"⚠️ No models found for years: {missing_years}")
    if error_years and verbose:
        print("❌ Errors occurred:")
        for y, msg in error_years.items():
            print(f"   {y}: {msg}")

    if not projections_data:
        return {
            "dimension": None,
            "reference_year": reference_year,
            "method": method,
            "projections": pd.DataFrame(),
            "component_loadings": {},
            "missing_years": missing_years,
            "error_years": error_years,
            "yearly_dimensions": yearly_dimensions,
        }

    projections_df = pd.DataFrame.from_dict(projections_data, orient="index")
    projections_df = projections_df.sort_index()

    # Optionally apply baseline correction
    projections_corrected_df: Optional[pd.DataFrame] = None
    baseline_applied = False
    aligned_baseline = pd.Series(dtype=float)
    
    # Check for mutually exclusive baseline parameters
    if baseline_result is not None and baseline_words is not None:
        raise ValueError("baseline_result and baseline_words are mutually exclusive; provide only one.")
    
    # Option 1: Use pre-computed baseline from compute_baseline_set
    if baseline_result is not None:
        baseline_series = baseline_result.get("baseline") if isinstance(baseline_result, dict) else None
        if isinstance(baseline_series, pd.Series) and not projections_df.empty:
            # Align baseline to available years
            aligned_baseline = baseline_series.reindex(projections_df.index)
            aligned_baseline = aligned_baseline.fillna(0.0)  # leave raw values where baseline is unavailable
            projections_corrected_df = projections_df.sub(aligned_baseline, axis=0)
            baseline_applied = True
    
    # Option 2: Compute baseline from user-specified words
    elif baseline_words is not None:
        if baseline_agg not in ["mean", "median"]:
            raise ValueError("baseline_agg must be 'mean' or 'median'.")
        
        # Check which baseline words are available in projections
        available_baseline_words = [w for w in baseline_words if w in projections_df.columns]
        
        if not available_baseline_words:
            if verbose:
                print(f"⚠️ None of the {len(baseline_words)} baseline words are in the projection set.")
        else:
            if verbose and len(available_baseline_words) < len(baseline_words):
                missing = set(baseline_words) - set(available_baseline_words)
                print(f"⚠️ {len(missing)} baseline words not found: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
            
            # Aggregate baseline words
            if baseline_agg == "median":
                aligned_baseline = projections_df[available_baseline_words].median(axis=1)
            else:  # mean
                aligned_baseline = projections_df[available_baseline_words].mean(axis=1)
            
            aligned_baseline = aligned_baseline.fillna(0.0)
            projections_corrected_df = projections_df.sub(aligned_baseline, axis=0)
            baseline_applied = True
            
            if verbose:
                print(f"✓ Baseline computed from {len(available_baseline_words)} words using {baseline_agg}")

    if plot:
        plt.figure(figsize=(10, 5))
        for word in test_words:
            if word not in projections_df.columns:
                continue
            # Choose series: corrected (if requested and available) or raw
            if baseline_applied and plot_corrected_if_baseline and projections_corrected_df is not None:
                series = projections_corrected_df[word]
            else:
                series = projections_df[word]
            if smooth:
                values = series.values
                if np.all(np.isnan(values)):
                    smoothed = values
                else:
                    filled = np.nan_to_num(values, nan=np.nanmean(values))
                    smoothed = gaussian_filter1d(filled, sigma=sigma)
                series_to_plot = pd.Series(smoothed, index=series.index)
            else:
                series_to_plot = series

            label = word if not (baseline_applied and plot_corrected_if_baseline) else f"{word} (baseline-corrected)"
            plt.plot(series_to_plot.index, series_to_plot.values, marker="o", linestyle="-", label=label)

        plt.xlabel("Year", fontsize=12)
        ylabel = "Projection (cosine)"
        title_suffix = ""
        if baseline_applied and plot_corrected_if_baseline:
            ylabel = "Baseline-corrected projection (cosine)"
            title_suffix = " (baseline-corrected)"
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"Word projections on {method} dimension{title_suffix}", fontsize=14, fontweight="bold")
        
        # Add regression lines if requested
        if plot_regression:
            for word in test_words:
                if word not in projections_df.columns:
                    continue
                # Choose series: corrected (if requested and available) or raw
                if baseline_applied and plot_corrected_if_baseline and projections_corrected_df is not None:
                    series = projections_corrected_df[word]
                else:
                    series = projections_df[word]
                
                # Drop NaN values for regression
                valid_mask = ~series.isna()
                years_valid = series.index[valid_mask].values
                values_valid = series.values[valid_mask]
                
                if len(years_valid) >= 2:
                    # Fit linear regression
                    slope, intercept, r_value, p_value, std_err = linregress(years_valid, values_valid)
                    # Plot regression line with p-value in label
                    years_range = np.array([years_valid.min(), years_valid.max()])
                    line_values = slope * years_range + intercept
                    sig_marker = "*" if p_value < 0.05 else ""
                    plt.plot(years_range, line_values, linestyle='--', alpha=0.7, linewidth=1.5, 
                            label=f"{word} regression (p={p_value:.4f}){sig_marker}")
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "dimension": None,  # Year-specific: no single reference dimension
        "reference_year": reference_year,
        "method": method,
        "projections": projections_df,
        "projections_corrected": projections_corrected_df if projections_corrected_df is not None else pd.DataFrame(),
        "baseline_aligned": aligned_baseline if baseline_applied else pd.Series(dtype=float),
        "baseline_applied": baseline_applied,
        "component_loadings": {},  # Year-specific: loadings vary by year
        "yearly_dimensions": yearly_dimensions,  # NEW: dimensions for each year
        "missing_years": missing_years,
        "error_years": error_years,
    }


def compute_baseline_set(
    model_dir: str,
    contrast_pairs: Sequence[tuple],
    test_words: Optional[Sequence[str]] = None,
    start_year: int = 1900,
    end_year: int = 2019,
    year_step: int = 1,
    method: str = 'meandiff',
    ensure_sign_positive: bool = True,
    exclusion_pattern: Optional[Union[str, re.Pattern]] = None,
    eps_mean: float = 0.03,
    eps_trend: float = 0.002,
    eps_sigma: float = 0.05,
    min_years: int = 10,
    agg: str = "median",
    plot: bool = False,
    plot_baseline: bool = True,
    corr_n_permutations: int = 1000,
    corr_random_state: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Build a neutral word set and compute a yearly baseline from projection series.

    This function implements an algorithmic neutral baseline correction by:
    1) Computing projections for vocabulary across years using compute_projection_over_years
    2) Extracting anchor words from contrast pairs and applying exclusion patterns
    3) Building a neutral candidate pool from available vocabulary
    4) Computing time-series statistics (mean, trend, variability) for each candidate
    5) Selecting words with near-zero mean projection, minimal secular drift, and low noise
    6) Computing a yearly baseline as the median/mean of stable neutral words

    Typical workflow:
        baseline_result = compute_baseline_set(
            model_dir=model_path,
            contrast_pairs=gender_contrasts,
            test_words=targets,
            exclusion_pattern=r"(man|men|woman|women|girl|boy|wife|husband)",
        )
        corrected = baseline_result["projections"][targets].sub(baseline_result["baseline"], axis=0)

    Args:
        model_dir: Path to directory containing year-specific Word2Vec models.
        contrast_pairs: List of (token1, token2) tuples defining the semantic dimension.
            All tokens from these pairs are automatically excluded from the neutral pool.
        test_words: Optional list of specific words to include in projections (e.g., target
            words for analysis). If None, only vocabulary shared across all models is used.
        start_year: First year to process (default: 1900).
        end_year: Last year to process (default: 2019).
        year_step: Year interval (default: 1).
        method: Dimension extraction method ('pca' or 'meandiff', default: 'meandiff').
        ensure_sign_positive: If True, ensure first contrast token in first pair has
            positive projection (default: True).
        exclusion_pattern: Optional regex pattern (str or compiled) to exclude additional
            words by substring matching (e.g., morphological variants, semantic category
            members). Applied case-insensitively. If None, only contrast tokens are excluded.
        eps_mean: Threshold for mean projection magnitude (default: 0.03). Words with
            |mean| < eps_mean are considered near-neutral on average.
        eps_trend: Threshold for secular trend (default: 0.002 projection units per year).
            Words with |beta| < eps_trend show minimal long-term drift.
        eps_sigma: Threshold for temporal variability (default: 0.05). Words with
            std < eps_sigma are temporally stable.
        min_years: Minimum number of valid (non-NaN) years required to compute stats
            (default: 10).
        agg: Aggregation method for yearly baseline ("median" or "mean"). Median is more
            robust to outliers (default: "median").
        plot: Forwarded to compute_projection_over_years; defaults to False to avoid
            plotting thousands of trajectories when using full vocabulary baselines.
        plot_baseline: If True, plot the yearly baseline (mean/median projection across
            neutral words) after selection.
        corr_n_permutations: Number of null permutations for the over-time correlation
            test (per-word year shuffles). Set to 0 to skip entirely (default: 1000).
        corr_random_state: Seed for reproducible permutation testing (default: None).
        verbose: If True, print progress information (default: False).

    Returns:
        dict with keys:
            'projections' (pd.DataFrame): Full projection DataFrame (years × words).
            'stats' (pd.DataFrame): Per-word statistics (mu, beta, sigma, n) for all
                neutral candidates.
            'neutral_words' (list): Words that passed all stability thresholds.
            'baseline' (pd.Series): Yearly baseline values (median or mean across neutral
                words), indexed by year.
            'neutral_candidates' (list): All words considered before stability filtering.
            'correlation_stats' (dict): Statistics about over-time correlations between
                neutral words, with keys: 'mean', 'median', 'std', 'min', 'max',
                'all_correlations' (array of pairwise correlations).

    Raises:
        ValueError: If no words pass the selection criteria (thresholds too strict),
            if agg is not "median" or "mean", or if no neutral candidates remain after
            exclusion filtering.

    Example:
        >>> baseline_result = compute_baseline_set(
        ...     model_dir=model_path,
        ...     contrast_pairs=gender_contrasts,
        ...     test_words=["physician"],
        ...     exclusion_pattern=r"(man|men|woman|women|mother|father)",
        ...     eps_mean=0.03,
        ...     eps_trend=0.002,
        ... )
        >>> print(f"Neutral candidates: {len(baseline_result['neutral_candidates'])}")
        >>> print(f"Selected stable words: {len(baseline_result['neutral_words'])}")
        >>> corrected = baseline_result["projections"][["physician"]].sub(baseline_result["baseline"], axis=0)
    """
    
    # Step 0: If test_words is None, build shared vocabulary from all models
    if test_words is None:
        from pathlib import Path
        model_dir = Path(model_dir)
        model_files = sorted(model_dir.glob("*.kv"))
        
        if not model_files:
            raise FileNotFoundError(f"No .kv model files found in {model_dir}")
        
        # Get shared vocabulary across all requested years
        shared_vocab = None
        for model_file in model_files:
            model = W2VModel(str(model_file))
            # Handle vocab as either set or dictionary
            if hasattr(model.vocab, 'keys'):
                model_vocab = set(model.vocab.keys())
            else:
                model_vocab = set(model.vocab)
            
            if shared_vocab is None:
                shared_vocab = model_vocab
            else:
                shared_vocab = shared_vocab.intersection(model_vocab)
        
        test_words = list(shared_vocab)
        if verbose:
            print(f"Using shared vocabulary: {len(test_words)} words found across all models")
    
    # Step 1: Compute projections for all vocabulary
    projection_result = compute_projection_over_years(
        model_dir=model_dir,
        token_contrasts=contrast_pairs,
        test_words=test_words,
        start_year=start_year,
        end_year=end_year,
        year_step=year_step,
        method=method,
        ensure_sign_positive=ensure_sign_positive,
        smooth=False,
        plot=plot,
        verbose=False,
    )
    
    projections = projection_result["projections"]

    # Step 2: Build neutral candidate pool
    anchors = {tok for pair in contrast_pairs for tok in pair}
    vocab = list(projections.columns)
    
    # Compile exclusion pattern if provided
    if exclusion_pattern is not None:
        if isinstance(exclusion_pattern, str):
            pattern = re.compile(exclusion_pattern, re.IGNORECASE)
        else:
            pattern = exclusion_pattern
        neutral_candidates = [w for w in vocab if w not in anchors and not pattern.search(w)]
    else:
        neutral_candidates = [w for w in vocab if w not in anchors]
    
    if not neutral_candidates:
        raise ValueError(
            f"No neutral candidates remain after exclusion filtering. "
            f"Anchors: {len(anchors)}, Total vocab: {len(vocab)}"
        )

    def fit_stats(series: pd.Series) -> pd.Series:
        """Compute time-series stats: mean, OLS trend slope, std dev, sample size."""
        s = series.dropna()
        if len(s) < 5:
            return pd.Series({"mu": np.nan, "beta": np.nan, "sigma": np.nan, "n": len(s)})
        slope = linregress(s.index.values, s.values).slope
        return pd.Series({"mu": s.mean(), "beta": slope, "sigma": s.std(), "n": len(s)})

    stats = projections[neutral_candidates].apply(fit_stats, axis=0).T

    mask = (
        (stats["n"] >= min_years)
        & (stats["mu"].abs() < eps_mean)
        & (stats["beta"].abs() < eps_trend)
        & (stats["sigma"] < eps_sigma)
    )
    neutral_words = stats[mask].index.tolist()

    if not neutral_words:
        raise ValueError(
            f"No neutral words selected with thresholds: eps_mean={eps_mean}, "
            f"eps_trend={eps_trend}, eps_sigma={eps_sigma}, min_years={min_years}. "
            f"Started with {len(neutral_candidates)} candidates. Relax thresholds."
        )

    # Compute baseline
    if agg == "median":
        baseline = projections[neutral_words].median(axis=1)
    elif agg == "mean":
        baseline = projections[neutral_words].mean(axis=1)
    else:
        raise ValueError("agg must be 'median' or 'mean'.")

    def _upper_triangle_values(corr_df: pd.DataFrame) -> np.ndarray:
        """Return upper-triangular (k=1) correlation values or empty if <2 cols."""
        if corr_df.shape[0] < 2:
            return np.array([])
        mask_local = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
        return corr_df.where(mask_local).stack().values

    # Compute correlation statistics among neutral words
    correlation_stats: Dict[str, object] = {
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "min": np.nan,
        "max": np.nan,
        "all_correlations": np.array([]),
        "p_value": np.nan,
        "null_mean": np.nan,
        "null_std": np.nan,
        "null_distribution": np.array([]),
        "method": "permute_years",
        "n_permutations": corr_n_permutations,
    }

    if len(neutral_words) >= 2:
        neutral_proj = projections[neutral_words].T  # words × years
        corr_matrix = neutral_proj.corr()  # Pairwise correlations

        correlations = _upper_triangle_values(corr_matrix)

        avg_correlation = np.mean(correlations)
        median_correlation = np.median(correlations)
        std_correlation = np.std(correlations)

        correlation_stats.update({
            "mean": avg_correlation,
            "median": median_correlation,
            "std": std_correlation,
            "min": correlations.min(),
            "max": correlations.max(),
            "all_correlations": correlations,
        })

        # Permutation test: shuffle years within each word to destroy shared temporal structure
        if corr_n_permutations and corr_n_permutations > 0:
            rng = np.random.default_rng(corr_random_state)
            values = projections[neutral_words].values  # years × words
            null_means = np.empty(corr_n_permutations, dtype=float)

            for i in range(corr_n_permutations):
                permuted = np.empty_like(values)
                for col_idx in range(values.shape[1]):
                    col = values[:, col_idx]
                    permuted[:, col_idx] = rng.permutation(col)
                perm_df = pd.DataFrame(permuted, index=projections.index, columns=neutral_words)
                perm_corr = perm_df.corr()
                perm_vals = _upper_triangle_values(perm_corr)
                null_means[i] = np.mean(perm_vals) if len(perm_vals) else np.nan

            # Two-sided p-value against zero-correlated null
            valid_null = null_means[~np.isnan(null_means)]
            if len(valid_null):
                p_value = np.mean(np.abs(valid_null) >= abs(avg_correlation))
                correlation_stats.update({
                    "p_value": p_value,
                    "null_mean": float(np.mean(valid_null)),
                    "null_std": float(np.std(valid_null)),
                    "null_distribution": valid_null,
                })

    # Print summary statistics if verbose
    if verbose:
        print(f"\nBaseline word set selection:")
        print(f"  Total vocabulary:          {len(vocab):,}")
        print(f"  Neutral candidates:        {len(neutral_candidates):,} (after anchor & pattern exclusion)")
        print(f"  Selected neutral words:    {len(neutral_words):,} (passed stability thresholds)")
        print(f"\n  Selection thresholds:")
        print(f"    |mu| < {eps_mean:.3f}  (mean projection)")
        print(f"    |beta| < {eps_trend:.4f}  (trend/year)")
        print(f"    sigma < {eps_sigma:.3f}  (temporal variability)")
        print(f"    n >= {min_years}  (minimum years)")
        print(f"\n  Neutral word statistics:")
        print(f"    Mean |mu|:     {stats.loc[neutral_words, 'mu'].abs().mean():.4f}")
        print(f"    Mean |beta|:   {stats.loc[neutral_words, 'beta'].abs().mean():.4f}")
        print(f"    Mean sigma:    {stats.loc[neutral_words, 'sigma'].mean():.4f}")
        print(f"\n  Over-time correlation:")
        print(f"    Mean:          {correlation_stats['mean']:.4f}")
        print(f"    Median:        {correlation_stats['median']:.4f}")
        print(f"    Std:           {correlation_stats['std']:.4f}")
        print(f"    Range:         [{correlation_stats['min']:.4f}, {correlation_stats['max']:.4f}]")
        if corr_n_permutations > 0 and not np.isnan(correlation_stats.get('p_value', np.nan)):
            print(f"    Permutation p:  {correlation_stats['p_value']:.4f} (n={corr_n_permutations})")
        print(f"\n  Aggregation:               {agg}\n")

    if plot_baseline:
        plt.figure(figsize=(10, 4))
        plt.plot(
            baseline.index,
            baseline.values,
            marker="o",
            linestyle="-",
            label=f"Baseline ({agg}) across {len(neutral_words)} words",
        )
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Baseline projection (cosine)", fontsize=12)
        plt.title("Baseline trajectory over time", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "projections": projections,
        "stats": stats,
        "neutral_words": neutral_words,
        "baseline": baseline,
        "neutral_candidates": neutral_candidates,
        "correlation_stats": correlation_stats,
    }
