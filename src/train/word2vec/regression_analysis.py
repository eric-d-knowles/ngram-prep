"""
Regression analysis for Word2Vec evaluation results.

Performs mixed-effects regression to analyze the effects of hyperparameters
on model performance (similarity/analogy scores), accounting for clustering
within years.
"""

import os
import re
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import mixedlm, ols
from statsmodels.regression.mixed_linear_model import MixedLMResults
from statsmodels.regression.linear_model import RegressionResultsWrapper

__all__ = ["run_regression_analysis", "plot_regression_results"]


def _validate_inputs(
    csv_file: str,
    outcome: str,
    predictors: List[str],
    interactions: Optional[List[Tuple[str, str]]]
) -> Tuple[bool, Optional[str]]:
    """
    Validate input parameters for regression analysis.

    Args:
        csv_file: Path to evaluation results CSV
        outcome: Outcome variable ('similarity_score' or 'analogy_score')
        predictors: List of predictor variables
        interactions: List of interaction terms as tuples

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        return False, f"CSV file not found: {csv_file}"

    # Validate outcome
    valid_outcomes = ['similarity_score', 'analogy_score']
    if outcome not in valid_outcomes:
        return False, f"Outcome must be one of {valid_outcomes}, got: {outcome}"

    # Validate predictors
    valid_predictors = ['year', 'weight_by', 'vector_size', 'window',
                       'min_count', 'approach', 'epochs']
    invalid_predictors = [p for p in predictors if p not in valid_predictors]
    if invalid_predictors:
        return False, f"Invalid predictors: {invalid_predictors}. Valid: {valid_predictors}"

    # Validate interactions
    if interactions:
        for interaction in interactions:
            if len(interaction) != 2:
                return False, f"Each interaction must be a tuple of 2 variables, got: {interaction}"
            for var in interaction:
                if var not in valid_predictors:
                    return False, f"Invalid variable in interaction: {var}"

    return True, None


def _prepare_data(
    csv_file: str,
    outcome: str,
    predictors: List[str]
) -> pd.DataFrame:
    """
    Load and prepare data for regression analysis.

    Args:
        csv_file: Path to evaluation results CSV
        outcome: Outcome variable
        predictors: List of predictor variables

    Returns:
        Prepared DataFrame with outcome and predictors
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Check if outcome exists
    if outcome not in df.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in CSV")

    # Remove rows with missing outcome
    df = df.dropna(subset=[outcome])

    # Check if predictors exist
    missing_predictors = [p for p in predictors if p not in df.columns]
    if missing_predictors:
        raise ValueError(f"Predictors not found in CSV: {missing_predictors}")

    # Select relevant columns (avoid duplicates)
    cols_needed = [outcome] + predictors
    if 'year' not in cols_needed:
        cols_needed.append('year')  # Always need year for grouping
    df = df[cols_needed].copy()

    # Remove any rows with missing predictors
    df = df.dropna()

    return df


def _build_formula(
    outcome: str,
    predictors: List[str],
    interactions: Optional[List[Tuple[str, str]]] = None,
    standardize: bool = True
) -> str:
    """
    Build the regression formula for mixed-effects model.

    Args:
        outcome: Outcome variable
        predictors: List of predictor variables
        interactions: List of interaction terms as tuples
        standardize: Whether to standardize continuous predictors

    Returns:
        Formula string for statsmodels
    """
    # Identify categorical vs continuous predictors
    categorical = ['weight_by', 'approach']
    continuous = ['vector_size', 'window', 'min_count', 'epochs', 'year']

    formula_parts = []

    # Add main effects
    for pred in predictors:
        if pred in categorical:
            formula_parts.append(f"C({pred})")
        elif pred in continuous:
            if standardize:
                # Standardize continuous predictors for interpretability
                formula_parts.append(f"scale({pred})")
            else:
                formula_parts.append(pred)

    # Add interactions
    if interactions:
        for var1, var2 in interactions:
            # Determine if variables are categorical or continuous
            var1_cat = var1 in categorical
            var2_cat = var2 in categorical

            if var1_cat and var2_cat:
                formula_parts.append(f"C({var1}):C({var2})")
            elif var1_cat:
                if standardize:
                    formula_parts.append(f"C({var1}):scale({var2})")
                else:
                    formula_parts.append(f"C({var1}):{var2}")
            elif var2_cat:
                if standardize:
                    formula_parts.append(f"scale({var1}):C({var2})")
                else:
                    formula_parts.append(f"{var1}:C({var2})")
            else:
                if standardize:
                    formula_parts.append(f"scale({var1}):scale({var2})")
                else:
                    formula_parts.append(f"{var1}:{var2}")

    formula = f"{outcome} ~ " + " + ".join(formula_parts)
    return formula


def run_regression_analysis(
    csv_file: Optional[str] = None,
    outcome: Optional[str] = None,
    predictors: Optional[List[str]] = None,
    ngram_size: Optional[int] = None,
    repo_release_id: Optional[str] = None,
    repo_corpus_id: Optional[str] = None,
    db_path_stub: Optional[str] = None,
    dir_suffix: Optional[str] = None,
    interactions: Optional[List[Tuple[str, str]]] = None,
    random_effects: Optional[str] = 'year',
    standardize: bool = True,
    output_file: Optional[str] = None,
    verbose: bool = True,
    model_type: str = 'auto',
    corpus_path: Optional[str] = None,
    genre_focus: Optional[List[str]] = None
) -> Union[MixedLMResults, RegressionResultsWrapper]:
    """
    Run regression analysis on Word2Vec evaluation results.

    This function fits a regression model to assess the impact of
    hyperparameters on model performance. Can use mixed-effects (with random
    effects) or OLS regression.

    Can be called in three ways:
    1. Direct mode: Provide csv_file path directly
    2. Auto-detect mode (ngrams): Provide ngram_size, repo_release_id, repo_corpus_id, db_path_stub, and dir_suffix
    3. Auto-detect mode (Davies): Provide corpus_path, dir_suffix, and optionally genre_focus

    Args:
        csv_file: Path to evaluation results CSV (from evaluate_models). If None, will auto-detect.
        outcome: Outcome variable ('similarity_score' or 'analogy_score')
        predictors: List of predictor variables to include as fixed effects.
                   Valid options: 'year', 'weight_by', 'vector_size', 'window',
                   'min_count', 'approach', 'epochs'
        ngram_size (int, optional): N-gram size (e.g., 5 for 5grams) - used for ngram auto-detection
        repo_release_id (str, optional): Release date in YYYYMMDD format (e.g., "20200217") - used for ngram auto-detection
        repo_corpus_id (str, optional): Corpus identifier (e.g., "eng", "eng-fiction") - used for ngram auto-detection
        db_path_stub (str, optional): Base directory for data (e.g., "/scratch/edk202/NLP_corpora/Google_Books/") - used for ngram auto-detection
        dir_suffix (str, optional): Suffix for model/log directories (e.g., 'test', 'final') - used for auto-detection
        interactions: List of two-way interactions as tuples, e.g.,
                     [('year', 'vector_size'), ('year', 'epochs')]
        random_effects: Variable to use for random effects (default: 'year').
                       Only used if model_type is 'mixed' or 'auto'. Set to None
                       to force OLS regression.
        standardize: Whether to standardize continuous predictors (default: True)
        output_file: Optional path to save detailed results as text file
        verbose: Whether to print results to console (default: True)
        model_type: Type of model to fit. Options:
                   - 'auto': Try mixed-effects, fall back to OLS if it fails (default)
                   - 'mixed': Force mixed-effects model (will error if singular)
                   - 'ols': Force OLS regression (ignores random_effects)
        corpus_path (str, optional): Path to corpus directory (e.g., '/scratch/edk202/NLP_corpora/COHA') - used for Davies auto-detection
        genre_focus (list, optional): List of genres for Davies corpora (e.g., ['fic']) - used for Davies auto-detection

    Returns:
        MixedLMResults or RegressionResultsWrapper object containing model results

    Example:
        >>> from train.word2vec import run_regression_analysis
        >>>
        >>> # Direct mode - provide csv_file directly
        >>> results = run_regression_analysis(
        ...     csv_file='evaluation_results_test.csv',
        ...     outcome='similarity_score',
        ...     predictors=['year', 'vector_size', 'epochs']
        ... )
        >>>
        >>> # Auto-detect mode (Davies) - NEW PREFERRED METHOD
        >>> results = run_regression_analysis(
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     dir_suffix='test',
        ...     genre_focus=['fic'],
        ...     outcome='similarity_score',
        ...     predictors=['year', 'vector_size', 'epochs', 'approach'],
        ...     interactions=[('year', 'vector_size'), ('year', 'epochs')]
        ... )
        >>>
        >>> # Auto-detect mode (ngrams)
        >>> results = run_regression_analysis(
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng-fiction',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     dir_suffix='test',
        ...     outcome='similarity_score',
        ...     predictors=['year', 'vector_size', 'epochs', 'approach'],
        ...     interactions=[('year', 'vector_size'), ('year', 'epochs')]
        ... )
        >>>
        >>> # Complex model: main effects + interactions
        >>> results = run_regression_analysis(
        ...     csv_file='evaluation_results_test.csv',
        ...     outcome='similarity_score',
        ...     predictors=['year', 'vector_size', 'epochs', 'approach'],
        ...     interactions=[('year', 'vector_size'), ('year', 'epochs')],
        ...     output_file='regression_results.txt'
        ... )
    """
    # Auto-detect csv_file if path stub parameters are provided
    if csv_file is None:
        # New method: corpus_path (for Davies corpora)
        if corpus_path is not None and dir_suffix is not None:
            from .config import construct_model_path

            corpus_path = corpus_path.rstrip('/')
            model_base = construct_model_path(corpus_path)
            csv_file = os.path.join(model_base, f"evaluation_results_{dir_suffix}.csv")
        # Old method: ngram_size + repo_release_id + repo_corpus_id + db_path_stub (for ngrams)
        elif all(param is not None for param in [ngram_size, repo_release_id, repo_corpus_id, db_path_stub, dir_suffix]):
            # Construct path from stub parameters
            from ngramprep.ngram_acquire.db.build_path import build_db_path
            from pathlib import Path
            from .config import construct_model_path

            base_path = Path(build_db_path(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)).parent
            model_base = construct_model_path(str(base_path))
            csv_file = os.path.join(model_base, f"evaluation_results_{dir_suffix}.csv")
        else:
            raise ValueError(
                "Either csv_file must be provided, or one of the following:\n"
                "  - For Davies corpora: corpus_path and dir_suffix\n"
                "  - For ngrams: ngram_size, repo_release_id, repo_corpus_id, db_path_stub, and dir_suffix"
            )

    # Validate required parameters
    if outcome is None:
        raise ValueError("outcome parameter is required")
    if predictors is None:
        raise ValueError("predictors parameter is required")
    # Validate model_type parameter
    valid_model_types = ['auto', 'mixed', 'ols']
    if model_type not in valid_model_types:
        raise ValueError(f"model_type must be one of {valid_model_types}, got: {model_type}")

    # Force OLS if random_effects is None
    if random_effects is None:
        model_type = 'ols'

    # Validate inputs
    is_valid, error_msg = _validate_inputs(csv_file, outcome, predictors, interactions)
    if not is_valid:
        raise ValueError(error_msg)

    # Load and prepare data
    if verbose:
        print("Loading data...")
    df = _prepare_data(csv_file, outcome, predictors)

    if verbose:
        print(f"Loaded {len(df)} observations")
        print(f"Number of years: {df['year'].nunique()}")
        print("")

    # Build formula
    formula = _build_formula(outcome, predictors, interactions, standardize)

    if verbose:
        print("Model specification:")
        print(f"  Formula: {formula}")
        print(f"  Random effects: {random_effects}")
        print("")

    # Fit model based on model_type parameter
    if model_type == 'ols':
        # Force OLS regression
        if verbose:
            print("Fitting OLS regression model...")
        model = ols(formula, df)
        results = model.fit()
    elif model_type == 'mixed':
        # Force mixed-effects (will error if singular)
        if verbose:
            print("Fitting mixed-effects model...")
        groups_array = np.asarray(df[random_effects]).flatten()
        groups = groups_array.tolist()
        model = mixedlm(formula, df, groups=groups)
        results = model.fit(method='lbfgs')
    else:  # model_type == 'auto'
        # Try mixed-effects first, fall back to OLS if singular
        if verbose:
            print("Fitting mixed-effects model...")
        try:
            groups_array = np.asarray(df[random_effects]).flatten()
            groups = groups_array.tolist()
            model = mixedlm(formula, df, groups=groups)
            results = model.fit(method='lbfgs')
        except (np.linalg.LinAlgError, Exception) as mixed_error:
            # Fall back to OLS if mixed-effects fails
            if verbose:
                print(f"Mixed-effects model failed ({mixed_error}). Falling back to OLS regression...")
            model = ols(formula, df)
            results = model.fit()

    if verbose:
        print("Model converged successfully!")
        print("")
        print("=" * 80)
        print(results.summary())
        print("=" * 80)
        print("")

    # Save results to file if requested
    if output_file:
        model_type = "MIXED-EFFECTS" if isinstance(results, MixedLMResults) else "OLS"
        with open(output_file, 'w') as f:
            f.write(f"{model_type} REGRESSION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Outcome: {outcome}\n")
            f.write(f"Predictors: {', '.join(predictors)}\n")
            if interactions:
                f.write(f"Interactions: {', '.join([f'{v1}×{v2}' for v1, v2 in interactions])}\n")
            if isinstance(results, MixedLMResults):
                f.write(f"Random effects: {random_effects}\n")
                f.write(f"N groups ({random_effects}): {df[random_effects].nunique()}\n")
            f.write(f"N observations: {len(df)}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            f.write(str(results.summary()))
            f.write("\n\n")

        if verbose:
            print(f"Results saved to: {output_file}")
            print("")

    return results


def plot_regression_results(
    results: Union[MixedLMResults, RegressionResultsWrapper],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    alpha: float = 0.05,
    plot_random_effects: bool = True
) -> None:
    """
    Visualize regression results with coefficient plots.

    Creates two subplots:
    1. Fixed effects coefficients with confidence intervals
    2. Random effects (random intercepts by group) if applicable

    Args:
        results: MixedLMResults object from run_regression_analysis
        output_file: Optional path to save figure
        figsize: Figure size as (width, height)
        alpha: Significance level for confidence intervals (default: 0.05)
        plot_random_effects: Whether to plot random effects (default: True)

    Example:
        >>> from train.word2vec import run_regression_analysis, plot_regression_results
        >>>
        >>> results = run_regression_analysis(
        ...     csv_file='evaluation_results_test.csv',
        ...     outcome='similarity_score',
        ...     predictors=['year', 'vector_size', 'epochs']
        ... )
        >>>
        >>> plot_regression_results(
        ...     results,
        ...     output_file='regression_coefficients.png'
        ... )
    """
    # Set modern seaborn theme to match evaluation plots
    sns.set_theme(
        style="ticks",
        context="notebook",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
        }
    )

    # Colorblind-friendly palette (IBM Design Color Palette)
    colorblind_palette = [
        '#648FFF',  # Blue
        '#785EF0',  # Purple
        '#DC267F',  # Magenta
        '#FE6100',  # Orange
        '#FFB000',  # Yellow
    ]

    # Determine number of subplots - only create 2 if we have random effects
    has_random_effects = isinstance(results, MixedLMResults) and hasattr(results, 'random_effects')
    n_subplots = 2 if (plot_random_effects and has_random_effects) else 1

    # Adjust figsize to keep FE plot dimensions consistent
    if n_subplots == 1:
        # Use half the width for single plot to match the FE subplot size when n_subplots=2
        adjusted_figsize = (figsize[0] / 2, figsize[1])
    else:
        adjusted_figsize = figsize

    fig, axes = plt.subplots(1, n_subplots, figsize=adjusted_figsize, dpi=100)
    if n_subplots == 1:
        axes = [axes]

    # Extract coefficients (OLS uses 'params', mixed models use 'fe_params')
    fe_params = results.fe_params if isinstance(results, MixedLMResults) else results.params
    fe_conf = results.conf_int(alpha=alpha)

    # Remove intercept for cleaner visualization
    if 'Intercept' in fe_params.index:
        fe_params = fe_params.drop('Intercept')
    if 'Intercept' in fe_conf.index:
        fe_conf = fe_conf.drop('Intercept')

    # Clean up parameter names for better readability
    # Remove 'scale()' and 'C()' wrappers
    def clean_name(name):
        cleaned = re.sub(r'scale\((.*?)\)', r'\1', name)
        cleaned = re.sub(r'C\((.*?)\)\[(.*?)\]', r'\1: \2', cleaned)
        return cleaned

    fe_params.index = [clean_name(name) for name in fe_params.index]
    fe_conf.index = [clean_name(name) for name in fe_conf.index]

    # Sort by coefficient magnitude
    sort_idx = fe_params.abs().sort_values(ascending=True).index
    fe_params = fe_params[sort_idx]
    fe_conf = fe_conf.loc[sort_idx]

    # Plot 1: Fixed effects
    ax = axes[0]
    y_pos = np.arange(len(fe_params))

    # Plot coefficients with colorblind-friendly color
    ax.scatter(fe_params.values, y_pos, s=120, color=colorblind_palette[0],
               zorder=3, alpha=0.8, edgecolors='white', linewidth=1)

    # Plot confidence intervals
    for i, param_name in enumerate(fe_params.index):
        lower = fe_conf.loc[param_name, 0]
        upper = fe_conf.loc[param_name, 1]
        ax.plot([lower, upper], [i, i], color='#333333', linewidth=2.5, zorder=2, alpha=0.7)

    # Add vertical line at zero
    ax.axvline(x=0, color='#DC267F', linestyle='--', alpha=0.6, linewidth=1.5)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fe_params.index)
    ax.set_xlabel('Coefficient Estimate', fontsize=13, labelpad=10)
    ax.set_title('Fixed Effects Coefficients\n(with {}% CI)'.format(int((1-alpha)*100)),
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.8)
    sns.despine(ax=ax)

    # Plot 2: Random effects (if applicable)
    if has_random_effects:
        ax = axes[1]

        # Extract random effects
        rand_effects = results.random_effects
        re_groups = sorted(rand_effects.keys())
        re_values = [rand_effects[group]['Group'] for group in re_groups]

        # Plot random intercepts with colorblind-friendly color
        y_pos = np.arange(len(re_groups))
        ax.scatter(re_values, y_pos, s=80, color=colorblind_palette[1],
                   alpha=0.7, edgecolors='white', linewidth=0.5)

        # Add vertical line at zero
        ax.axvline(x=0, color='#DC267F', linestyle='--', alpha=0.6, linewidth=1.5)

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(g) for g in re_groups])
        ax.set_xlabel('Random Intercept', fontsize=13, labelpad=10)
        ax.set_ylabel('Year', fontsize=13, labelpad=10)
        ax.set_title('Random Effects by Year', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.8)
        sns.despine(ax=ax)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_file}")

    plt.show()


def get_model_summary(
    results: Union[MixedLMResults, RegressionResultsWrapper],
    predictors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract a summary table of regression coefficients.

    Args:
        results: MixedLMResults or RegressionResultsWrapper object from run_regression_analysis
        predictors: Optional list of specific predictors to include

    Returns:
        DataFrame with coefficient, std error, t-value, p-value, and CI
    """
    # Extract summary (OLS and mixed models have different attribute names)
    is_mixed = isinstance(results, MixedLMResults)
    summary_df = pd.DataFrame({
        'Coefficient': results.fe_params if is_mixed else results.params,
        'Std Error': results.bse_fe if is_mixed else results.bse,
        't-value': results.tvalues,
        'p-value': results.pvalues,
        'CI Lower': results.conf_int()[0],
        'CI Upper': results.conf_int()[1]
    })

    # Add significance stars
    def add_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    summary_df['Sig'] = summary_df['p-value'].apply(add_stars)

    # Filter predictors if specified
    if predictors:
        # Create filter pattern
        pattern = '|'.join(predictors)
        mask = summary_df.index.str.contains(pattern, case=False, regex=True)
        summary_df = summary_df[mask]

    return summary_df
