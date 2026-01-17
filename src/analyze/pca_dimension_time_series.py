"""
Time series analysis of semantic dimensions via PCA-based contrast pairs.

This module provides functions to compute semantic dimensions across years
using the PCA contrast pair method, analogous to WEAT but more interpretable.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from ngramprep.common.w2v_model import W2VModel


def compute_pca_dimension_over_years(
    model_dir,
    token_contrasts,
    reference_year=None,
    ensure_sign_positive=None,
    years=None,
    verbose=True,
    n_components_diagnostic=None
):
    """
    Compute semantic dimension projections across years using PCA on contrast pairs.

    This function:
    1. Fits PCA dimension on a reference year's model (or the earliest available)
    2. Projects all words onto this fixed dimension for each year
    3. Returns vocabulary projections as time series
    4. Optionally computes PCA diagnostics (scree plot, variance breakdown)

    Args:
        model_dir (str or Path): Directory containing yearly word2vec models (*.kv files).
        token_contrasts (list of tuples): List of (token1, token2) pairs defining the semantic dimension.
            Example: [('he', 'she'), ('him', 'her'), ('man', 'woman'), ...]
        reference_year (int, optional): Year to fit PCA dimension on. If None, uses earliest available year.
        ensure_sign_positive (bool, list, or None): Controls sign orientation of PC1.
            - If True: infer positive tokens from pair order (second element of each pair)
            - If list of str: use specified tokens as positive pole
            - If None/False: no sign consistency enforcement
            Example: True ensures second element of each pair projects positively.
        years (list of int, optional): Specific years to compute (must exist as .kv files). 
            If None, automatically finds all years with models.
        verbose (bool): If True, prints progress updates.
        n_components_diagnostic (int, optional): If provided, fit this many components for diagnostics
            (scree plot, variance breakdown). Creates 'pca_diagnostics' in returned dict.

    Returns:
        dict: Dictionary with structure:
            {
                'dimension': np.ndarray,  # Reference year's PC1 (shape: vector_size)
                'variance_explained': float,  # Proportion of variance explained by PC1
                'reference_year': int,  # Year used for dimension fitting
                'projections': pd.DataFrame,  # Shape: (n_vocab, n_years)
                    columns are years, rows are words, values are projections
                'yearly_stats': dict,  # {year: {'mean': float, 'std': float, 'n_words': int}}
                'pca_diagnostics': dict  # (optional) PCA diagnostic info if n_components_diagnostic provided
            }

    Example:
        >>> contrasts = [('he', 'she'), ('him', 'her'), ('man', 'woman')]
        >>> result = compute_pca_dimension_over_years(
        ...     model_dir='/path/to/models',
        ...     token_contrasts=contrasts,
        ...     ensure_sign_positive=['she', 'her', 'woman'],
        ...     years=[1950, 1960, 1970],
        ...     reference_year=1950,
        ...     n_components_diagnostic=10  # Enable diagnostics
        ... )
        >>> gender_projections = result['projections']
        >>> print(gender_projections.loc['woman'])  # Woman's projection across years
        >>> 
        >>> # View diagnostics
        >>> W2VModel.print_pca_variance_summary(result['pca_diagnostics'])
        >>> fig = W2VModel.plot_pca_diagnostics(result['pca_diagnostics'])
    """
    model_dir = Path(model_dir)
    
    # Find available models
    model_files = sorted(model_dir.glob("*.kv"))
    if not model_files:
        raise FileNotFoundError(f"No .kv model files found in {model_dir}")
    
    # Extract years from filenames and maintain mapping to filenames
    # (e.g., "w2v_y1950_*.kv" -> 1950 or "model_1950.kv" -> 1950)
    available_years = []
    year_to_filename = {}  # Map year -> actual filename
    
    for f in model_files:
        try:
            stem = f.stem
            # Try format: w2v_y<YEAR>_... (look for 'y' followed by digits)
            parts = stem.split("_")
            year = None
            for part in parts:
                if part.startswith('y') and len(part) > 1:
                    try:
                        year = int(part[1:])  # Extract digits after 'y'
                        break
                    except ValueError:
                        pass
            # Fallback: try extracting last part (for "model_1950" format)
            if year is None:
                year = int(parts[-1])
            available_years.append(year)
            year_to_filename[year] = f  # Store actual filename
        except (ValueError, IndexError):
            if verbose:
                print(f"⚠️ Skipping file with non-standard name: {f.name}")
    
    available_years = sorted(available_years)
    
    # Determine reference year for fitting PCA
    if reference_year is None:
        reference_year = available_years[0]
    elif reference_year not in available_years:
        raise ValueError(
            f"Reference year {reference_year} not found. Available: {available_years}"
        )
    
    # Determine which years to analyze
    if years is None:
        years_to_analyze = available_years
    else:
        years_to_analyze = sorted(set(years) & set(available_years))
        if not years_to_analyze:
            raise ValueError(f"No requested years found in models. Available: {available_years}")
    
    if verbose:
        print(f"📊 PCA Dimension Analysis: {len(years_to_analyze)} years "
              f"[{min(years_to_analyze)}-{max(years_to_analyze)}]")
        print(f"   Reference year for PCA: {reference_year}")
        print(f"   Contrast pairs: {len(token_contrasts)}")
    
    # Load reference model and compute dimension
    if verbose:
        print(f"   Loading reference model ({reference_year})...")
    
    ref_model_path = year_to_filename[reference_year]
    ref_model = W2VModel(ref_model_path)
    
    # Compute PCA dimension on reference year
    pca_result = ref_model.compute_pca_dimension(
        token_contrasts=token_contrasts,
        ensure_sign_positive=ensure_sign_positive,
        n_components_diagnostic=n_components_diagnostic
    )
    
    dimension = pca_result['dimension']
    variance_explained = pca_result['variance_explained']
    component_loadings = pca_result['component_loadings']
    
    if verbose:
        print(f"   PC1 variance explained: {variance_explained:.2%}")
        print(f"   Component loadings:")
        for pair, loading in sorted(component_loadings.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"     {pair}: {loading:+.3f}")
    
    # Get reference vocabulary
    ref_vocab = set(ref_model.vocab.keys())
    
    if verbose:
        print(f"   Reference vocabulary: {len(ref_vocab)} words")
    
    # Project all years onto the reference dimension
    projections_data = {}
    yearly_stats = {}
    
    for year in years_to_analyze:
        if verbose:
            print(f"   Projecting year {year}...", end=" ")
        
        model_path = year_to_filename[year]
        model = W2VModel(model_path)
        
        # Project each word in reference vocabulary
        year_projections = {}
        for word in ref_vocab:
            if word in model.vocab:
                try:
                    proj = model.project_onto_dimension(word, dimension)
                    year_projections[word] = proj
                except ValueError:
                    # Word not in this year's vocabulary
                    pass
        
        projections_data[year] = year_projections
        
        # Compute statistics
        proj_values = list(year_projections.values())
        yearly_stats[year] = {
            'mean': np.mean(proj_values),
            'std': np.std(proj_values),
            'n_words': len(proj_values)
        }
        
        if verbose:
            print(f"✓ {len(year_projections)} words projected "
                  f"(mean={yearly_stats[year]['mean']:.3f}, std={yearly_stats[year]['std']:.3f})")
    
    # Create DataFrame: words × years
    projections_df = pd.DataFrame(projections_data)
    
    # Reorder columns to be chronological
    projections_df = projections_df[[y for y in years_to_analyze if y in projections_df.columns]]
    
    result = {
        'dimension': dimension,
        'variance_explained': variance_explained,
        'reference_year': reference_year,
        'projections': projections_df,
        'yearly_stats': yearly_stats,
        'component_loadings': component_loadings
    }
    
    # Add diagnostics if requested
    if n_components_diagnostic:
        result['pca_diagnostics'] = {
            'all_variance_explained': pca_result['all_variance_explained'],
            'pca_object': pca_result['pca_object'],
            'n_contrast_pairs': len(token_contrasts)
        }
    
    if verbose:
        print(f"✅ Analysis complete: {len(projections_df)} words × {len(years_to_analyze)} years")
    
    return result


def get_dimension_trajectory(pca_result, token, smoothing=None):
    """
    Extract the projection trajectory of a single word across years.

    Args:
        pca_result (dict): Output from compute_pca_dimension_over_years().
        token (str): The word to extract trajectory for.
        smoothing (int, optional): Window size for rolling mean smoothing. If None, no smoothing.

    Returns:
        pd.Series: Projections of token across years (index=year, values=projection).
                   NaN for years where word wasn't in model vocabulary.

    Raises:
        KeyError: If token not found in projections.
    """
    if token not in pca_result['projections'].index:
        raise KeyError(f"Token '{token}' not in computed projections.")
    
    trajectory = pca_result['projections'].loc[token]
    
    if smoothing and smoothing > 1:
        trajectory = trajectory.rolling(window=smoothing, center=True, min_periods=1).mean()
    
    return trajectory


def get_dimension_changes(pca_result, n_top=10):
    """
    Get words with largest positive and negative changes across the time series.

    Args:
        pca_result (dict): Output from compute_pca_dimension_over_years().
        n_top (int): Number of top words in each direction.

    Returns:
        dict: {
            'positive_change': [(word, change), ...],  # Words that moved most positive
            'negative_change': [(word, change), ...],  # Words that moved most negative
            'first_year': int,
            'last_year': int
        }
    """
    projections = pca_result['projections']
    years = projections.columns.tolist()
    
    first_year = min(years)
    last_year = max(years)
    
    # Compute changes
    first_proj = projections[first_year]
    last_proj = projections[last_year]
    
    changes = last_proj - first_proj
    changes = changes.dropna()
    
    # Sort by magnitude
    positive_change = changes.nlargest(n_top)
    negative_change = changes.nsmallest(n_top)
    
    return {
        'positive_change': list(zip(positive_change.index, positive_change.values)),
        'negative_change': list(zip(negative_change.index, negative_change.values)),
        'first_year': first_year,
        'last_year': last_year
    }
