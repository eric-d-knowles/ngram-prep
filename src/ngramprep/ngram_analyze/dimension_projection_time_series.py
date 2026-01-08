"""
Time series projection of words onto a semantic dimension (PCA or mean-diff).

Analogous to WEAT-over-years, this utility:
1) Fits a semantic dimension on a reference year using either PCA or mean-diff
2) Projects specified words onto that fixed dimension for each year
3) Optionally plots trajectories for quick visual inspection
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

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
    verbose: bool = True,
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
        verbose: Print progress information.
        **method_kwargs: Extra kwargs forwarded to dimension computation methods.

    Returns:
        dict with keys:
            'dimension' (np.ndarray): The fitted dimension vector.
            'reference_year' (int): Year used to fit the dimension.
            'method' (str): 'pca' or 'meandiff'.
            'projections' (pd.DataFrame): Index=years, columns=test_words.
            'component_loadings' (dict): Pair loadings on the dimension.
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
        print(f"   Reference year: {reference_year}")
        print(f"   Method: {method}")
        print(f"   Contrast pairs: {len(token_contrasts)}")

    # Fit dimension on reference year
    ref_model = W2VModel(str(year_to_path[reference_year]))
    if method.lower() == "pca":
        dimension_result = ref_model.compute_pca_dimension(
            token_contrasts=token_contrasts,
            ensure_sign_positive=ensure_sign_positive,
            **method_kwargs,
        )
    elif method.lower() == "meandiff":
        dimension_result = ref_model.compute_meandiff_dimension(
            token_contrasts=token_contrasts,
            **method_kwargs,
        )
    else:
        raise ValueError("method must be 'pca' or 'meandiff'")

    dimension = dimension_result["dimension"]
    component_loadings = dimension_result.get("component_loadings", {})

    projections_data: Dict[int, Dict[str, float]] = {}
    error_years: Dict[int, str] = {}

    for year in years_to_analyze:
        model_path = year_to_path.get(year)

        if verbose:
            print(f"   Projecting {year}...", end=" ")

        try:
            model = W2VModel(str(model_path))
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
            "dimension": dimension,
            "reference_year": reference_year,
            "method": method,
            "projections": pd.DataFrame(),
            "component_loadings": component_loadings,
            "missing_years": missing_years,
            "error_years": error_years,
        }

    projections_df = pd.DataFrame.from_dict(projections_data, orient="index")
    projections_df = projections_df.sort_index()

    if plot:
        plt.figure(figsize=(10, 5))
        for word in test_words:
            if word not in projections_df.columns:
                continue
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

            plt.plot(series_to_plot.index, series_to_plot.values, marker="o", linestyle="-", label=word)

        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Projection (cosine)", fontsize=12)
        plt.title(f"Word projections on {method} dimension", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "dimension": dimension,
        "reference_year": reference_year,
        "method": method,
        "projections": projections_df,
        "component_loadings": component_loadings,
        "missing_years": missing_years,
        "error_years": error_years,
    }
