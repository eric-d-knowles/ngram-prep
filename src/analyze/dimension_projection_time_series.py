"""
Time series projection of words onto semantic dimensions (PCA or mean-diff).

Analogous to WEAT-over-years, this utility:
1) For each year, fits a semantic dimension using either PCA or mean-diff
2) Projects specified words onto that year's own dimension
3) Optionally plots trajectories for quick visual inspection

NOTE: Each year uses its own fitted dimension (not a reference year dimension),
allowing the dimension itself to drift over time as the model and language evolve.

FEMVAR: this module also emits per-year, per-canonical estimation-variance
proxies (fem_var) alongside the projections, computed from the yearly
models' own token counts at the projection-averaging site — so the
components counted are by construction the components averaged, including
per-year availability. Under the wbnone training scheme the .kv counts are
TYPE counts = actual training exposures, exactly the quantity the
1/n estimation-variance bound concerns; the models' training used gensim's
default sample=1e-3, under which every roster lexeme sits below the
subsampling threshold, so subsample_t=None is exact (certified July 2026).
Edit sites are marked ``FEMVAR: edit N`` in the style of the SYNONYMS edits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import re

import numpy as np
import pandas as pd
from scipy.stats import linregress          # used by compute_baseline_set
import matplotlib.pyplot as plt             # used by compute_baseline_set (plot_baseline)

from ngramprep.common.w2v_model import W2VModel


# FEMVAR: edit 1 — count/total accessors with layered fallbacks over
# W2VModel's possible internals. Counts unavailable -> NaN (detected
# and warned about after the loop, never silently zero).
def _token_count(model, token) -> float:
    """Vocabulary count for token from the model's KeyedVectors, or NaN."""
    candidates = [model] + [getattr(model, attr, None)
                            for attr in ("kv", "wv", "model", "keyed_vectors")]
    for obj in candidates:
        if obj is None:
            continue
        get_vecattr = getattr(obj, "get_vecattr", None)
        if get_vecattr is not None:
            try:
                return float(get_vecattr(token, "count"))
            except Exception:  # noqa: BLE001
                continue
    return np.nan


def _year_total_tokens(model) -> float:
    """Sum of vocabulary counts (proxy for corpus tokens; used only for
    the subsampling adjustment's relative-frequency term)."""
    candidates = [getattr(model, attr, None)
                  for attr in ("kv", "wv", "model", "keyed_vectors")] + [model]
    for obj in candidates:
        if obj is None:
            continue
        expandos = getattr(obj, "expandos", None)
        if isinstance(expandos, dict) and "count" in expandos:
            try:
                return float(np.sum(expandos["count"]))
            except Exception:  # noqa: BLE001
                pass
        index_to_key = getattr(obj, "index_to_key", None)
        get_vecattr = getattr(obj, "get_vecattr", None)
        if index_to_key is not None and get_vecattr is not None:
            try:
                return float(sum(get_vecattr(w, "count") for w in index_to_key))
            except Exception:  # noqa: BLE001
                continue
    return np.nan


# FEMVAR: edit 2 — the variance arithmetic, factored pure for testability.
def _compute_fem_var_frame(
    projections_full: pd.DataFrame,
    counts_full: pd.DataFrame,
    synonyms: Optional[Dict[str, List[str]]],
    output_columns: Sequence[str],
) -> pd.DataFrame:
    """Per-year estimation-variance proxy for each output (canonical) column.

    fem_var(unit, year) ∝ (1/k²)·Σ_i 1/n_eff(component_i, year), where the
    sum runs over exactly the components whose projection is non-NaN that
    year (the same set the synonym mean averaged — availability is read
    off the pre-drop projection frame, so estimator and variance cannot
    drift apart). Strictness rule: if any component USED in the mean has
    a missing count, fem_var is NaN for that unit-year (never a silent
    partial sum).
    """
    inv = 1.0 / counts_full
    out = {}
    syn = synonyms or {}
    for col in output_columns:
        comp = [t for t in syn.get(col, [col]) if t in projections_full.columns]
        if not comp:
            out[col] = pd.Series(np.nan, index=projections_full.index)
            continue
        avail = projections_full[comp].notna()
        k = avail.sum(axis=1)
        masked = inv[comp].where(avail)
        n_counted = masked.notna().sum(axis=1)
        ssum = masked.sum(axis=1)
        fv = ssum / (k.astype(float) ** 2)
        fv[(k == 0) | (n_counted != k)] = np.nan
        out[col] = fv
    return pd.DataFrame(out, index=projections_full.index)[list(output_columns)]


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
    verbose: bool = True,
    baseline_source: Optional[Union[pd.Series, Sequence[str]]] = None,
    baseline_agg: str = "median",
    synonyms: Optional[Dict[str, List[str]]] = None,  # SYNONYMS: edit 1 — must precede **method_kwargs
    emit_fem_var: bool = True,        # FEMVAR: edit 3 — must precede **method_kwargs
    subsample_t: Optional[float] = None,  # FEMVAR: edit 3 — training `sample`; NOT recoverable from .kv
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
        verbose: Print progress information.
        baseline_source: Optional baseline input used for correction. Accepted forms:
            - pd.Series indexed by year: interpreted as a pre-computed yearly baseline.
            - Sequence[str]: treated as baseline words and projected independently for
              each year using that year's fitted dimension, then aggregated via
              baseline_agg.
        baseline_agg: Aggregation method for word-list baseline_source ('mean' or
            'median'). Ignored when baseline_source is a pd.Series. Default: 'median'.
        synonyms: Optional mapping of canonical label -> list of synonym tokens whose
            projections are averaged each year to produce the canonical column.
            Example: ``{'surgeon': ['surgeon', 'physician', 'doctor']}``.
            Synonym tokens not equal to the canonical key are projected internally
            but removed from the output DataFrame; the canonical column receives
            the per-year mean (ignoring NaN) of all synonym projections.
        emit_fem_var: If True (default), also return per-year estimation-variance
            proxies computed from each yearly model's own token counts:
            fem_var ∝ (1/k²)·Σ 1/n_eff over exactly the components averaged
            into each output column that year. Adds negligible cost.
        subsample_t: Word2Vec subsampling threshold from the TRAINING config
            (the ``sample`` parameter; KeyedVectors .kv files do not store it).
            When given, counts are converted to effective post-subsampling
            exposures n_eff = count · min(1, (sqrt(f/t)+1)·t/f) with f the
            within-model relative frequency. None (default) uses raw counts;
            the between-unit ordering — the quantity the TIpred carries —
            is robust to this choice.
        **method_kwargs: Extra kwargs forwarded to dimension computation methods.
 
    Returns:
        dict with keys:
            'dimension' (None): Set to None since year-specific dimensions are used.
            'reference_year' (int): Original reference_year parameter (for backwards compatibility).
            'method' (str): 'pca' or 'meandiff'.
            'projections' (pd.DataFrame): Index=years, columns=test_words. Projections onto each year's own dimension.
            'fem_var' (pd.DataFrame): Same shape/columns as 'projections' —
                per-year estimation-variance proxies (empty when
                emit_fem_var=False or counts unavailable in the models).
            'token_counts' (pd.DataFrame): Raw per-year counts for ALL projected
                tokens (pre-synonym resolution; audit artifact).
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
 
    # SYNONYMS: edit 2 — collect all tokens to project: test_words plus any
    # extra synonym tokens not already in test_words
    _synonym_extras: set = set()
    if synonyms:
        for tokens in synonyms.values():
            _synonym_extras.update(tokens)
    _synonym_extras -= set(test_words)  # only the ones not already in test_words
    _all_words = list(test_words) + sorted(_synonym_extras)
 
    projections_data: Dict[int, Dict[str, float]] = {}
    counts_data: Dict[int, Dict[str, float]] = {}   # FEMVAR: edit 4
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
 
            # FEMVAR: edit 5 — per-model total (only needed for subsampling)
            _year_total = (_year_total_tokens(model)
                           if (emit_fem_var and subsample_t) else np.nan)
 
            # Project words onto THIS year's dimension.
            # Space-separated bigrams (e.g. "marketing manager") are stored in
            # W2V models as hyphen-joined tokens ("marketing-manager"), so
            # fall back to the hyphenated form when the raw label isn't found.
            row = {}
            count_row = {}   # FEMVAR: edit 5
            for word in _all_words:  # SYNONYMS: edit 3 — was test_words
                lookup = word if word in model.vocab else word.replace(" ", "-")
                if lookup in model.vocab:
                    try:
                        row[word] = model.project_onto_dimension(lookup, dimension)
                    except ValueError:
                        row[word] = np.nan
                    # FEMVAR: edit 5 — same lookup token as the projection,
                    # with the subsampling adjustment when configured
                    if emit_fem_var:
                        c = _token_count(model, lookup)
                        if (subsample_t and np.isfinite(c) and c > 0
                                and np.isfinite(_year_total) and _year_total > 0):
                            f = c / _year_total
                            p_keep = min(1.0, (np.sqrt(f / subsample_t) + 1.0)
                                         * (subsample_t / f))
                            c = c * p_keep
                        count_row[word] = c if (np.isfinite(c) and c >= 1.0) else np.nan
                    else:
                        count_row[word] = np.nan
                else:
                    row[word] = np.nan
                    count_row[word] = np.nan   # FEMVAR: edit 5
            projections_data[year] = row
            counts_data[year] = count_row      # FEMVAR: edit 5
            if verbose:
                valid = sum(1 for v in row.values() if not pd.isna(v))
                print(f"✓ {valid}/{len(_all_words)} words")  # SYNONYMS: edit 3 — was len(test_words)
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
            "fem_var": pd.DataFrame(),        # FEMVAR: edit 7
            "token_counts": pd.DataFrame(),   # FEMVAR: edit 7
            "component_loadings": {},
            "missing_years": missing_years,
            "error_years": error_years,
            "yearly_dimensions": yearly_dimensions,
        }
 
    projections_df = pd.DataFrame(projections_data).T
    projections_df.index = projections_df.index.astype(int)
    projections_df = projections_df.sort_index()
 
    # FEMVAR: edit 6 — counts frame parallel to the PRE-DROP projections,
    # then the variance arithmetic on the same availability mask the
    # synonym mean uses. Computed BEFORE the synonym drop so component
    # NaN patterns are still visible.
    counts_df = pd.DataFrame(counts_data).T
    counts_df.index = counts_df.index.astype(int)
    counts_df = counts_df.sort_index().reindex(columns=projections_df.columns)
    fem_var_df = pd.DataFrame()
    if emit_fem_var:
        if counts_df.notna().values.any():
            _final_cols = ([w for w in test_words]
                           if synonyms else list(projections_df.columns))
            fem_var_df = _compute_fem_var_frame(
                projections_df, counts_df, synonyms, _final_cols)
        else:
            print("⚠️ FEMVAR: token counts unavailable in these .kv models "
                  "(get_vecattr 'count' returned nothing) — fem_var will be "
                  "EMPTY. Verify the models were saved with vocabulary "
                  "counts before relying on the Fem-side injection.")
 
    # SYNONYMS: edit 4 — average synonym groups and drop non-canonical
    # tokens from the output
    if synonyms:
        _to_drop = set()
        for canonical, tokens in synonyms.items():
            cols = [t for t in tokens if t in projections_df.columns]
            if cols:
                projections_df[canonical] = projections_df[cols].mean(axis=1)
                _to_drop.update(t for t in cols if t != canonical)
        projections_df = projections_df.drop(columns=sorted(_to_drop), errors='ignore')
        # Restore original column order (test_words, preserving any canonicals)
        ordered = [w for w in test_words if w in projections_df.columns]
        projections_df = projections_df[ordered]
 
    # FEMVAR: edit 6b — align fem_var to the final output exactly: same
    # columns, and NaN wherever the released projection is NaN.
    if emit_fem_var and not fem_var_df.empty:
        fem_var_df = fem_var_df.reindex(columns=projections_df.columns)
        fem_var_df = fem_var_df.where(projections_df.notna())
 
    # Optionally apply baseline correction
    projections_corrected_df: Optional[pd.DataFrame] = None
    baseline_applied = False
    aligned_baseline = pd.Series(dtype=float)
 
    # Option 1: Use pre-computed yearly baseline series
    if isinstance(baseline_source, pd.Series):
        if not projections_df.empty:
            # Align baseline to available years
            aligned_baseline = baseline_source.reindex(projections_df.index)
            aligned_baseline = aligned_baseline.fillna(0.0)  # leave raw values where baseline is unavailable
            projections_corrected_df = projections_df.sub(aligned_baseline, axis=0)
            baseline_applied = True
 
    # Option 2: Compute baseline from baseline word list
    elif baseline_source is not None:
        if baseline_agg not in ["mean", "median"]:
            raise ValueError("baseline_agg must be 'mean' or 'median'.")
 
        if isinstance(baseline_source, (str, bytes)):
            raise ValueError("baseline_source must be a pd.Series or a sequence of words, not a string.")
 
        try:
            baseline_words_unique = list(dict.fromkeys(baseline_source))
        except TypeError as exc:
            raise ValueError("baseline_source must be a pd.Series or an iterable of words.") from exc
 
        if not baseline_words_unique:
            raise ValueError("baseline_source word list is empty.")
 
        word_found_any = {word: False for word in baseline_words_unique}
        baseline_by_year: Dict[int, float] = {}
 
        for year in projections_df.index:
            model_path = year_to_path.get(year)
            dimension = yearly_dimensions.get(year)
 
            if model_path is None or dimension is None:
                baseline_by_year[year] = np.nan
                continue
 
            try:
                year_model = W2VModel(str(model_path))
                year_values = []
                for word in baseline_words_unique:
                    if word in year_model.vocab:
                        try:
                            value = year_model.project_onto_dimension(word, dimension)
                            year_values.append(value)
                            word_found_any[word] = True
                        except ValueError:
                            continue
 
                if year_values:
                    if baseline_agg == "median":
                        baseline_by_year[year] = float(np.median(year_values))
                    else:  # mean
                        baseline_by_year[year] = float(np.mean(year_values))
                else:
                    baseline_by_year[year] = np.nan
            except Exception as exc:  # noqa: BLE001
                baseline_by_year[year] = np.nan
                if verbose:
                    print(f"⚠️ Could not compute baseline for year {year}: {exc}")
 
        available_baseline_words = [word for word, found in word_found_any.items() if found]
 
        if not available_baseline_words:
            if verbose:
                print(f"⚠️ None of the {len(baseline_words_unique)} baseline words were available across yearly models.")
        else:
            if verbose and len(available_baseline_words) < len(baseline_words_unique):
                missing = set(baseline_words_unique) - set(available_baseline_words)
                print(f"⚠️ {len(missing)} baseline words not found in any year: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
 
            aligned_baseline = pd.Series(baseline_by_year, dtype=float).reindex(projections_df.index)
 
            if aligned_baseline.notna().any():
                aligned_baseline = aligned_baseline.fillna(0.0)
                projections_corrected_df = projections_df.sub(aligned_baseline, axis=0)
                baseline_applied = True
 
                if verbose:
                    print(f"✓ Baseline computed from {len(available_baseline_words)} words using {baseline_agg}")
            elif verbose:
                print("⚠️ Baseline values are NaN for all years; baseline correction not applied.")
 
    return {
        "dimension": None,  # Year-specific: no single reference dimension
        "reference_year": reference_year,
        "method": method,
        "projections": projections_df,
        "fem_var": fem_var_df,          # FEMVAR: edit 7
        "token_counts": counts_df,      # FEMVAR: edit 7 — pre-synonym audit frame
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
        return corr_df.where(mask_local).stack(dropna=True).values

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