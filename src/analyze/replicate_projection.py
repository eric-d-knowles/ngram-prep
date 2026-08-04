"""
replicate_projection.py — per-replicate projections for the Fem-side
observation-noise program.

WHY THIS MODULE EXISTS
----------------------
The CT-SEM needs observation-error variance on the *projection* scale (the
scalar gender-axis score), not on the embedding-vector scale. This module
runs the production projection calculation inside each noise-ensemble
replicate and returns the resulting (year x replicate) x occupation cube, from
which the injectable noise columns are collapsed.

Two invariants the design turns on:

1. THE AXIS IS FIT INSIDE EACH REPLICATE. Procrustes alignment is orthogonal
   and orthogonal maps preserve inner products, so a within-replicate axis
   makes alignment a no-op for the projection and drops the alignment-residual
   component entirely.

2. THE PRODUCTION PROJECTION AND THE NOISE ESTIMATE SHARE ONE CODE PATH.
   ``project_one_model`` below is the single per-model body; both
   ``compute_projection_over_years`` (patched, see PATCH note at the bottom)
   and ``compute_projection_over_replicates`` call it. The estimator and its
   variance cannot drift apart.

Synonym groups are averaged WITHIN a replicate before any SD is taken, so the
noise reduction that averaging buys is reflected in the estimate.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ngramprep.common.w2v_model import W2VModel


# --------------------------------------------------------------------------
# 1. Shared per-model core
# --------------------------------------------------------------------------

def project_one_model(
    model: "W2VModel",
    words: Sequence[str],
    token_contrasts: Sequence[tuple],
    method: str = "meandiff",
    ensure_sign_positive: Optional[Union[bool, List[str]]] = True,
    **method_kwargs,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Optional[str]]]:
    """Fit this model's own dimension and project ``words`` onto it.

    This is the body formerly inlined in ``compute_projection_over_years``'s
    year loop, extracted verbatim so the year path and the replicate path
    cannot diverge.

    Returns
    -------
    row : dict
        word -> projection (np.nan when the token is out of vocabulary or
        the projection raises).
    dimension : np.ndarray
        The fitted semantic dimension for this model.
    lookups : dict
        word -> the token actually used (hyphen-joined fallback resolved), or
        None if not found. Callers that need per-token counts must use THIS
        token so counts and projections refer to the same vocabulary entry.
    """
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

    row: Dict[str, float] = {}
    lookups: Dict[str, Optional[str]] = {}
    for word in words:
        # Space-separated bigrams are stored hyphen-joined in the models.
        lookup = word if word in model.vocab else word.replace(" ", "-")
        if lookup in model.vocab:
            try:
                row[word] = model.project_onto_dimension(lookup, dimension)
            except ValueError:
                row[word] = np.nan
            lookups[word] = lookup
        else:
            row[word] = np.nan
            lookups[word] = None
    return row, dimension, lookups


def apply_synonyms(
    df: pd.DataFrame,
    synonyms: Optional[Dict[str, List[str]]],
    test_words: Sequence[str],
) -> pd.DataFrame:
    """Average synonym groups and drop non-canonical columns.

    Identical logic to SYNONYMS edit 4 in the year path. Operates row-wise, so
    it is index-agnostic: each row is one model, whether the index is ``year``
    or ``(year, replicate)``. Averaging therefore happens WITHIN a replicate.
    """
    if not synonyms:
        return df
    df = df.copy()
    to_drop: set = set()
    for canonical, tokens in synonyms.items():
        cols = [t for t in tokens if t in df.columns]
        if cols:
            df[canonical] = df[cols].mean(axis=1)
            to_drop.update(t for t in cols if t != canonical)
    df = df.drop(columns=sorted(to_drop), errors="ignore")
    ordered = [w for w in test_words if w in df.columns]
    return df[ordered]


# --------------------------------------------------------------------------
# 2. Model discovery, with the duplicate-year guard
# --------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(?:^|_)y(\d{4})(?:_|$)")
_BARE_YEAR_RE = re.compile(r"^(\d{4})$")


def _parse_year_from_path(rel: Path) -> Optional[int]:
    """Year from anywhere in the relative path — stem first, then parents."""
    for c in [rel.stem, *rel.parts[:-1]]:
        m = _YEAR_RE.search(c)
        if m:
            return int(m.group(1))
        m = _BARE_YEAR_RE.match(c)
        if m and 1800 <= int(m.group(1)) <= 2100:
            return int(m.group(1))
    parts = rel.stem.split("_")
    for part in parts:
        if part.startswith("y") and len(part) > 1:
            try:
                return int(part[1:])
            except ValueError:
                pass
    try:
        return int(parts[-1])
    except ValueError:
        return None


def _parse_replicate(
    stem: str, replicate_pattern: Optional[Union[str, re.Pattern]]
) -> Tuple[str, float, float]:
    """Replicate id and, when the pattern supplies them, corpus/seed indices.

    ``replicate_pattern`` is optional. Without it every file gets a unique
    replicate id (its stem) and corpus/seed come back NaN — enough for the raw
    and residual SDs, NOT enough for the seed/corpus variance split. To enable
    the split, pass a regex with named groups ``corpus`` and ``seed``, e.g.

        replicate_pattern=r"c(?P<corpus>\\d+)_s(?P<seed>\\d+)"

    Check it against one real filename before running 15,000 models through it.
    """
    if replicate_pattern is None:
        return stem, np.nan, np.nan
    pat = (re.compile(replicate_pattern)
           if isinstance(replicate_pattern, str) else replicate_pattern)
    m = pat.search(stem)
    if m is None:
        return stem, np.nan, np.nan
    gd = m.groupdict()
    corpus = float(gd["corpus"]) if gd.get("corpus") is not None else np.nan
    seed = float(gd["seed"]) if gd.get("seed") is not None else np.nan
    return (m.group(0) or stem), corpus, seed


def discover_models(
    model_dir: Union[str, Path],
    glob_pattern: str = "*.kv",
    replicate_pattern: Optional[Union[str, re.Pattern]] = None,
    recursive: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Catalog the models under ``model_dir``.

    Returns a frame with columns [year, replicate, corpus, seed, path].
    Unlike the original inline discovery, this NEVER silently collapses
    several files onto one year — that is what ``require_one_per_year``
    below is for.
    """
    model_dir = Path(model_dir)
    globber = model_dir.rglob if recursive else model_dir.glob
    files = sorted(globber(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {glob_pattern!r} found in {model_dir}"
            f"{' (recursive)' if recursive else ''}"
        )

    rows = []
    skipped = []
    for f in files:
        rel = f.relative_to(model_dir)
        year = _parse_year_from_path(rel)
        if year is None:
            skipped.append(str(rel))
            continue
        rep, corpus, seed = _parse_replicate(rel.as_posix(), replicate_pattern)
        rows.append({"year": year, "replicate": rep,
                     "corpus": corpus, "seed": seed, "path": f})

    if skipped and verbose:
        print(f"⚠️ {len(skipped)} file(s) with non-standard names skipped: "
              f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}")
    if not rows:
        raise ValueError("No valid year-parsable model filenames found.")

    catalog = pd.DataFrame(rows).sort_values(["year", "replicate"])
    return catalog.reset_index(drop=True)


def require_one_per_year(catalog: pd.DataFrame) -> Dict[int, Path]:
    """Year -> path, raising if any year has several models.

    THE GUARD. The original ``year_to_path[year] = f`` loop silently kept
    whichever replicate sorted last, which would produce a normal-looking
    one-row-per-year frame and a noise estimate of exactly zero. Fail loudly
    instead.
    """
    counts = catalog.groupby("year").size()
    dupes = counts[counts > 1]
    if len(dupes):
        detail = ", ".join(f"{y}: {n} files" for y, n in dupes.items())
        raise ValueError(
            "Several models map to the same year — this directory looks like a "
            "replicate ensemble, not a single-model-per-year directory. Use "
            f"compute_projection_over_replicates() instead. ({detail})"
        )
    return dict(zip(catalog["year"], catalog["path"]))


# --------------------------------------------------------------------------
# 3. The replicate path
# --------------------------------------------------------------------------

def compute_projection_over_replicates(
    model_dir: Union[str, Path],
    token_contrasts: Sequence[tuple],
    test_words: Sequence[str],
    years: Optional[Sequence[int]] = None,
    method: str = "meandiff",
    synonyms: Optional[Dict[str, List[str]]] = None,
    baseline_words: Optional[Sequence[str]] = None,
    baseline_agg: str = "median",
    replicate_pattern: Optional[Union[str, re.Pattern]] = None,
    glob_pattern: str = "*.kv",
    recursive: bool = True,
    ensure_sign_positive: Optional[Union[bool, List[str]]] = True,
    verbose: bool = True,
    **method_kwargs,
) -> Dict[str, object]:
    """Project ``test_words`` inside every replicate model.

    The baseline (when ``baseline_words`` is given) is computed FROM THE SAME
    REPLICATE — its words ride along in the same projection call, so the
    common axis wobble partly cancels in the corrected series rather than
    having to be removed afterward. This also avoids the year path's second
    pass over the models, which at ensemble scale would double the I/O.

    ``fem_var`` is deliberately not computed here: the count proxy is the thing
    this program replaces.

    Returns
    -------
    dict with keys:
        'cube'            : (year, replicate) x occupation projections
        'cube_corrected'  : same, baseline-subtracted (empty if no baseline)
        'baseline'        : per (year, replicate) baseline value
        'meta'            : (year, replicate) -> corpus, seed
        'catalog'         : the full discovery frame
        'dimensions'      : (year, replicate) -> fitted dimension vector
        'error_models'    : (year, replicate) -> error string
    """
    catalog = discover_models(
        model_dir, glob_pattern=glob_pattern,
        replicate_pattern=replicate_pattern, recursive=recursive,
        verbose=verbose,
    )
    if years is not None:
        catalog = catalog[catalog["year"].isin(list(years))].reset_index(drop=True)
        if catalog.empty:
            raise ValueError(f"No models found for years {list(years)}")

    if catalog.duplicated(["year", "replicate"]).any():
        bad = catalog[catalog.duplicated(["year", "replicate"], keep=False)]
        raise ValueError(
            "Duplicate (year, replicate) keys — your replicate_pattern is not "
            "discriminating between files. First offenders:\n"
            f"{bad.head(6)[['year', 'replicate', 'path']]}"
        )

    # Every token that has to come out of each model, projected in one pass.
    synonym_extras: set = set()
    if synonyms:
        for tokens in synonyms.values():
            synonym_extras.update(tokens)
    synonym_extras -= set(test_words)
    baseline_words = list(dict.fromkeys(baseline_words)) if baseline_words else []
    baseline_extras = [w for w in baseline_words
                       if w not in test_words and w not in synonym_extras]
    all_words = list(test_words) + sorted(synonym_extras) + baseline_extras

    if verbose:
        n_years = catalog["year"].nunique()
        print(f"📈 Replicate projections: {len(catalog)} models across "
              f"{n_years} year(s)")
        print(f"   Method: {method} | tokens/model: {len(all_words)}")
        if baseline_words:
            print(f"   Baseline: {len(baseline_words)} words, agg={baseline_agg}")

    rows: Dict[Tuple[int, str], Dict[str, float]] = {}
    dimensions: Dict[Tuple[int, str], np.ndarray] = {}
    error_models: Dict[Tuple[int, str], str] = {}

    for rec in catalog.itertuples(index=False):
        key = (int(rec.year), str(rec.replicate))
        try:
            model = W2VModel(str(rec.path))
            row, dimension, _ = project_one_model(
                model, all_words, token_contrasts,
                method=method, ensure_sign_positive=ensure_sign_positive,
                **method_kwargs,
            )
            rows[key] = row
            dimensions[key] = dimension
        except Exception as exc:  # noqa: BLE001
            error_models[key] = str(exc)
            if verbose:
                print(f"   ⚠️ {key}: {exc}")

    if not rows:
        raise RuntimeError("Every replicate failed; see error_models.")

    raw = pd.DataFrame(rows).T
    raw.index = pd.MultiIndex.from_tuples(raw.index, names=["year", "replicate"])
    raw = raw.sort_index()

    # Baseline, computed per model from that model's own projections.
    baseline = pd.Series(dtype=float)
    if baseline_words:
        if baseline_agg not in ("mean", "median"):
            raise ValueError("baseline_agg must be 'mean' or 'median'")
        present = [w for w in baseline_words if w in raw.columns]
        if not present:
            warnings.warn("No baseline words present in any model; "
                          "baseline correction skipped.")
        else:
            vals = raw[present]
            baseline = (vals.median(axis=1) if baseline_agg == "median"
                        else vals.mean(axis=1))

    cube = apply_synonyms(raw, synonyms, test_words)
    if not synonyms:
        cube = cube[[w for w in test_words if w in cube.columns]]

    cube_corrected = pd.DataFrame()
    if not baseline.empty and baseline.notna().any():
        cube_corrected = cube.sub(baseline.fillna(0.0), axis=0)

    meta = (catalog.set_index(["year", "replicate"])[["corpus", "seed"]]
            .reindex(cube.index))

    if verbose:
        n_rep = cube.groupby(level="year").size()
        print(f"✓ cube {cube.shape} | replicates per year: "
              f"min {n_rep.min()}, max {n_rep.max()}")
        if error_models:
            print(f"⚠️ {len(error_models)} model(s) failed")

    return {
        "cube": cube,
        "cube_corrected": cube_corrected,
        "baseline": baseline,
        "meta": meta,
        "catalog": catalog,
        "dimensions": dimensions,
        "error_models": error_models,
    }


# --------------------------------------------------------------------------
# 4. Collapsing the cube
# --------------------------------------------------------------------------

def collapse_cube(cube: pd.DataFrame, mode: str = "residual") -> Dict[str, pd.DataFrame]:
    """Collapse replicates to a per (year, occupation) mean and SD.

    mode='raw'
        SD straight across replicates. Includes the common axis-wobble
        component, which ``year_demean`` later strips — so this OVERSTATES the
        error the CT-SEM actually sees.

    mode='residual'
        Within each year, remove additive replicate and occupation effects by
        double-centering, then take the SD of what's left. The replicate main
        effect absorbs the common axis wobble; the residual is the
        occupation-specific noise, which is the injectable quantity.

    A common rotation is not exactly an additive shift, so the replicate main
    effect absorbs most but not all of it. Compute both and compare — the gap
    tells you how much of the raw SD was common.
    """
    if mode not in ("raw", "residual"):
        raise ValueError("mode must be 'raw' or 'residual'")

    means, sds, counts = {}, {}, {}
    for year, block in cube.groupby(level="year"):
        vals = block.droplevel("year")
        means[year] = vals.mean(axis=0, skipna=True)
        counts[year] = vals.notna().sum(axis=0)

        if mode == "raw":
            sds[year] = vals.std(axis=0, ddof=1, skipna=True)
            continue

        # Double-centre: e = x - rowmean - colmean + grandmean
        arr = vals.to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_m = np.nanmean(arr, axis=1, keepdims=True)
            col_m = np.nanmean(arr, axis=0, keepdims=True)
            grand = np.nanmean(arr)
            resid = arr - row_m - col_m + grand

        R, C = arr.shape
        if R < 2 or C < 2:
            sds[year] = pd.Series(np.nan, index=vals.columns)
            continue

        # Degrees of freedom PER COLUMN, from the counts actually present.
        # The additive two-way fit on N observations with R row effects and
        # C_eff non-empty column effects leaves N - R - C_eff + 1 residual df
        # in total; allocate that to each column in proportion to its share of
        # the observations:
        #     df_j = n_j * (N - R - C_eff + 1) / N
        # Balanced (n_j = R, N = R*C) reduces to the textbook (R-1)(C-1)/C.
        # Using each column's actual n_j is what keeps a partially-observed
        # column from being divided by a full-panel df and so reported as far
        # more precise than it is.
        n_j = np.isfinite(arr).sum(axis=0).astype(float)
        N = float(n_j.sum())
        C_eff = float((n_j > 0).sum())
        df_total = N - R - C_eff + 1.0
        df_col = (n_j * df_total / N) if (N > 0 and df_total > 0) else np.zeros(C)

        ss = np.nansum(resid ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd_vals = np.sqrt(ss / df_col)
        # An absent or too-thin column has no SD. Without this, np.nansum of
        # an all-NaN column returns 0.0 and the occupation reads as measured
        # with perfect precision -- the most dangerous possible failure for a
        # quantity that becomes an observation-error variance.
        sd_vals = np.where((n_j >= 2) & (df_col > 0), sd_vals, np.nan)
        sds[year] = pd.Series(sd_vals, index=vals.columns)

    n_rep = pd.DataFrame(counts).T.sort_index()
    if len(n_rep):
        lo, hi = int(n_rep.to_numpy().min()), int(n_rep.to_numpy().max())
        if lo < hi:
            warnings.warn(
                f"collapse_cube: ragged replicate counts across cells "
                f"(min {lo}, max {hi}). SDs are computed from each cell's own "
                f"count, but thin cells are noisier estimates and the "
                f"seed/corpus split assumes a balanced design.",
                stacklevel=2,
            )

    return {
        "mean": pd.DataFrame(means).T.sort_index(),
        "sd": pd.DataFrame(sds).T.sort_index(),
        "n_replicates": n_rep,
        "mode": mode,
    }


def print_cube_projections(
    cube: pd.DataFrame,
    year: Optional[int] = None,
    test_words: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    agg: str = "mean",
) -> pd.Series:
    """Bar-chart display of ensemble projections, mirroring
    ``W2VModel.print_word_projections`` but sourced from an already-computed
    replicate ``cube`` instead of a live (model, dimension) pair.

    There is no single "the dimension" to hand to ``print_word_projections``
    in the replicate path — every replicate fits its own axis (see
    ``compute_projection_over_replicates``'s docstring) — so this collapses
    replicates to one value per word (mean/median ± SD across replicates)
    and renders it with the same bar-chart layout for visual parity with the
    single-model/ensemble-model sections above.

    Parameters
    ----------
    cube : DataFrame
        (year, replicate) x word projections, e.g. ``result["cube"]``.
    year : int, optional
        Which year's replicates to aggregate. Required if ``cube`` spans
        more than one year.
    test_words : list of str, optional
        Subset/order of words to display. Defaults to all cube columns.
    agg : {"mean", "median"}
        How to collapse replicates for each word.
    """
    years = cube.index.get_level_values("year").unique()
    if year is None:
        if len(years) > 1:
            raise ValueError(f"cube spans several years {list(years)}; pass year=")
        year = years[0]
    block = cube.xs(year, level="year")

    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'")
    values = block.mean(axis=0) if agg == "mean" else block.median(axis=0)
    sds = block.std(axis=0, ddof=1)

    words = list(test_words) if test_words is not None else list(values.index)
    projections = {w: values[w] for w in words if w in values.index and pd.notna(values[w])}

    if title is None:
        title = f"WORD PROJECTIONS ON DIMENSION (replica ensemble {agg}, {year}, n={len(block)})"

    print("\n" + "─" * 100)
    print(title)
    print("─" * 100)
    print(f"{'Word':<20} {'Projection':>18} {'Visualization':>52}")
    print("─" * 100)
    for word, proj in sorted(projections.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(abs(proj) * 30)
        bar = '█' * bar_length if proj > 0 else '░' * bar_length
        proj_str = f"{proj:.4f}±{sds[word]:.4f}"
        print(f"  {word:<18} {proj_str:>18}  {bar:>50}")

    return pd.Series(projections, name=f"{agg}_{year}")


def variance_components(cube: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Split replicate variance into seed and corpus components.

    Requires ``meta`` to carry non-null ``corpus`` and ``seed`` — i.e. a
    ``replicate_pattern`` with named groups was supplied at discovery.

        sigma2_seed   = mean over corpus replicates of the across-seed variance
        sigma2_corpus = var(corpus means) - sigma2_seed / S

    sigma2_corpus can come out slightly negative when the corpus component is
    near zero; that is sampling noise in the subtraction, not an error. It is
    left signed rather than clipped so you can see it.

    Returns a long frame [year, occupation, sigma2_seed, sigma2_corpus,
    sigma2_total, seed_share, n_corpus, n_seed].
    """
    if meta[["corpus", "seed"]].isna().all().any():
        raise ValueError(
            "meta lacks corpus/seed indices — re-run discovery with a "
            "replicate_pattern exposing named groups 'corpus' and 'seed'."
        )

    joined = cube.join(meta)
    out = []
    for year, block in joined.groupby(level="year"):
        occ_cols = [c for c in cube.columns]
        for occ in occ_cols:
            sub = block[[occ, "corpus", "seed"]].dropna()
            if sub["corpus"].nunique() < 2:
                continue
            within = sub.groupby("corpus")[occ].var(ddof=1)
            s_per = sub.groupby("corpus")[occ].size()
            sigma2_seed = float(np.nanmean(within.to_numpy()))
            corpus_means = sub.groupby("corpus")[occ].mean()
            S = float(np.nanmean(s_per.to_numpy()))
            v_between = float(corpus_means.var(ddof=1))
            sigma2_corpus = v_between - (sigma2_seed / S if S > 0 else 0.0)
            total = sigma2_seed + sigma2_corpus
            out.append({
                "year": year,
                "occupation": occ,
                "sigma2_seed": sigma2_seed,
                "sigma2_corpus": sigma2_corpus,
                "sigma2_total": total,
                "seed_share": sigma2_seed / total if total > 0 else np.nan,
                "n_corpus": int(sub["corpus"].nunique()),
                "n_seed": S,
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 5. Injection columns and the gate check
# --------------------------------------------------------------------------

def injection_columns(
    sd: pd.DataFrame,
    k: Optional[int] = None,
    components: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Build the per-obs and TIpred noise columns from collapsed SDs.

    Construction deliberately mirrors the W% side: variances are averaged
    FIRST, then half-logged, so ``fem_logsd`` is on the same scale as
    ``samp_logsd`` and ``mcoef`` means the same thing in both models. Averaging
    log SDs instead gives a different number by Jensen.

    If ``components`` (from ``variance_components``) and ``k`` are supplied,
    the per-observation variance is the error of a k-model ENSEMBLE MEAN,

        sigma2_corpus + sigma2_seed / k

    since seed error averages down across ensemble members and corpus error
    does not (all members resample the same corpus).

    Returns
    -------
    'fem_var_obs'   : year x occupation variance
    'fem_logsd_obs' : 0.5 * log(fem_var_obs)  -> the per-obs (SVE) column
    'fem_logsd'     : occupation-level TIpred, 0.5*log(mean variance)
    """
    if components is not None:
        if k is None:
            raise ValueError("k is required when components are supplied")
        wide = components.pivot(index="year", columns="occupation",
                                values=["sigma2_corpus", "sigma2_seed"])
        var_obs = wide["sigma2_corpus"] + wide["sigma2_seed"] / float(k)
        var_obs = var_obs.reindex(columns=sd.columns)
    else:
        var_obs = sd ** 2

    var_obs = var_obs.where(var_obs > 0)
    logsd_obs = 0.5 * np.log(var_obs)
    tipred = 0.5 * np.log(var_obs.mean(axis=0, skipna=True))
    tipred.name = "fem_logsd"

    return {
        "fem_var_obs": var_obs,
        "fem_logsd_obs": logsd_obs,
        "fem_logsd": tipred,
    }


def detrend_panel(
    projections: pd.DataFrame,
    *,
    order: int = 3,
    year_demean: bool = True,
) -> pd.DataFrame:
    """Apply the CT-SEM panel's detrending to a years x occupation frame.

    Mirrors the wrapper's ``year_demean+<poly>`` preprocessing: subtract each
    year's cross-occupation mean (mean-preserving), then remove a
    per-occupation polynomial in calendar year (also mean-preserving). Both
    steps are what the fitted model sees, so a variance computed from the
    result is on the same footing as noise measured across replicates.
    """
    df = projections.astype("float64")
    if year_demean:
        ybar = df.mean(axis=1)
        df = df.sub(ybar, axis=0).add(float(np.nanmean(ybar.to_numpy())))

    if order and order > 0:
        out = {}
        for col in df.columns:
            s = df[col]
            v = s.dropna()
            if len(v) < order + 2:
                out[col] = s
                continue
            coef = np.polyfit(v.index.to_numpy(dtype=float),
                              v.to_numpy(), order)
            fitted = np.polyval(coef, s.index.to_numpy(dtype=float))
            out[col] = s - fitted + float(v.mean())
        df = pd.DataFrame(out, index=df.index)[list(df.columns)]
    return df


def noise_share(
    var_obs: pd.DataFrame,
    projections: pd.DataFrame,
    *,
    year: Optional[int] = None,
    order: int = 3,
    year_demean: bool = True,
    dof_correct: bool = True,
) -> pd.DataFrame:
    """THE GATE: measured noise variance as a share of deviation variance.

    Both sides are in RAW PROJECTION UNITS, which is what makes the comparison
    meaningful. The CT-SEM rescales its inputs, so a fitted MANIFESTVAR is NOT
    comparable to a variance measured across replicates -- an earlier version
    of this function divided by a hard-coded MANIFESTcov and returned numbers
    two orders of magnitude too small. Do not reinstate that comparison.

    ``var_obs``      year x occupation replicate variance, e.g.
                     ``collapse_cube(cube)["sd"] ** 2``.
    ``projections``  year x occupation PRODUCTION series, UNDETRENDED -- the
                     same object the panel is built from. Detrending is
                     applied here so both sides match.

    Reading ``share``: near 1 means the replicate ensemble accounts for
    essentially all of the deviation-scale variance -- the detrended series is
    measurement noise nearly all the way down, and the instrument sees it.
    Near 0.01 means the instrument sees only a thin floor beneath a much
    thicker corpus-composition layer, and an injection built on it would
    behave like the count-based proxy did.

    ``share`` somewhat above 1 is expected, not alarming. With
    ``dof_correct``, the degrees of freedom spent on detrending are accounted
    for (approximately: order+1 polynomial terms per occupation, plus
    n_years/n_occupations for the year-demeaning). What remains is dominated
    by the perturbation regime -- Poisson replicates train on corpora missing
    ~37% of their types, so their spread overstates the error of a
    full-corpus fit. Measure that inflation against an unperturbed seed-only
    ensemble before quoting agreement with any model-internal number.

    ``lag1`` is an independent route to the same question: a detrended series
    that is nearly all white noise has lag-1 autocorrelation near zero,
    implying a noise fraction near 1 - lag1. It uses no replicate information,
    so agreement between ``share`` and ``1 - lag1`` is real corroboration.

    Returns a frame indexed by occupation with columns
    [noise_var, total_var, share, lag1], sorted by share.
    """
    noise = (var_obs.loc[year] if year is not None
             else var_obs.mean(axis=0, skipna=True))
    det = detrend_panel(projections, order=order, year_demean=year_demean)

    common = noise.index.intersection(det.columns)
    if len(common) == 0:
        raise ValueError(
            "no occupations in common between var_obs and projections -- "
            "check both use the same (post-synonym canonical) column labels"
        )
    if len(common) < len(noise.index) or len(common) < len(det.columns):
        warnings.warn(
            f"noise_share: using {len(common)} occupation(s) common to both "
            f"inputs (var_obs has {len(noise.index)}, projections "
            f"{len(det.columns)})"
        )

    det = det[common]
    total = det.var(axis=0, ddof=1)
    if dof_correct:
        n = int(det.notna().sum(axis=0).median())
        spent = (order + 1 if order else 0) + (n / len(common) if year_demean else 0.0)
        if n - spent > 1:
            total = total * (n - 1) / (n - spent)

    out = pd.DataFrame({
        "noise_var": noise[common].astype(float),
        "total_var": total.astype(float),
        "lag1": det.apply(lambda s: s.autocorr(1)).astype(float),
    })
    out["share"] = out["noise_var"] / out["total_var"]
    return out[["noise_var", "total_var", "share", "lag1"]].sort_values("share")


# --------------------------------------------------------------------------
# PATCH for dimension_projection_time_series.py
# --------------------------------------------------------------------------
# Two edits keep the year path on this same core.
#
# (a) Add near the top:
#
#     from .replicate_projection import project_one_model, discover_models, \
#         require_one_per_year
#
# (b) Replace the discovery block (from ``model_files = sorted(...)`` through
#     ``available_years = sorted(set(available_years))``) with:
#
#     _catalog = discover_models(model_dir, verbose=verbose)
#     year_to_path = require_one_per_year(_catalog)
#     available_years = sorted(year_to_path)
#
# (c) Inside the year loop, replace the dimension/projection block — from
#     ``if method.lower() == "pca":`` through the end of the
#     ``for word in _all_words:`` loop — with:
#
#     row, dimension, _lookups = project_one_model(
#         model, _all_words, token_contrasts, method=method,
#         ensure_sign_positive=ensure_sign_positive, **method_kwargs)
#     yearly_dimensions[year] = dimension
#     count_row = {}
#     for word in _all_words:
#         lookup = _lookups.get(word)
#         if emit_fem_var and lookup is not None:
#             c = _token_count(model, lookup)
#             if (subsample_t and np.isfinite(c) and c > 0
#                     and np.isfinite(_year_total) and _year_total > 0):
#                 f = c / _year_total
#                 p_keep = min(1.0, (np.sqrt(f / subsample_t) + 1.0)
#                              * (subsample_t / f))
#                 c = c * p_keep
#             count_row[word] = c if (np.isfinite(c) and c >= 1.0) else np.nan
#         else:
#             count_row[word] = np.nan
#
# The FEMVAR invariant is preserved: counts are read with the SAME token the
# projection used, now handed back explicitly in ``_lookups`` instead of being
# a shared local. Everything downstream — synonym averaging, fem_var, baseline
# correction — is untouched, and the year path's integer index is unchanged.