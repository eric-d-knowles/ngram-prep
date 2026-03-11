"""
This module provides two main workflows:
1. **Raw aggregation**: Convert already-downloaded IPUMS extracts (CSV/Parquet/Stata)
   into BLS-compatible profession CSVs via `aggregate_ipums_professions_csv()`.
2. **Web fetch + aggregation**: Retrieve data from the IPUMS API, download extracts,
   and aggregate them in one step via `fetch_and_aggregate_ipums_professions_csv()`.

The IPUMS API requires `ipumspy` (pip install ipumspy) and an IPUMS API key.
Get your API key at https://account.ipums.org/api.

Prestige workflow:
3. **Static prestige scores**: Fetch an OCC2010→prestige crosswalk from IPUMS USA
   via `fetch_prestige_crosswalk()` (one-time, cacheable), aggregate weighted-mean
   prestige per generic label from the CPS extract via `aggregate_prestige_by_label()`,
   then join onto a panel DataFrame via `add_prestige_to_panel()`.
   Requires IPUMS USA registration at https://uma.pop.umn.edu/usa/registration/new.
"""

from pathlib import Path
import csv
import os
import re
import time
from typing import Optional, List, Dict, Any, Iterable, Union

import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    "TotalEmployed",
    "Women",
    "AfricanAmerican",
    "Asian",
    "HispanicLatino",
    "none",
    "label1",
    "label2",
    "label3",
    "label4",
    "label5",
]

LABEL_STOPWORDS = {
    "and", "or", "of", "the", "a", "an", "for", "to", "in", "on", "at", "by", "with",
    "except", "including", "all", "other", "miscellaneous", "related", "non", "total",
    "first", "second", "third", "line", "years", "over", "percent", "employed",
}

GENERIC_ROLE_WORDS = {
    "accountant", "advisor", "aide", "analyst", "appraiser", "architect", "assembler", "assistant", "attendant",
    "auditor", "bailiff", "baker", "barber", "bartender", "brickmason", "blockmason", "carpenter", "cashier",
    "chef", "chemist", "clergy", "clerk", "collector", "concierge", "conductor", "cook", "counselor", "courier",
    "developer", "dishwasher", "doctor", "drafter", "driver", "editor", "educator", "electrician", "engineer",
    "estimator", "examiner", "finisher", "firefighter", "fitter", "guard", "hairdresser", "handler", "helper",
    "hygienist", "inspector", "installer", "instructor", "investigator", "jailer", "janitor", "laborer", "lawyer",
    "librarian", "machinist", "manager", "mason", "mechanic", "messenger", "modeler", "mover", "nurse",
    "nutritionist", "officer", "operator", "paramedic", "paralegal", "pathologist", "pharmacist", "phlebotomist",
    "pilot", "pipelayer", "pipefitter", "planner", "plumber", "porter", "practitioner", "president", "programmer",
    "psychologist", "receptionist", "repairer", "roofer", "salesperson", "scientist", "secretary", "specialist",
    "steamfitter", "stonemason", "supervisor", "surgeon", "taper", "teacher", "technician", "teller", "therapist",
    "veterinarian", "waiter", "waitress", "worker", "writer",
}


# ── Label tokenization helpers ────────────────────────────────────────────────

def _is_generic_role_token(token):
    """Check if a token is a generic occupational role word."""
    if token in GENERIC_ROLE_WORDS:
        return True
    for role in GENERIC_ROLE_WORDS:
        if len(role) >= 4 and token.endswith(role) and len(token) > len(role):
            return True
    return False


def _normalize_label_token(token):
    """Normalize an occupation label token (handle plurals, etc.)."""
    token = token.lower().strip()
    if token.endswith("men") and len(token) > 5:
        token = token[:-3] + "man"
    if token.endswith("s") and len(token) > 3:
        singular = token[:-1]
        if singular in GENERIC_ROLE_WORDS or _is_generic_role_token(singular):
            token = singular
    return token


def _tokenize_occupation(occupation, max_tokens=5):
    """
    Tokenize and extract meaningful role words from an occupation label.

    Args:
        occupation: Raw occupation string from BLS/IPUMS table
        max_tokens: Maximum number of tokens to return
    Returns:
        List of extracted tokens (padded to max_tokens length)
    """
    text = str(occupation).lower()
    text = re.sub(r"[\(\)\[\].;:]+", " ", text)
    text = text.replace("&", " and ")

    segments = [s.strip() for s in re.split(r",|/|\band\b|\bor\b", text) if s.strip()]

    def cleaned_tokens(segment):
        raw = re.findall(r"[a-z0-9']+", segment)
        out = []
        for token in raw:
            normalized = _normalize_label_token(token)
            if (
                normalized
                and normalized not in LABEL_STOPWORDS
                and not normalized.isdigit()
                and len(normalized) > 1
            ):
                out.append(normalized)
        return out

    segment_heads = []
    for segment in segments:
        tokens = cleaned_tokens(segment)
        if not tokens:
            continue
        role_tokens = [t for t in tokens if _is_generic_role_token(t)]
        if "supervisor" in role_tokens:
            segment_heads.append("supervisor")
        elif role_tokens:
            segment_heads.append(role_tokens[-1])

    all_tokens = cleaned_tokens(text)
    role_tokens_global = [t for t in all_tokens if _is_generic_role_token(t)]

    ordered = segment_heads + role_tokens_global
    if not ordered:
        ordered = all_tokens

    deduped = []
    seen = set()
    for token in ordered:
        if token not in seen:
            deduped.append(token)
            seen.add(token)

    return (deduped[:max_tokens] + [""] * max_tokens)[:max_tokens]


# ── Internal I/O helpers ──────────────────────────────────────────────────────

def _output_path_with_year(output_csv, year, inject_year_in_filename=True):
    """Return output path with year injected before extension if requested."""
    path = Path(output_csv)
    if inject_year_in_filename and year is not None:
        stem = path.stem
        ext = path.suffix
        return str(path.parent / f"{stem}_{year}{ext}")
    return str(output_csv)


def _read_ipums_extract(extract_file):
    """Read an IPUMS extract from CSV, Parquet, or Stata format."""
    path = Path(extract_file)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".dta":
        return pd.read_stata(path)

    raise ValueError(
        f"Unsupported extract format for {extract_file}. "
        "Use .csv/.csv.gz, .parquet, or .dta."
    )


def _load_occupation_map(map_file, code_col="code", label_col="label"):
    """Load a two-column occupation map (code -> label)."""
    map_df = pd.read_csv(map_file)
    missing = [col for col in [code_col, label_col] if col not in map_df.columns]
    if missing:
        raise ValueError(f"occupation_map_file missing required columns: {missing}")

    mapping = (
        map_df[[code_col, label_col]]
        .dropna()
        .drop_duplicates(subset=[code_col])
    )
    mapping[code_col] = mapping[code_col].astype(str).str.strip()
    mapping[label_col] = mapping[label_col].astype(str).str.strip()
    return dict(zip(mapping[code_col], mapping[label_col]))


def _format_percent(x):
    """Format percentage values to match BLS CSV conventions."""
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _resolve_occupation_labels(df, occupation_code_col, occupation_label_col, occupation_map_file):
    """Return a standardized occupation label series."""
    if occupation_label_col and occupation_label_col in df.columns:
        labels = df[occupation_label_col].astype("string").str.strip()
        return labels.mask(labels == "", pd.NA)

    if occupation_map_file:
        mapping = _load_occupation_map(occupation_map_file)
        codes = df[occupation_code_col].astype(str).str.strip()
        labels = codes.map(mapping)
        labels = labels.astype("string").str.strip()
        return labels.mask(labels == "", pd.NA)

    raise ValueError(
        "Need occupation labels to build label1-label5 columns. "
        "Provide `occupation_label_col` in extract or `occupation_map_file`."
    )


# ── IPUMS API client ──────────────────────────────────────────────────────────

def _get_ipums_api_client(api_key: Optional[str] = None):
    """
    Get an IPUMS API client, reading the API key from the environment or parameter.

    Args:
        api_key: Optional API key. If not provided, tries to read from IPUMS_API_KEY env var.

    Returns:
        IpumsApiClient instance

    Raises:
        ImportError: If ipumspy is not installed
        ValueError: If no API key is found
    """
    try:
        from ipumspy import IpumsApiClient
    except ImportError:
        raise ImportError(
            "ipumspy is required for web fetching. "
            "Install with: pip install ipumspy"
        )

    if api_key is None:
        api_key = os.getenv("IPUMS_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing IPUMS_API_KEY. Set the environment variable or pass api_key parameter. "
            "Get your key at https://account.ipums.org/api"
        )

    return IpumsApiClient(api_key=api_key, base_url="https://api.ipums.org", api_version=2)


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_ipums_microdata_cps(
    api_key: Optional[str] = None,
    years: Optional[Iterable[int]] = None,
    samples: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    download_dir: Optional[str] = None,
    initial_wait_time: float = 2,
    max_wait_time: float = 30,
    timeout: float = 600,
    description: str = "lexichron IPUMS CPS extract",
    data_format: str = "csv",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fetch and download IPUMS CPS ASEC microdata via the IPUMS API.

    Args:
        api_key: IPUMS API key (reads from IPUMS_API_KEY env var if not provided)
        years: ASEC years to download - accepts a list, range, or any iterable of ints.
               Examples: [2015, 2020, 2024] or range(2010, 2025).
               Each year is resolved to its ASEC sample ID automatically.
               If None and samples is None, uses the most recent ASEC sample.
               Cannot be used together with `samples`.
        samples: List of ASEC sample identifiers (e.g., ['cps2024_03s']).
                 If None and years is None, uses the most recent ASEC sample.
                 Only ASEC samples are supported (harmonized occupation codes + person weights).
                 Cannot be used together with `years`.
        variables: List of variable names to extract (default: occupation + demographics).
                   If None, uses: ['YEAR', 'SEX', 'RACE', 'HISPAN', 'OCC2010', 'ASECWT']
                   OCC2010 provides harmonized detailed occupation codes across years.
                   Add 'OCCSCORE' to include prestige scores for use with
                   aggregate_prestige_by_label().
        download_dir: Directory to save downloaded files. Uses ipums_api_downloads/ in cwd if not specified.
        initial_wait_time: Initial seconds to wait before polling API status (default: 2)
        max_wait_time: Max seconds between polls (default: 30)
        timeout: Max total seconds to wait for extract completion (default: 600/10 min)
        description: Description for the extract (helps identify in IPUMS account)
        data_format: Output format ('csv', 'parquet', 'stata', 'dta')
        verbose: Print status messages

    Returns:
        Dictionary with keys:
        - 'extract_id': IPUMS extract ID
        - 'status': Final status ('completed' or error message)
        - 'download_dir': Path to downloaded files
        - 'files': List of downloaded file paths
        - 'samples_used': Sample IDs that were submitted

    Example:
        >>> result = fetch_ipums_microdata_cps(years=[2020, 2024])
        >>> print(f"Downloaded to {result['download_dir']}")
        >>>
        >>> # Include prestige scores
        >>> result = fetch_ipums_microdata_cps(
        ...     years=range(1968, 2026),
        ...     variables=['YEAR', 'SEX', 'RACE', 'HISPAN', 'OCC2010', 'ASECWT', 'OCCSCORE'],
        ... )
    """
    try:
        from ipumspy import MicrodataExtract
    except ImportError:
        raise ImportError(
            "ipumspy is required for web fetching. "
            "Install with: pip install ipumspy"
        )

    client = _get_ipums_api_client(api_key)

    if years is not None and samples is not None:
        raise ValueError("Cannot specify both 'years' and 'samples'. Use one or the other.")

    if download_dir is None:
        download_dir = "ipums_api_downloads"

    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Discovering available CPS ASEC samples...")
    sample_info = client.get_all_sample_info("cps")
    asec_catalog = {
        sid: desc for sid, desc in sample_info.items()
        if "asec" in str(desc).lower()
    }
    if not asec_catalog:
        raise ValueError(
            "No ASEC samples found in IPUMS CPS catalog. "
            "Only ASEC samples are supported (yearly data with "
            "harmonized occupation codes and person weights)."
        )

    if years is not None:
        year_to_sid: Dict[int, str] = {}
        for sid, desc in asec_catalog.items():
            m = re.search(r'\b(\d{4})\b', str(desc))
            if m:
                year_to_sid[int(m.group(1))] = sid
            else:
                m2 = re.search(r'cps(\d{4})', sid)
                if m2:
                    year_to_sid[int(m2.group(1))] = sid

        missing = [y for y in years if y not in year_to_sid]
        if missing:
            available = sorted(year_to_sid.keys())
            raise ValueError(
                f"No ASEC samples found for year(s): {missing}. "
                f"Available ASEC years: {available}"
            )
        samples = [year_to_sid[y] for y in sorted(years)]
        if verbose:
            print(f"  Resolved years {sorted(years)} -> samples {samples}")

    elif samples is None:
        samples = [sorted(asec_catalog.keys())[-1]]
        if verbose:
            print(f"  Selected most recent ASEC sample: {samples[0]}")
    else:
        for s in samples:
            if "_03" not in s:
                import warnings
                warnings.warn(
                    f"Sample '{s}' does not look like an ASEC March supplement. "
                    f"ASEC samples typically have '_03' in the ID (e.g. 'cps2025_03s'). "
                    f"Proceeding anyway.",
                    stacklevel=2,
                )

    if variables is None:
        variables = ["YEAR", "SEX", "RACE", "HISPAN", "OCC2010", "ASECWT"]
        if verbose:
            print(f"  Auto-selected variables: {variables}")

    extract = MicrodataExtract(
        collection="cps",
        samples=samples,
        variables=variables,
        description=description,
        data_format=data_format,
    )

    if verbose:
        print(f"Submitting extract (samples={samples}, variables={variables})...")
    submitted = client.submit_extract(extract)
    extract_id = submitted.extract_id
    if verbose:
        print(f"  Extract ID: {extract_id}")
        print(f"  Status: {client.extract_status(extract_id, collection='cps')}")

    if verbose:
        print("Waiting for extract to complete...")
    try:
        client.wait_for_extract(
            submitted,
            collection="cps",
            inital_wait_time=initial_wait_time,
            max_wait_time=max_wait_time,
            timeout=timeout,
        )
        final_status = client.extract_status(extract_id, collection="cps")
        if verbose:
            print(f"  Final status: {final_status}")
    except Exception as exc:
        return {
            "extract_id": extract_id,
            "status": f"failed: {str(exc)}",
            "download_dir": str(download_path),
            "files": [],
            "samples_used": samples,
        }

    if verbose:
        print(f"Downloading to {download_path}...")
    try:
        client.download_extract(
            extract_id, download_dir=str(download_path), collection="cps"
        )
        import gzip
        import shutil
        for gz_path in list(download_path.glob("*.gz")):
            out_path = gz_path.with_suffix("")
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
            if verbose:
                print(f"  Decompressed {gz_path.name} -> {out_path.name}")

        supported_suffixes = {".csv", ".parquet", ".dta", ".stata"}
        files = sorted([
            str(p) for p in download_path.glob("*")
            if p.is_file() and p.suffix.lower() in supported_suffixes
        ])
        if verbose:
            print(f"  Downloaded {len(files)} data file(s)")
            for f in files:
                print(f"    - {Path(f).name}  ({Path(f).stat().st_size / 1024:.0f} KB)")
    except Exception as exc:
        return {
            "extract_id": extract_id,
            "status": f"download failed: {str(exc)}",
            "download_dir": str(download_path),
            "files": [],
            "samples_used": samples,
        }

    return {
        "extract_id": extract_id,
        "status": "completed",
        "download_dir": str(download_path),
        "files": files,
        "samples_used": samples,
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_ipums_professions_csv(
    extract_file,
    output_csv="professionsIPUMS.csv",
    year=None,
    occupation_code_col="OCC1990",
    occupation_label_col=None,
    occupation_map_file=None,
    year_col="YEAR",
    sex_col="SEX",
    race_col="RACE",
    hispanic_col="HISPAN",
    weight_col="PERWT",
    female_codes=(2,),
    black_codes=(2,),
    asian_codes=(4, 5, 6),
    non_hispanic_codes=(0, 9),
    min_total_employed=50,
    inject_year_in_filename=True,
    return_year=False,
):
    """
    Aggregate IPUMS person-level data into a BLS-compatible profession CSV.

    Args:
        extract_file: Path to IPUMS extract (.csv/.csv.gz/.parquet/.dta)
        output_csv: Output CSV path
        year: Optional year filter (single year)
        occupation_code_col: Occupation code column (e.g., OCC1990)
        occupation_label_col: Occupation label text column (if present)
        occupation_map_file: Optional CSV mapping occupation codes to labels
        year_col: Year column name
        sex_col: Sex column name (IPUMS default: SEX, female code usually 2)
        race_col: Race column name
        hispanic_col: Hispanic origin column name
        weight_col: Person weight column
        female_codes: Values treated as female in sex_col
        black_codes: Values treated as Black/African American in race_col
        asian_codes: Values treated as Asian in race_col
        non_hispanic_codes: Values treated as non-Hispanic in hispanic_col
        min_total_employed: Minimum weighted total employment to keep row
        inject_year_in_filename: Inject year in output filename when available
        return_year: If True, returns (DataFrame, year)

    Returns:
        DataFrame in the same schema as BLS profession CSV output.
    """
    df = _read_ipums_extract(extract_file)

    required = [occupation_code_col, sex_col, weight_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"extract_file missing required column: {col}")

    if year is not None:
        if year_col not in df.columns:
            raise ValueError(f"year filter requested but `{year_col}` not found in extract")
        df = df[df[year_col] == year].copy()

    if df.empty:
        raise ValueError("No rows available after filtering extract")

    labels = _resolve_occupation_labels(
        df=df,
        occupation_code_col=occupation_code_col,
        occupation_label_col=occupation_label_col,
        occupation_map_file=occupation_map_file,
    )

    work = pd.DataFrame(
        {
            "Occupation": labels,
            "weight": pd.to_numeric(df[weight_col], errors="coerce"),
            "is_woman": df[sex_col].isin(female_codes),
            "is_black": df[race_col].isin(black_codes) if race_col in df.columns else False,
            "is_asian": df[race_col].isin(asian_codes) if race_col in df.columns else False,
            "is_hispanic": (~df[hispanic_col].isin(non_hispanic_codes)) if hispanic_col in df.columns else False,
        }
    )

    work = work.dropna(subset=["Occupation", "weight"]).copy()
    work["Occupation"] = work["Occupation"].astype(str).str.strip()
    work = work[(work["Occupation"] != "") & (work["weight"] > 0)]

    if work.empty:
        raise ValueError("No valid weighted occupation rows found in extract")

    work["women_weight"]    = work["weight"].where(work["is_woman"],    0.0)
    work["black_weight"]    = work["weight"].where(work["is_black"],    0.0)
    work["asian_weight"]    = work["weight"].where(work["is_asian"],    0.0)
    work["hispanic_weight"] = work["weight"].where(work["is_hispanic"], 0.0)

    grouped = work.groupby("Occupation", dropna=False, as_index=False).agg(
        TotalEmployed=("weight",          "sum"),
        women_weight= ("women_weight",    "sum"),
        black_weight= ("black_weight",    "sum"),
        asian_weight= ("asian_weight",    "sum"),
        hispanic_weight=("hispanic_weight","sum"),
    )

    agg = pd.DataFrame(
        {
            "Occupation":    grouped["Occupation"],
            "TotalEmployed": grouped["TotalEmployed"],
            "Women":         100.0 * grouped["women_weight"]    / grouped["TotalEmployed"],
            "AfricanAmerican":100.0 * grouped["black_weight"]   / grouped["TotalEmployed"],
            "Asian":         100.0 * grouped["asian_weight"]    / grouped["TotalEmployed"],
            "HispanicLatino":100.0 * grouped["hispanic_weight"] / grouped["TotalEmployed"],
        }
    )

    cleaned = agg[agg["TotalEmployed"] >= float(min_total_employed)].copy()
    if cleaned.empty:
        raise ValueError("No occupation rows meet min_total_employed threshold")

    cleaned["TotalEmployed"] = cleaned["TotalEmployed"].round().astype(int)
    cleaned = cleaned.sort_values("Women", ascending=True).reset_index(drop=True)

    label_columns = ["label1", "label2", "label3", "label4", "label5"]
    label_df = pd.DataFrame(
        cleaned["Occupation"].map(lambda x: _tokenize_occupation(x, max_tokens=5)).tolist(),
        columns=label_columns,
        index=cleaned.index,
    )

    out = pd.DataFrame(
        {
            "TotalEmployed": cleaned["TotalEmployed"],
            "Women":         cleaned["Women"],
            "AfricanAmerican":cleaned["AfricanAmerican"],
            "Asian":         cleaned["Asian"],
            "HispanicLatino":cleaned["HispanicLatino"],
            "none":          "",
        }
    )
    out = pd.concat([out, label_df], axis=1)
    out = out[TARGET_COLUMNS]

    detected_year = None
    if year is not None:
        detected_year = int(year)
    elif year_col in df.columns:
        year_values = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
        if len(year_values.unique()) == 1:
            detected_year = int(year_values.iloc[0])

    resolved_output_csv = _output_path_with_year(
        output_csv,
        detected_year,
        inject_year_in_filename=inject_year_in_filename,
    )

    with open(resolved_output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TARGET_COLUMNS)
        for row in out.itertuples(index=False):
            writer.writerow(
                [
                    int(row.TotalEmployed),
                    _format_percent(row.Women),
                    _format_percent(row.AfricanAmerican),
                    _format_percent(row.Asian),
                    _format_percent(row.HispanicLatino),
                    "",
                    row.label1,
                    row.label2,
                    row.label3,
                    row.label4,
                    row.label5,
                ]
            )

    print(f"Saved {len(out)} rows to {resolved_output_csv}")
    if detected_year is not None:
        print(f"Detected IPUMS reference year: {detected_year}")

    out.attrs["ipums_reference_year"] = detected_year
    out.attrs["output_csv"] = resolved_output_csv

    if return_year:
        return out, detected_year
    return out


def aggregate_ipums_professions_csv_batch(
    extract_file,
    output_dir,
    years=None,
    output_basename="professionsIPUMS.csv",
    continue_on_error=True,
    **kwargs,
):
    """Export one BLS-compatible CSV per year from an IPUMS extract."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    df = _read_ipums_extract(extract_file)
    year_col = kwargs.get("year_col", "YEAR")
    if years is None:
        if year_col not in df.columns:
            raise ValueError("years=None requires a year column in extract")
        year_values = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
        years = sorted(year_values.unique().tolist())

    runs = []
    output_csv = str(output_dir_path / output_basename)

    for year in years:
        try:
            out_df, out_year = aggregate_ipums_professions_csv(
                extract_file=extract_file,
                output_csv=output_csv,
                year=year,
                return_year=True,
                **kwargs,
            )
            runs.append(
                {
                    "year":       out_year,
                    "rows":       len(out_df),
                    "status":     "ok",
                    "output_csv": out_df.attrs.get("output_csv", ""),
                    "error":      "",
                }
            )
        except Exception as exc:
            runs.append(
                {
                    "year":       year,
                    "rows":       None,
                    "status":     "failed",
                    "output_csv": "",
                    "error":      str(exc),
                }
            )
            if not continue_on_error:
                break

    runs_df = pd.DataFrame(runs)
    if runs_df.empty:
        return runs_df

    return runs_df[["year", "rows", "status", "output_csv", "error"]].sort_values(
        by=["year", "status"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def fetch_and_aggregate_ipums_professions_csv(
    output_csv: str = "professionsIPUMS.csv",
    occupation_map_file: Optional[str] = None,
    api_key: Optional[str] = None,
    years: Optional[Iterable[int]] = None,
    samples: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    download_dir: Optional[str] = None,
    keep_extract_file: bool = True,
    year: Optional[int] = None,
    occupation_code_col: str = "OCC2010",
    occupation_label_col: Optional[str] = None,
    year_col: str = "YEAR",
    sex_col: str = "SEX",
    race_col: str = "RACE",
    hispanic_col: str = "HISPAN",
    weight_col: Optional[str] = None,
    female_codes: tuple = (2,),
    black_codes: tuple = (2,),
    asian_codes: tuple = (4, 5, 6),
    non_hispanic_codes: tuple = (0, 9),
    min_total_employed: int = 50,
    inject_year_in_filename: bool = True,
    initial_wait_time: float = 2,
    max_wait_time: float = 30,
    timeout: float = 600,
    return_year: bool = False,
    overwrite: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    Fetch CPS ASEC data from IPUMS API and aggregate into a BLS-compatible profession CSV.

    This is a convenience function that combines `fetch_ipums_microdata_cps()` and
    `aggregate_ipums_professions_csv()` in one step. Only ASEC samples are supported
    (yearly data with harmonized occupation codes and person weights).

    Args:
        output_csv: Output CSV path (year will be injected if inject_year_in_filename=True).
                   When years is provided, this is used as output_basename for the batch.
        occupation_map_file: CSV file mapping occupation codes to labels (required)
        api_key: IPUMS API key (reads from IPUMS_API_KEY env var if not provided)
        years: ASEC years to download and aggregate - accepts a list, range, or any
               iterable of ints (e.g. range(1968, 2026)). Each year is fetched as a
               single extract and then batch-aggregated into per-year CSVs.
               Cannot be used together with `samples`.
        samples: CPS ASEC sample IDs (e.g., ['cps2024_03s']). Uses most recent ASEC if None.
        variables: Variable names to fetch. If None, uses standard ASEC set.
        download_dir: Directory for downloaded extracts. Uses ipums_api_downloads/ in cwd if None.
        keep_extract_file: If True, keep the downloaded extract file after aggregation (default: True).
                   Set to False to delete after use and save disk space.
        year: Optional filter to extract specific year (extracts all years if None)
        occupation_code_col: Occupation code column name (default: 'OCC2010', harmonized)
        occupation_label_col: Occupation label column (if in extract)
        year_col: Year column name
        sex_col: Sex column name
        race_col: Race column name
        hispanic_col: Hispanic origin column name
        weight_col: Person weight column name. If None, auto-detects from fetched data
                    (defaults to ASECWT for ASEC samples).
        female_codes: Values representing female in sex_col
        black_codes: Values representing Black/African American in race_col
        asian_codes: Values representing Asian in race_col
        non_hispanic_codes: Values representing non-Hispanic in hispanic_col
        min_total_employed: Minimum weighted employment to keep row
        inject_year_in_filename: Inject detected year into output filename
        initial_wait_time: Initial wait time (seconds) before polling API
        max_wait_time: Max wait time (seconds) between polls
        timeout: Total timeout (seconds) for extract completion
        return_year: If True, returns (DataFrame, year)
        overwrite: If False, skip download and use existing files in download_dir (default: True).
                   In multi-year mode, returns results for years with existing aggregated CSVs.
        verbose: Print status messages

    Returns:
        When years is provided: DataFrame of batch results (year, rows, status, output_csv, error).
        Otherwise: DataFrame in BLS profession CSV schema. If return_year=True, returns (DataFrame, year).

    Raises:
        ImportError: If ipumspy is not installed
        ValueError: If occupation_map_file is not provided or missing columns

    Example:
        >>> # Single-sample fetch
        >>> out_df = fetch_and_aggregate_ipums_professions_csv(
        ...     output_csv='/data/professions_ipums.csv',
        ...     occupation_map_file='/data/occ1990_map.csv',
        ...     samples=['cps2024_05s2'],
        ... )
        >>> print(f"Aggregated {len(out_df)} occupations")
        >>>
        >>> # Multi-year batch fetch
        >>> runs_df = fetch_and_aggregate_ipums_professions_csv(
        ...     occupation_map_file='/data/occ2010_map.csv',
        ...     years=range(2010, 2026),
        ... )
    """
    if occupation_map_file is None:
        raise ValueError("occupation_map_file is required for web fetching")

    if download_dir is None:
        download_dir = "ipums_api_downloads"

    download_path = Path(download_dir)

    if not overwrite:
        if years is not None:
            download_path.mkdir(parents=True, exist_ok=True)
            base_name = Path(output_csv).stem
            if inject_year_in_filename:
                existing_runs = []
                for yr in sorted(years):
                    expected_csv = download_path / f"{base_name}_{yr}.csv"
                    if expected_csv.exists():
                        try:
                            df = pd.read_csv(expected_csv)
                            existing_runs.append({
                                "year": yr, "rows": len(df), "status": "ok",
                                "output_csv": str(expected_csv), "error": "",
                            })
                        except Exception as e:
                            existing_runs.append({
                                "year": yr, "rows": None, "status": "failed",
                                "output_csv": str(expected_csv),
                                "error": f"Could not read existing file: {e}",
                            })
                if existing_runs:
                    if verbose:
                        print(f"overwrite=False: Using {len(existing_runs)} existing file(s) from {download_dir}")
                        for run in existing_runs[:3]:
                            print(f"  - {Path(run['output_csv']).name} ({run['rows']} rows)")
                        if len(existing_runs) > 3:
                            print(f"  ... and {len(existing_runs) - 3} more")
                    return pd.DataFrame(existing_runs)[["year", "rows", "status", "output_csv", "error"]].reset_index(drop=True)
                if verbose:
                    print(f"overwrite=False: No existing files found in {download_dir}, proceeding with download...")
            else:
                expected_csv = download_path / Path(output_csv).name
                if expected_csv.exists():
                    if verbose:
                        print(f"overwrite=False: Using existing file {expected_csv}")
                    try:
                        df = pd.read_csv(expected_csv)
                        return pd.DataFrame([{
                            "year": None, "rows": len(df), "status": "ok",
                            "output_csv": str(expected_csv), "error": "",
                        }])
                    except Exception as e:
                        raise ValueError(f"Failed to read existing file {expected_csv}: {e}")
                if verbose:
                    print(f"overwrite=False: No existing file found at {expected_csv}, proceeding with download...")
        else:
            download_path.mkdir(parents=True, exist_ok=True)
            expected_csv = Path(output_csv) if Path(output_csv).is_absolute() else download_path / Path(output_csv).name
            if expected_csv.exists():
                if verbose:
                    print(f"overwrite=False: Using existing file {expected_csv}")
                try:
                    df = pd.read_csv(expected_csv)
                    inferred_year = None
                    if inject_year_in_filename:
                        match = re.search(r'_(\d{4})\.csv$', str(expected_csv))
                        if match:
                            inferred_year = int(match.group(1))
                    df.attrs["output_csv"] = str(expected_csv)
                    if return_year:
                        return df, inferred_year
                    return df
                except Exception as e:
                    raise ValueError(f"Failed to read existing file {expected_csv}: {e}")
            if verbose:
                print(f"overwrite=False: No existing file found at {expected_csv}, proceeding with download...")

    if verbose:
        print("=" * 70)
        print("STEP 1: Fetch data from IPUMS API")
        print("=" * 70)
    fetch_result = fetch_ipums_microdata_cps(
        api_key=api_key,
        years=years,
        samples=samples,
        variables=variables,
        download_dir=download_dir,
        initial_wait_time=initial_wait_time,
        max_wait_time=max_wait_time,
        timeout=timeout,
        verbose=verbose,
    )

    if fetch_result["status"] != "completed":
        raise RuntimeError(
            f"IPUMS fetch failed: {fetch_result['status']}. "
            f"Extract ID: {fetch_result['extract_id']}"
        )

    extract_files = fetch_result["files"]
    if not extract_files:
        raise ValueError(f"No files downloaded from IPUMS (ID: {fetch_result['extract_id']})")

    extract_file = extract_files[0]

    if weight_col is None:
        peek_df = _read_ipums_extract(extract_file)
        for candidate in ("ASECWT", "WTFINL", "PERWT", "EARNWT"):
            if candidate in peek_df.columns:
                weight_col = candidate
                break
        if weight_col is None:
            raise ValueError(
                "Could not auto-detect weight column. Available columns: "
                f"{list(peek_df.columns)}. Pass weight_col explicitly."
            )
        if verbose:
            print(f"  Auto-detected weight column: {weight_col}")

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: Aggregate extract into BLS-compatible CSV")
        print("=" * 70)

    if years is not None:
        output_dir = download_dir or "ipums_api_downloads"
        if Path(output_dir).exists():
            current_extract_names = {Path(f).name for f in extract_files}
            for f in Path(output_dir).glob("*.xml"):
                if f.is_file() and f.name not in current_extract_names:
                    f.unlink()
                    if verbose:
                        print(f"  Cleared old extract metadata: {f.name}")
            for f in Path(output_dir).glob("cps_*.csv"):
                if f.is_file() and f.name not in current_extract_names:
                    f.unlink()
                    if verbose:
                        print(f"  Cleared old raw extract: {f.name}")
            if inject_year_in_filename:
                base_name = Path(output_csv).stem
                pattern = f"{base_name}_[0-9]{{4}}.csv"
            else:
                pattern = Path(output_csv).name
            for f in Path(output_dir).glob(pattern if "*" in pattern else f"{pattern}*"):
                if f.is_file() and f.suffix == ".csv":
                    f.unlink()
                    if verbose:
                        print(f"  Cleared old aggregated: {f.name}")

        try:
            runs_df = aggregate_ipums_professions_csv_batch(
                extract_file=extract_file,
                output_dir=output_dir,
                years=sorted(years),
                output_basename=output_csv,
                continue_on_error=True,
                occupation_code_col=occupation_code_col,
                occupation_label_col=occupation_label_col,
                occupation_map_file=occupation_map_file,
                year_col=year_col,
                sex_col=sex_col,
                race_col=race_col,
                hispanic_col=hispanic_col,
                weight_col=weight_col,
                female_codes=female_codes,
                black_codes=black_codes,
                asian_codes=asian_codes,
                non_hispanic_codes=non_hispanic_codes,
                min_total_employed=min_total_employed,
                inject_year_in_filename=inject_year_in_filename,
            )
            if verbose:
                ok   = (runs_df["status"] == "ok").sum()
                fail = (runs_df["status"] == "failed").sum()
                print(f"\n✓ Batch complete: {ok} succeeded, {fail} failed")
        finally:
            if not keep_extract_file and Path(extract_file).exists():
                if verbose:
                    print(f"Cleaning up: {extract_file}")
                Path(extract_file).unlink()
        return runs_df

    try:
        out_df, out_year = aggregate_ipums_professions_csv(
            extract_file=extract_file,
            output_csv=output_csv,
            year=year,
            occupation_code_col=occupation_code_col,
            occupation_label_col=occupation_label_col,
            occupation_map_file=occupation_map_file,
            year_col=year_col,
            sex_col=sex_col,
            race_col=race_col,
            hispanic_col=hispanic_col,
            weight_col=weight_col,
            female_codes=female_codes,
            black_codes=black_codes,
            asian_codes=asian_codes,
            non_hispanic_codes=non_hispanic_codes,
            min_total_employed=min_total_employed,
            inject_year_in_filename=inject_year_in_filename,
            return_year=True,
        )
        if verbose:
            output_path = out_df.attrs.get("output_csv", output_csv)
            print(f"\n✓ Success! Output: {output_path}")
    finally:
        if not keep_extract_file and Path(extract_file).exists():
            if verbose:
                print(f"Cleaning up: {extract_file}")
            Path(extract_file).unlink()

    if return_year:
        return out_df, out_year
    return out_df


# ── Prestige ──────────────────────────────────────────────────────────────────

# Supported IPUMS USA prestige variables and their descriptions
PRESTIGE_VARIABLES = {
    "OCCSCORE": "Duncan SEI — occupational socioeconomic index (income + education), harmonized to OCC2010",
    "SEI":      "Nam-Powers SEI — socioeconomic index based on education and income",
    "HWSEI":    "Hauser-Warren SEI — perception-based prestige scale, 1990 occupation basis",
    "PRESGL":   "Siegel prestige score — survey-based occupational prestige",
    "PRENT":    "Nam-Powers-Boyd prestige score — assigned by occupational category",
}


def fetch_prestige_crosswalk(
    api_key: Optional[str] = None,
    prestige_variable: str = "OCCSCORE",
    sample: str = "us2019a",
    occupation_code_col: str = "OCC2010",
    download_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    initial_wait_time: float = 2,
    max_wait_time: float = 30,
    timeout: float = 600,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch a static OCC2010→prestige crosswalk from IPUMS USA (Census/ACS).

    Submits a minimal IPUMS USA extract containing only OCC2010 and the chosen
    prestige variable, then deduplicates to a unique OCC2010→score lookup table.
    Since prestige scores are fixed lookup values (not computed from ACS respondents),
    any single ACS year produces the complete crosswalk.

    Requires IPUMS USA registration at https://uma.pop.umn.edu/usa/registration/new.
    Your IPUMS CPS API key works for USA once registered.

    Available prestige variables (pass as prestige_variable):
        'OCCSCORE' — Duncan SEI, harmonized to OCC2010 (default, recommended)
        'SEI'      — Nam-Powers SEI
        'HWSEI'    — Hauser-Warren SEI, perception-based, 1990 occupation basis
        'PRESGL'   — Siegel prestige score, survey-based
        'PRENT'    — Nam-Powers-Boyd prestige score

    Args:
        api_key:            IPUMS API key (reads from IPUMS_API_KEY env var if not provided)
        prestige_variable:  IPUMS USA prestige variable to fetch (default: 'OCCSCORE')
        sample:             IPUMS USA sample ID to use (default: 'us2019a' — 2019 ACS 1-year)
        occupation_code_col: Occupation code column (default: 'OCC2010')
        download_dir:       Directory for temporary extract files (deleted after use)
        output_csv:         If provided, save the crosswalk CSV here for reuse
        initial_wait_time:  Initial seconds before polling API status
        max_wait_time:      Max seconds between polls
        timeout:            Total timeout in seconds for extract completion
        verbose:            Print progress and summary

    Returns:
        DataFrame with columns:
            occ2010   — OCC2010 occupation code (int)
            prestige  — prestige score for that code

    Example:
        >>> # Fetch once and cache
        >>> crosswalk_df = fetch_prestige_crosswalk(
        ...     api_key='...',
        ...     output_csv='/scratch/edk202/lexichron/occ2010_prestige.csv',
        ... )
        >>>
        >>> # Subsequent runs: just read from CSV
        >>> crosswalk_df = pd.read_csv('/scratch/edk202/lexichron/occ2010_prestige.csv')
    """
    if prestige_variable not in PRESTIGE_VARIABLES:
        raise ValueError(
            f"Unknown prestige_variable '{prestige_variable}'. "
            f"Choose from: {list(PRESTIGE_VARIABLES)}"
        )

    try:
        from ipumspy import IpumsApiClient, MicrodataExtract
    except ImportError:
        raise ImportError("ipumspy is required. Install with: pip install ipumspy")

    if api_key is None:
        api_key = os.getenv("IPUMS_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing IPUMS_API_KEY. Set the environment variable or pass api_key. "
            "Get your key at https://account.ipums.org/api"
        )

    if download_dir is None:
        download_dir = "ipums_api_downloads"
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    client = IpumsApiClient(
        api_key=api_key,
        base_url="https://api.ipums.org",
        api_version=2,
    )

    if verbose:
        print(f"Fetching prestige crosswalk: {prestige_variable}")
        print(f"  {PRESTIGE_VARIABLES[prestige_variable]}")
        print(f"  Sample: {sample}")

    extract = MicrodataExtract(
        collection="usa",
        samples=[sample],
        variables=[occupation_code_col, prestige_variable],
        description=f"lexichron prestige crosswalk: {prestige_variable}",
        data_format="csv",
    )

    if verbose:
        print("Submitting IPUMS USA extract...")
    submitted = client.submit_extract(extract)
    extract_id = submitted.extract_id
    if verbose:
        print(f"  Extract ID: {extract_id}")

    if verbose:
        print("Waiting for extract to complete...")
    client.wait_for_extract(
        submitted,
        collection="usa",
        inital_wait_time=initial_wait_time,
        max_wait_time=max_wait_time,
        timeout=timeout,
    )
    if verbose:
        print(f"  Status: {client.extract_status(extract_id, collection='usa')}")

    import gzip, shutil
    if verbose:
        print(f"Downloading to {download_path}...")
    client.download_extract(
        extract_id, download_dir=str(download_path), collection="usa"
    )

    # Decompress if needed — restricted to usa_*.gz only
    for gz_path in list(download_path.glob("usa_*.gz")):
        out_path = gz_path.with_suffix("")
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        if verbose:
            print(f"  Decompressed {gz_path.name} -> {out_path.name}")

    # Find the downloaded CSV by extract ID
    id_str = str(extract_id).zfill(5)
    exact_csv = download_path / f"usa_{id_str}.csv"
    if exact_csv.is_file():
        extract_file = str(exact_csv)
    else:
        csv_files = [p for p in download_path.glob("usa_*.csv") if p.is_file()]
        if not csv_files:
            raise ValueError(f"No USA extract CSV found in {download_path} after download.")
        extract_file = str(sorted(csv_files)[-1])

    # Build crosswalk: unique OCC2010 → prestige score
    raw = pd.read_csv(extract_file, usecols=[occupation_code_col, prestige_variable],
                      low_memory=False)
    raw[occupation_code_col] = pd.to_numeric(raw[occupation_code_col], errors="coerce")
    raw[prestige_variable]   = pd.to_numeric(raw[prestige_variable],   errors="coerce")

    # Prestige == 0 indicates missing/NIU in IPUMS
    raw = raw[(raw[prestige_variable] > 0) & raw[occupation_code_col].notna()]

    crosswalk = (
        raw.drop_duplicates(subset=[occupation_code_col])
        .rename(columns={occupation_code_col: "occ2010", prestige_variable: "prestige"})
        [["occ2010", "prestige"]]
        .sort_values("occ2010")
        .reset_index(drop=True)
    )
    crosswalk["occ2010"] = crosswalk["occ2010"].astype(int)

    # Clean up only the USA extract files
    Path(extract_file).unlink()
    for xml_path in download_path.glob(f"usa_{id_str}.xml"):
        xml_path.unlink()
    if verbose:
        print(f"  Cleaned up USA extract files")

    if verbose:
        print(f"\n  OCC2010 codes with prestige scores: {len(crosswalk)}")
        print(f"  Prestige range: {crosswalk['prestige'].min():.1f} – {crosswalk['prestige'].max():.1f}")

    if output_csv:
        crosswalk.to_csv(output_csv, index=False)
        if verbose:
            print(f"  Saved crosswalk to {output_csv}")

    return crosswalk


def aggregate_prestige_by_label(
    extract_file: str,
    occupation_map_file: str,
    crosswalk_df: Optional[pd.DataFrame] = None,
    crosswalk_file: Optional[str] = None,
    occupation_code_col: str = "OCC2010",
    weight_col: str = "ASECWT",
    occupation_label_col: Optional[str] = None,
    min_total_employed: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute a single static weighted-mean prestige score per generic occupation label,
    collapsed across all years in the CPS extract.

    Prestige scores are supplied via an OCC2010→prestige crosswalk produced by
    fetch_prestige_crosswalk(), rather than being read from the extract itself.
    Time-variation is not modeled — scores reflect the full-extract weighted average,
    consistent with Treiman's cross-temporal prestige stability thesis.

    Args:
        extract_file:          Path to IPUMS CPS extract (.csv/.csv.gz/.parquet/.dta)
        occupation_map_file:   CSV mapping OCC2010 codes to occupation labels
                               (must have 'code' and 'label' columns)
        crosswalk_df:          DataFrame with columns 'occ2010' and 'prestige',
                               as returned by fetch_prestige_crosswalk(). Takes
                               precedence over crosswalk_file if both are provided.
        crosswalk_file:        Path to a saved crosswalk CSV (alternative to crosswalk_df)
        occupation_code_col:   Occupation code column in extract (default: OCC2010)
        weight_col:            Person weight column (default: ASECWT)
        occupation_label_col:  Optional label column in extract (overrides map file)
        min_total_employed:    Minimum weighted employment to retain a label
        verbose:               Print progress and summary

    Returns:
        DataFrame with columns:
            label           — generic occupation label (e.g. 'engineer')
            prestige_mean   — weighted mean prestige score across all years
            prestige_sd     — weighted SD (for diagnostics)
            total_employed  — total weighted employment across all years

    Example:
        >>> crosswalk_df = fetch_prestige_crosswalk(
        ...     api_key='...',
        ...     output_csv='/scratch/edk202/lexichron/occ2010_prestige.csv',
        ... )
        >>> prestige_df = aggregate_prestige_by_label(
        ...     extract_file='/scratch/edk202/lexichron/ipums_api_downloads/cps_00037.csv',
        ...     occupation_map_file='/scratch/edk202/lexichron/ipums_occ2010_map.csv',
        ...     crosswalk_df=crosswalk_df,
        ... )
    """
    if crosswalk_df is None and crosswalk_file is None:
        raise ValueError(
            "Provide either crosswalk_df (from fetch_prestige_crosswalk()) "
            "or crosswalk_file (path to a saved crosswalk CSV)."
        )

    # Load crosswalk
    if crosswalk_df is not None:
        cw = crosswalk_df.copy()
    else:
        cw = pd.read_csv(crosswalk_file)

    missing_cols = [c for c in ["occ2010", "prestige"] if c not in cw.columns]
    if missing_cols:
        raise ValueError(
            f"Crosswalk missing required columns: {missing_cols}. "
            f"Expected 'occ2010' and 'prestige' (as produced by fetch_prestige_crosswalk())."
        )
    cw["occ2010"] = pd.to_numeric(cw["occ2010"], errors="coerce")
    prestige_lookup = cw.dropna(subset=["occ2010"]).set_index("occ2010")["prestige"].to_dict()

    if verbose:
        print(f"Reading extract: {extract_file}")
    df = _read_ipums_extract(extract_file)

    required = [occupation_code_col, weight_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Extract missing required columns: {missing}")

    if verbose:
        year_info = sorted(df["YEAR"].unique()) if "YEAR" in df.columns else "unknown"
        print(f"  Rows: {len(df):,}  |  Years: {year_info}")

    labels = _resolve_occupation_labels(
        df=df,
        occupation_code_col=occupation_code_col,
        occupation_label_col=occupation_label_col,
        occupation_map_file=occupation_map_file,
    )

    occ_codes = pd.to_numeric(df[occupation_code_col], errors="coerce")

    work = pd.DataFrame({
        "label_raw": labels,
        "prestige":  occ_codes.map(prestige_lookup),
        "weight":    pd.to_numeric(df[weight_col], errors="coerce"),
    }).dropna()

    work = work[(work["weight"] > 0) & (work["prestige"] > 0)]

    if work.empty:
        raise ValueError(
            "No valid prestige rows after joining crosswalk — "
            "check that OCC2010 codes in extract overlap with crosswalk."
        )

    n_total   = len(df)
    n_matched = len(work)
    if verbose:
        print(f"  Crosswalk coverage: {n_matched:,} / {n_total:,} rows matched "
              f"({100 * n_matched / n_total:.1f}%)")

    # Tokenize raw labels to generic labels using the same logic as the main pipeline
    work["label"] = work["label_raw"].map(
        lambda x: _tokenize_occupation(str(x), max_tokens=5)[0]
    )
    work = work[work["label"].str.strip() != ""]

    def weighted_stats(grp):
        w   = grp["weight"].values
        x   = grp["prestige"].values
        tot = w.sum()
        if tot == 0:
            return pd.Series({"prestige_mean": np.nan, "prestige_sd": np.nan, "total_employed": 0.0})
        mean = np.average(x, weights=w)
        var  = np.average((x - mean) ** 2, weights=w)
        return pd.Series({
            "prestige_mean":   mean,
            "prestige_sd":     np.sqrt(var),
            "total_employed":  tot,
        })

    result = (
        work.groupby("label")
        .apply(weighted_stats)
        .reset_index()
    )

    result = result[result["total_employed"] >= min_total_employed].copy()
    result = result.sort_values("prestige_mean", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n  Labels with prestige scores: {len(result)}")
        print(f"  Prestige range: {result['prestige_mean'].min():.1f} – {result['prestige_mean'].max():.1f}")
        print(f"\n  Top 10 by prestige:")
        for _, row in result.head(10).iterrows():
            print(f"    {row['label']:<20} {row['prestige_mean']:6.1f}  (n={row['total_employed']:>12,.0f})")
        print(f"\n  Bottom 10 by prestige:")
        for _, row in result.tail(10).iterrows():
            print(f"    {row['label']:<20} {row['prestige_mean']:6.1f}  (n={row['total_employed']:>12,.0f})")

    return result[["label", "prestige_mean", "prestige_sd", "total_employed"]]


def add_prestige_to_panel(
    panel_df: pd.DataFrame,
    prestige_df: pd.DataFrame,
    profession_col: str = "profession",
    prestige_col: str = "prestige_mean",
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Join static prestige scores onto a panel DataFrame and optionally z-score them.

    Args:
        panel_df:       Panel DataFrame with a profession column
        prestige_df:    Output of aggregate_prestige_by_label()
        profession_col: Column in panel_df containing generic occupation labels
        prestige_col:   Prestige column in prestige_df to join (default: prestige_mean)
        standardize:    If True, add prestige_z (z-scored across labels)

    Returns:
        Copy of panel_df with prestige_mean (and prestige_z if standardize=True) added.
        Warns if any labels are missing prestige scores.

    Example:
        >>> panel_df = add_prestige_to_panel(panel_df, prestige_df)
        >>> # prestige_z is now available as a moderator alongside baseline_proj_z
    """
    lookup = prestige_df.set_index("label")[prestige_col].to_dict()

    out = panel_df.copy()
    out["prestige"] = out[profession_col].map(lookup)

    n_missing = out["prestige"].isna().sum()
    if n_missing > 0:
        missing_labels = sorted(out.loc[out["prestige"].isna(), profession_col].unique())
        print(
            f"  Warning: {n_missing} rows missing prestige scores "
            f"({len(missing_labels)} labels: {missing_labels})"
        )

    if standardize:
        mean = out["prestige"].mean()
        sd   = out["prestige"].std()
        out["prestige_z"] = (out["prestige"] - mean) / sd

    return out


# ── CSV-level helpers ─────────────────────────────────────────────────────────

def calculate_women_percentage(csv_path, label):
    """
    Compute the percentage of female workers for a given label in an IPUMS CSV.
    Sums total employed for all matching rows, computes weighted mean women%.

    Args:
        csv_path: Path to IPUMS profession CSV (BLS-compatible format)
        label: Occupation label (e.g., 'engineer')

    Returns:
        women_pct: Weighted mean percentage of female workers for the label
                   as a proportion (0–1)
    """
    df = pd.read_csv(csv_path)
    label_cols = [f'label{i}' for i in range(1, 6)]
    mask = df[label_cols].apply(lambda row: label in row.values, axis=1)
    matched = df[mask]
    if matched.empty:
        raise ZeroDivisionError(f"No rows found for label '{label}' in {csv_path}")

    total_employed = matched['TotalEmployed'].sum()
    if total_employed == 0:
        raise ZeroDivisionError(f"Total employed is zero for label '{label}' in {csv_path}")

    women_weighted = (matched['Women'] * matched['TotalEmployed']).sum() / total_employed
    return women_weighted / 100.0  # Convert percent to proportion