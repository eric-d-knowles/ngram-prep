"""
IPUMS CPS fetch + aggregation for the occupational-feminization pipeline (v2).

v2 (July 2026): counts-only schema, no v1 compatibility.
  - Processed CSVs carry exact counts (weighted + unweighted, women + total)
    and ethnicity numerator/known-denominator pairs. No percentage columns,
    no rounding; proportions are computed downstream at full precision.
  - Universe filter: EMPSTAT in (10, 12) [currently employed], AGE >= 16,
    applied when the columns exist in the extract (loud warning otherwise).
  - Era-aware ethnicity: Hispanic known = HISPAN present and < 900 (so
    pre-1971 files show a zero known denominator, not a fabricated value);
    Asian identifiable only from 1988 (RACE 650, split 651/652 in 2003);
    Black = RACE 200, valid all years. Multiracial 8xx excluded by default.
  - CSV helpers are v2-only and raise on old-schema files.
  - Default fetch variables include EMPSTAT and AGE.

Workflows:
1. **Raw aggregation**: Convert already-downloaded IPUMS extracts
   (CSV/Parquet/Stata) into profession CSVs via
   `aggregate_ipums_professions_csv()` (single year) or
   `aggregate_ipums_professions_csv_batch()` (one CSV per year).
2. **Web fetch + aggregation**: Retrieve data from the IPUMS API, download
   extracts, and aggregate them in one step via
   `fetch_and_aggregate_ipums_professions_csv()`.
3. **Static prestige scores**: `fetch_prestige_crosswalk()` /
   `aggregate_prestige_by_label()` / `add_prestige_to_panel()` (unchanged).

The IPUMS API requires `ipumspy` (pip install ipumspy) and an IPUMS API key
(set the IPUMS_API_KEY environment variable; get a key at
https://account.ipums.org/api).
"""

from pathlib import Path
import csv
import multiprocessing as mp
import os
import re
import tempfile
import time
import warnings
from typing import Optional, List, Dict, Any, Iterable, Union

import numpy as np
import pandas as pd


# ── Schema ────────────────────────────────────────────────────────────────────

TARGET_COLUMNS = [
    "Occupation",          # raw OCC2010 title (auditable union matching)
    "TotalEmployed",       # sum of person weights (universe-filtered)
    "WeightedWomen",       # sum of person weights, SEX == female
    "UnweightedN",         # respondent rows in the cell
    "UnweightedWomen",     # respondent rows, SEX == female
    "BlackWeighted",       # weighted count, RACE Black, over known denominator
    "BlackKnownWeighted",  # weighted denominator where Black is identifiable
    "AsianWeighted",
    "AsianKnownWeighted",
    "HispanicWeighted",
    "HispanicKnownWeighted",
    "label1", "label2", "label3", "label4", "label5",
]

HISPAN_AVAILABLE_FROM = 1971    # HISPAN blank before this year (informational)
ASIAN_IDENTIFIABLE_FROM = 1988  # RACE 650 (Asian/PI) introduced; 651/652 split 2003

# IPUMS CPS RACE: 100 White | 200 Black | 300 AmIndian | 650 Asian/PI (1988-2002)
# | 651 Asian only | 652 PI only (2003+) | 700 Other | 801-820 multiracial (2003+)
# IPUMS CPS HISPAN: 0 not Hispanic | 100-612 Hispanic origins | 901/902 missing/NIU

DEFAULT_FETCH_VARIABLES = [
    "YEAR", "SEX", "RACE", "HISPAN", "OCC2010", "ASECWT", "EMPSTAT", "AGE",
]

LABEL_STOPWORDS = {
    "and", "or", "of", "the", "a", "an", "for", "to", "in", "on", "at", "by", "with",
    "except", "including", "all", "other", "miscellaneous", "related", "non", "total",
    "first", "second", "third", "line", "years", "over", "percent", "employed",
}

GENERIC_ROLE_WORDS = {
    "accountant", "actuary", "actor", "adjuster", "administrator", "advisor", "agent", "aide", "analyst",
    "announcer", "appraiser", "architect", "archivist", "assembler", "assessor", "assistant", "astronomer",
    "athlete", "attendant", "audiologist", "auditor",
    "bailiff", "baker", "barber", "bartender", "batchmaker", "biologist", "blockmason", "boilermaker",
    "brickmason", "broker", "builder", "butcher", "buyer",
    "cabinetmaker", "caretaker", "carpenter", "carrier", "cartographer", "cashier", "chef", "chemist",
    "chiropractor", "choreographer", "clergy", "cleaner", "clerk", "collector", "concierge", "conductor",
    "cook", "correspondent", "counselor", "courier", "curator",
    "dancer", "demonstrator", "dentist", "designer", "detective", "developer", "director", "dishwasher",
    "dispatcher", "diver", "doctor", "drafter", "dressmaker", "driller", "driver",
    "economist", "editor", "educator", "electrician", "engraver", "engineer", "erector", "estimator",
    "examiner", "executive",
    "fabricator", "finisher", "firefighter", "fitter", "fundraiser",
    "geoscientist", "glazier", "grader", "guard", "guide",
    "hairdresser", "handler", "helper", "host", "hostess", "hygienist",
    "inspector", "installer", "instructor", "interviewer", "investigator",
    "jailer", "janitor", "judge",
    "keyer",
    "laborer", "lawyer", "legislator", "librarian", "lifeguard", "logistician",
    "machinist", "maid", "maker", "manager", "mason", "mathematician", "mechanic", "messenger",
    "millwright", "modeler", "molder", "mover", "musician",
    "nurse", "nutritionist",
    "officer", "operator", "optician", "optometrist",
    "packer", "packager", "painter", "paramedic", "paralegal", "pathologist", "pharmacist", "phlebotomist",
    "photographer", "physicist", "physiologist", "pilot", "pipelayer", "pipefitter", "planner", "plumber",
    "podiatrist", "porter", "postmaster", "practitioner", "preparer", "presser", "president", "producer",
    "programmer", "projectionist", "promoter", "proofreader", "processor", "psychologist",
    "receptionist", "repairer", "reporter", "representative", "rigger", "roofer", "roustabout",
    "salesperson", "sampler", "scientist", "screener", "secretary", "server", "setter", "shaper", "singer",
    "sorter", "specialist", "statistician", "steamfitter", "stonemason", "superintendent", "supervisor",
    "surveyor", "surgeon",
    "tailor", "taper", "teacher", "technician", "technologist", "telemarketer", "teller", "tender",
    "therapist", "trainer", "transcriptionist", "typist",
    "underwriter", "upholsterer",
    "vendor", "veterinarian",
    "waiter", "waitress", "warden", "weigher", "woodworker", "worker", "writer",
    "yardmaster",
}


# ── Label tokenization helpers ────────────────────────────────────────────────

def _is_generic_role_token(token):
    """Check if a token is a generic occupational role word."""
    if token in GENERIC_ROLE_WORDS:
        return True
    for role in GENERIC_ROLE_WORDS:
        prefix_len = len(token) - len(role)
        if len(role) >= 4 and prefix_len >= 5 and token.endswith(role):
            return True
    return False


def _normalize_label_token(token):
    """Normalize an occupation label token (handle plurals, etc.)."""
    token = token.lower().strip()
    if token.endswith("men") and len(token) > 5:
        token = token[:-3] + "man"
    if token.endswith("ies") and len(token) > 4:
        candidate = token[:-3] + "y"
        if candidate in GENERIC_ROLE_WORDS:
            return candidate
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
    timeout: float = 3600,
    description: str = "lexichron IPUMS CPS extract",
    data_format: str = "csv",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fetch and download IPUMS CPS ASEC microdata via the IPUMS API.

    Args:
        api_key: IPUMS API key (reads from IPUMS_API_KEY env var if not provided)
        years: ASEC years to download - accepts a list, range, or any iterable of ints.
               Examples: [2015, 2020, 2024] or range(2010, 2026).
               Each year is resolved to its ASEC sample ID automatically.
               If None and samples is None, uses the most recent ASEC sample.
               Cannot be used together with `samples`.
        samples: List of ASEC sample identifiers (e.g., ['cps2024_03s']).
                 If None and years is None, uses the most recent ASEC sample.
                 Only ASEC samples are supported (harmonized occupation codes + person weights).
                 Cannot be used together with `years`.
        variables: List of variable names to extract. If None, uses
                   DEFAULT_FETCH_VARIABLES:
                   ['YEAR', 'SEX', 'RACE', 'HISPAN', 'OCC2010', 'ASECWT',
                    'EMPSTAT', 'AGE']
                   OCC2010 provides harmonized detailed occupation codes across
                   years; EMPSTAT/AGE feed the aggregation universe filter.
                   Add 'OCCSCORE' to include prestige scores for use with
                   aggregate_prestige_by_label().
        download_dir: Directory to save downloaded files. Uses ipums_api_downloads/ in cwd if not specified.
        initial_wait_time: Initial seconds to wait before polling API status (default: 2)
        max_wait_time: Max seconds between polls (default: 30)
        timeout: Max total seconds to wait for extract completion (default: 3600 —
                 multi-decade, multi-variable extracts can take IPUMS well over
                 the old 600s default to build)
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
        >>> result = fetch_ipums_microdata_cps(years=range(1968, 2026))
        >>> print(f"Downloaded to {result['download_dir']}")
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
                warnings.warn(
                    f"Sample '{s}' does not look like an ASEC March supplement. "
                    f"ASEC samples typically have '_03' in the ID (e.g. 'cps2025_03s'). "
                    f"Proceeding anyway.",
                    stacklevel=2,
                )

    if variables is None:
        variables = list(DEFAULT_FETCH_VARIABLES)
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
    occupation_code_col="OCC2010",
    occupation_label_col=None,
    occupation_map_file=None,
    year_col="YEAR",
    sex_col="SEX",
    race_col="RACE",
    hispanic_col="HISPAN",
    weight_col="ASECWT",
    female_codes=(2,),
    black_codes=(200,),
    asian_codes=(650, 651),
    hispanic_range=(100, 612),
    include_multiracial=False,
    empstat_col="EMPSTAT",
    employed_codes=(10, 12),
    age_col="AGE",
    min_age=16,
    min_unweighted_n=0,
    inject_year_in_filename=True,
    return_year=False,
    verbose=True,
):
    """
    Aggregate IPUMS person-level microdata into a counts-only profession CSV.

    Output schema (TARGET_COLUMNS): raw Occupation title; TotalEmployed
    (weighted) / WeightedWomen / UnweightedN / UnweightedWomen; ethnicity as
    numerator/known-denominator weighted pairs; label1-label5 tokens. No
    percentage columns — proportions are computed downstream from exact counts.

    Universe: rows with EMPSTAT in employed_codes and AGE >= min_age, when
    those columns exist in the extract; each missing column triggers a loud
    warning and the filter is skipped (an old extract is never silently
    reinterpreted).

    Ethnicity era logic: Hispanic known = HISPAN present and < 900 (pre-1971
    blanks fall out of the denominator); Asian known only from
    ASIAN_IDENTIFIABLE_FROM (coded into 700 Other before); Black known
    whenever RACE is present (code 200 valid all years). Multiracial 8xx codes
    are excluded by default; include_multiracial=True adds
    alone-or-in-combination code sets — verify those against the IPUMS CPS
    RACE codebook before first serious use.

    min_unweighted_n (default 0): panel units are UNIONS of rows matched via
    label1-label5, so dropping a thin row here silently changes union
    composition downstream. The intended design is to keep every row, carry
    the counts, and mask at panel level. Set > 0 only with that consequence
    in mind.

    Args:
        extract_file: Path to IPUMS extract (.csv/.csv.gz/.parquet/.dta)
        output_csv: Output CSV path
        year: Optional year filter (single year)
        occupation_code_col: Occupation code column (default: OCC2010, harmonized)
        occupation_label_col: Occupation label text column (if present)
        occupation_map_file: CSV mapping occupation codes to labels
        year_col, sex_col, race_col, hispanic_col, weight_col: column names
        female_codes: Values treated as female in sex_col (IPUMS SEX: 2)
        black_codes: RACE values treated as Black (IPUMS: 200)
        asian_codes: RACE values treated as Asian (IPUMS: 650 for 1988-2002,
                     651 for 2003+)
        hispanic_range: (lo, hi) inclusive HISPAN range counted as Hispanic
        include_multiracial: If True, add 2003+ alone-or-in-combination codes
        empstat_col, employed_codes: universe filter (currently employed)
        age_col, min_age: universe age floor
        min_unweighted_n: minimum respondent rows to keep an occupation row
        inject_year_in_filename: Inject year in output filename when available
        return_year: If True, returns (DataFrame, year)
        verbose: Print the per-year audit trail

    Returns:
        DataFrame in the TARGET_COLUMNS schema.
    """
    df = _read_ipums_extract(extract_file)

    for col in (occupation_code_col, sex_col, weight_col):
        if col not in df.columns:
            raise ValueError(f"extract_file missing required column: {col}")

    if year is not None:
        if year_col not in df.columns:
            raise ValueError(f"year filter requested but `{year_col}` not found in extract")
        df = df[df[year_col] == year].copy()

    if df.empty:
        raise ValueError("No rows available after filtering extract")

    n_start = len(df)

    # ── Universe filter (EMPSTAT / AGE) ──────────────────────────────────
    if empstat_col:
        if empstat_col in df.columns:
            emp = pd.to_numeric(df[empstat_col], errors="coerce")
            df = df[emp.isin(employed_codes)].copy()
        else:
            warnings.warn(
                f"Universe filter requested (empstat_col={empstat_col!r}) but the "
                f"column is absent from this extract — proceeding UNFILTERED. "
                f"Output is not comparable to EMPSTAT-filtered runs.",
                UserWarning,
            )
    n_after_emp = len(df)

    if age_col and min_age:
        if age_col in df.columns:
            age = pd.to_numeric(df[age_col], errors="coerce")
            df = df[age >= min_age].copy()
        else:
            warnings.warn(
                f"Age floor requested (age_col={age_col!r}, min_age={min_age}) "
                f"but the column is absent — proceeding without it.",
                UserWarning,
            )
    n_after_age = len(df)

    if df.empty:
        raise ValueError("No rows available after universe filtering")

    # ── Labels ───────────────────────────────────────────────────────────
    labels = _resolve_occupation_labels(
        df=df,
        occupation_code_col=occupation_code_col,
        occupation_label_col=occupation_label_col,
        occupation_map_file=occupation_map_file,
    )

    # ── Demographic flags (era-aware, known-denominator) ─────────────────
    weight = pd.to_numeric(df[weight_col], errors="coerce")
    sex = pd.to_numeric(df[sex_col], errors="coerce")
    is_woman = sex.isin(female_codes)

    year_ser = (
        pd.to_numeric(df[year_col], errors="coerce")
        if year_col in df.columns else pd.Series(np.nan, index=df.index)
    )

    if race_col in df.columns:
        race = pd.to_numeric(df[race_col], errors="coerce")
        race_known = race.notna()

        black_set = set(black_codes)
        asian_set = set(asian_codes)
        if include_multiracial:
            # Alone-or-in-combination (2003+ codes). VERIFY against the
            # IPUMS CPS RACE codebook before first serious use.
            black_set |= {801, 805, 806, 807, 810, 811, 814, 816, 818}
            asian_set |= {652, 803, 806, 808, 810, 812, 813, 814, 815, 816, 817, 818, 819}

        is_black = race.isin(black_set)
        black_known = race_known  # code 200 valid in all years
        is_asian = race.isin(asian_set)
        asian_known = race_known & (year_ser >= ASIAN_IDENTIFIABLE_FROM)
    else:
        is_black = is_asian = pd.Series(False, index=df.index)
        black_known = asian_known = pd.Series(False, index=df.index)

    if hispanic_col in df.columns:
        hisp = pd.to_numeric(df[hispanic_col], errors="coerce")
        hisp_known = hisp.notna() & (hisp < 900)   # pre-1971 blanks fall out
        lo, hi = hispanic_range
        is_hispanic = hisp.between(lo, hi)
    else:
        is_hispanic = pd.Series(False, index=df.index)
        hisp_known = pd.Series(False, index=df.index)

    work = pd.DataFrame({
        "Occupation": labels,
        "weight": weight,
        "is_woman": is_woman,
        "black_w":  weight.where(is_black & black_known, 0.0),
        "black_dw": weight.where(black_known, 0.0),
        "asian_w":  weight.where(is_asian & asian_known, 0.0),
        "asian_dw": weight.where(asian_known, 0.0),
        "hisp_w":   weight.where(is_hispanic & hisp_known, 0.0),
        "hisp_dw":  weight.where(hisp_known, 0.0),
    })

    work = work.dropna(subset=["Occupation", "weight"]).copy()
    work["Occupation"] = work["Occupation"].astype(str).str.strip()
    n_nonpos_weight = int((work["weight"] <= 0).sum())
    work = work[(work["Occupation"] != "") & (work["weight"] > 0)]

    if work.empty:
        raise ValueError("No valid weighted occupation rows found in extract")

    work["women_weight"] = work["weight"].where(work["is_woman"], 0.0)
    work["women_row"] = work["is_woman"].astype(int)

    grouped = work.groupby("Occupation", dropna=False, as_index=False).agg(
        TotalEmployed=("weight", "sum"),
        WeightedWomen=("women_weight", "sum"),
        UnweightedN=("weight", "size"),
        UnweightedWomen=("women_row", "sum"),
        BlackWeighted=("black_w", "sum"),
        BlackKnownWeighted=("black_dw", "sum"),
        AsianWeighted=("asian_w", "sum"),
        AsianKnownWeighted=("asian_dw", "sum"),
        HispanicWeighted=("hisp_w", "sum"),
        HispanicKnownWeighted=("hisp_dw", "sum"),
    )

    # ── Cell filter (respondent counts; default off — see docstring) ─────
    n_dropped_unw = 0
    if min_unweighted_n and min_unweighted_n > 0:
        before = len(grouped)
        grouped = grouped[grouped["UnweightedN"] >= int(min_unweighted_n)]
        n_dropped_unw = before - len(grouped)
    if grouped.empty:
        raise ValueError("No occupation rows meet min_unweighted_n")

    grouped = grouped.sort_values("Occupation").reset_index(drop=True)
    grouped["UnweightedN"] = grouped["UnweightedN"].astype(int)
    grouped["UnweightedWomen"] = grouped["UnweightedWomen"].astype(int)

    label_columns = ["label1", "label2", "label3", "label4", "label5"]
    label_df = pd.DataFrame(
        grouped["Occupation"].map(lambda x: _tokenize_occupation(x, max_tokens=5)).tolist(),
        columns=label_columns,
        index=grouped.index,
    )
    out = pd.concat([grouped, label_df], axis=1)[TARGET_COLUMNS]

    # ── Year detection + write ───────────────────────────────────────────
    detected_year = None
    if year is not None:
        detected_year = int(year)
    elif year_col in df.columns:
        year_values = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
        if len(year_values.unique()) == 1:
            detected_year = int(year_values.iloc[0])

    resolved_output_csv = _output_path_with_year(
        output_csv, detected_year, inject_year_in_filename=inject_year_in_filename,
    )

    weighted_cols = {
        "TotalEmployed", "WeightedWomen",
        "BlackWeighted", "BlackKnownWeighted",
        "AsianWeighted", "AsianKnownWeighted",
        "HispanicWeighted", "HispanicKnownWeighted",
    }
    with open(resolved_output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TARGET_COLUMNS)
        for row in out.itertuples(index=False):
            rec = []
            for col, val in zip(TARGET_COLUMNS, row):
                if col in ("UnweightedN", "UnweightedWomen"):
                    rec.append(int(val))
                elif col in weighted_cols:
                    rec.append(f"{float(val):.2f}")   # weights carry 2 decimals
                else:
                    rec.append(val)
            writer.writerow(rec)

    if verbose:
        print(f"Saved {len(out)} rows to {resolved_output_csv}")
        if detected_year is not None:
            print(f"Detected IPUMS reference year: {detected_year}")
        print(
            f"  Universe: {n_start:,} rows -> {n_after_emp:,} after EMPSTAT "
            f"-> {n_after_age:,} after AGE; {n_nonpos_weight} non-positive-weight "
            f"rows dropped"
        )
        if n_dropped_unw:
            print(f"  Cell filter: {n_dropped_unw} rows below "
                  f"min_unweighted_n={min_unweighted_n}")
        tot_w = float(out["TotalEmployed"].astype(float).sum())
        for name, dw in (("Black", "BlackKnownWeighted"),
                         ("Asian", "AsianKnownWeighted"),
                         ("Hispanic", "HispanicKnownWeighted")):
            cov = 100.0 * float(out[dw].astype(float).sum()) / tot_w if tot_w else 0.0
            print(f"  {name}: known-denominator coverage {cov:.1f}%"
                  + ("  (numerators are zero-information this year)" if cov == 0.0 else ""))

    result = out.copy()
    result.attrs["ipums_reference_year"] = detected_year
    result.attrs["output_csv"] = resolved_output_csv

    if return_year:
        return result, detected_year
    return result


def _aggregate_one_year(args):
    """
    Worker function: aggregate a single year-slice DataFrame into a profession CSV.

    Receives a pre-sliced DataFrame rather than re-reading the full extract file,
    avoiding the cost of 58+ full file reads in a batch run. The slice is written
    to a temporary parquet file so aggregate_ipums_professions_csv can use the
    existing _read_ipums_extract path without modification.
    """
    year, year_df, output_csv, kwargs = args

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        year_df.to_parquet(tmp_path, index=False)
        out_df, out_year = aggregate_ipums_professions_csv(
            extract_file=tmp_path,
            output_csv=output_csv,
            year=None,  # already filtered to one year — do not filter again
            return_year=True,
            **kwargs,
        )
        return {
            "year":       out_year,
            "rows":       len(out_df),
            "status":     "ok",
            "output_csv": out_df.attrs.get("output_csv", ""),
            "error":      "",
        }
    except Exception as exc:
        return {
            "year":       year,
            "rows":       None,
            "status":     "failed",
            "output_csv": "",
            "error":      str(exc),
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def aggregate_ipums_professions_csv_batch(
    extract_file,
    output_dir,
    years=None,
    output_basename="professionsIPUMS.csv",
    continue_on_error=True,
    num_workers=None,
    **kwargs,
):
    """
    Export one counts-schema CSV per year from an IPUMS extract.

    Reads the extract once, splits by year in memory, then processes each
    year-slice in parallel via a spawn-context multiprocessing pool.

    Args:
        extract_file: Path to IPUMS extract (.csv/.csv.gz/.parquet/.dta)
        output_dir: Directory to write per-year CSVs
        years: Years to process. If None, all years found in the extract are used.
        output_basename: Base filename for per-year CSVs (year is injected before extension)
        continue_on_error: If True, failed years are recorded but do not halt the batch.
                           If False, raises RuntimeError on the first failure.
        num_workers: Number of parallel workers. Defaults to SLURM_CPUS_PER_TASK if
                     available, otherwise os.cpu_count() capped at number of years.
        **kwargs: Forwarded to aggregate_ipums_professions_csv (occupation_code_col,
                  weight_col, empstat_col, min_age, etc.)

    Returns:
        DataFrame with columns: year, rows, status, output_csv, error.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Read the extract once and split by year in memory — avoids re-reading
    # the full file on every iteration of the original sequential loop.
    df = _read_ipums_extract(extract_file)
    year_col = kwargs.get("year_col", "YEAR")

    if years is None:
        if year_col not in df.columns:
            raise ValueError("years=None requires a year column in extract")
        year_values = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
        years = sorted(year_values.unique().tolist())

    if year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        year_slices = {
            year: df[df[year_col] == year].copy()
            for year in years
        }
    else:
        # No year column — every year gets the full DataFrame (single-year extract)
        year_slices = {year: df.copy() for year in years}

    output_csv = str(output_dir_path / output_basename)

    tasks = [
        (year, year_slices.get(year, pd.DataFrame()), output_csv, kwargs)
        for year in years
    ]

    if num_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        num_workers = int(slurm_cpus) if slurm_cpus else min(mp.cpu_count(), len(tasks))

    # Use spawn context to avoid fork-inherited lock deadlocks with numpy/pandas.
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_aggregate_one_year, tasks)

    if not continue_on_error:
        for r in results:
            if r["status"] == "failed":
                raise RuntimeError(
                    f"Aggregation failed for year {r['year']}: {r['error']}"
                )

    runs_df = pd.DataFrame(results)
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
    black_codes: tuple = (200,),
    asian_codes: tuple = (650, 651),
    hispanic_range: tuple = (100, 612),
    include_multiracial: bool = False,
    empstat_col: str = "EMPSTAT",
    employed_codes: tuple = (10, 12),
    age_col: str = "AGE",
    min_age: int = 16,
    min_unweighted_n: int = 0,
    inject_year_in_filename: bool = True,
    initial_wait_time: float = 2,
    max_wait_time: float = 30,
    timeout: float = 3600,
    return_year: bool = False,
    overwrite: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    Fetch CPS ASEC data from IPUMS API and aggregate into counts-schema
    profession CSVs (v2). Only ASEC samples are supported (yearly data with
    harmonized occupation codes and person weights).

    Combines fetch_ipums_microdata_cps() and aggregate_ipums_professions_csv()
    (or the batch variant when `years` is given). All v2 aggregation
    parameters (universe filter, ethnicity ranges, min_unweighted_n) are
    forwarded.

    Args mirror the two underlying functions; see their docstrings. Notable:
        weight_col: If None, auto-detects from fetched data (ASECWT for ASEC).
        timeout: Extract build+download timeout in seconds (default 3600).
        overwrite: If False, skip download and use existing files in
                   download_dir. In multi-year mode, returns results for years
                   with existing aggregated CSVs.

    Returns:
        When years is provided: DataFrame of batch results
        (year, rows, status, output_csv, error). Otherwise: DataFrame in the
        TARGET_COLUMNS schema (plus (df, year) if return_year=True).
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

    agg_kwargs = dict(
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
        hispanic_range=hispanic_range,
        include_multiracial=include_multiracial,
        empstat_col=empstat_col,
        employed_codes=employed_codes,
        age_col=age_col,
        min_age=min_age,
        min_unweighted_n=min_unweighted_n,
        inject_year_in_filename=inject_year_in_filename,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: Aggregate extract into counts-schema CSV")
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
                **agg_kwargs,
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
            return_year=True,
            **agg_kwargs,
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
    overwrite: bool = False,
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
        ...     output_csv='/scratch/edk202/lexichron/occ2010_prestige.csv',
        ... )
        >>>
        >>> # Subsequent runs: just read from CSV
        >>> crosswalk_df = pd.read_csv('/scratch/edk202/lexichron/occ2010_prestige.csv')
    """
    # If output_csv exists and overwrite is False, load and return
    if output_csv and not overwrite and os.path.exists(output_csv):
        if verbose:
            print(f"Loading cached prestige crosswalk from {output_csv}")
        return pd.read_csv(output_csv)

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
    output_csv: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    # If output_csv exists and overwrite is False, load and return
    if output_csv and not overwrite and os.path.exists(output_csv):
        if verbose:
            print(f"Loading cached aggregated prestige data from {output_csv}")
        return pd.read_csv(output_csv)
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

    out_df = result[["label", "prestige_mean", "prestige_sd", "total_employed"]]
    if output_csv:
        out_df.to_csv(output_csv, index=False)
        if verbose:
            print(f"  Saved aggregated prestige data to {output_csv}")
    return out_df


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
    """
    prestige_merge = prestige_df[["label", prestige_col]].rename(columns={"label": profession_col})
    out = panel_df.merge(prestige_merge, on=profession_col, how="left")
    out.rename(columns={prestige_col: "prestige"}, inplace=True)

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


# ── CSV-level helpers (v2 files only) ─────────────────────────────────────────

def _matched_rows(csv_path, label):
    """Read a v2 profession CSV and return the rows matching a label (union)."""
    df = pd.read_csv(csv_path)
    if "WeightedWomen" not in df.columns or "UnweightedN" not in df.columns:
        raise ValueError(
            f"{csv_path} is not a v2 counts-schema file — regenerate with the "
            f"v2 aggregator (v1 files are no longer supported)."
        )
    label_cols = [f"label{i}" for i in range(1, 6)]
    mask = df[label_cols].apply(lambda row: label in row.values, axis=1)
    matched = df[mask]
    if matched.empty:
        raise ZeroDivisionError(f"No rows found for label '{label}' in {csv_path}")
    return matched


def calculate_women_counts(csv_path, label, granularity=None):
    """
    Weighted (women, total) counts for a label — exact, union over label1-5.

    Args:
        csv_path: Path to a v2 IPUMS profession CSV
        label: Occupation label (e.g., 'engineer')
        granularity: Ignored; accepted for API compatibility with build_panel.

    Returns:
        (women_employed, total_employed): survey-weighted counts (floats).
    """
    matched = _matched_rows(csv_path, label)
    total = float(matched["TotalEmployed"].sum())
    if total == 0:
        raise ZeroDivisionError(f"Total employed is zero for label '{label}' in {csv_path}")
    return float(matched["WeightedWomen"].sum()), total


def calculate_women_counts_unweighted(csv_path, label, granularity=None):
    """
    Unweighted respondent (women, n) counts for a label — the honest x and n
    for the empirical logit log((x+0.5)/(n-x+0.5)) and binomial
    sampling-variance benchmarks.

    Args:
        csv_path: Path to a v2 IPUMS profession CSV
        label: Occupation label (e.g., 'engineer')
        granularity: Ignored; accepted for API compatibility with build_panel.

    Returns:
        (women_respondents, n_respondents): raw CPS respondent counts (ints).
    """
    matched = _matched_rows(csv_path, label)
    n = int(matched["UnweightedN"].sum())
    if n == 0:
        raise ZeroDivisionError(f"Unweighted N is zero for label '{label}' in {csv_path}")
    return int(matched["UnweightedWomen"].sum()), n


def calculate_women_proportion(csv_path, label, granularity=None):
    """
    Weighted proportion (0-1) of female workers for a label in a v2 CSV.

    Args:
        csv_path: Path to a v2 IPUMS profession CSV
        label: Occupation label (e.g., 'engineer')
        granularity: Ignored; accepted for API compatibility with build_panel.

    Returns:
        women_prop: Weighted proportion of female workers for the label (0-1)
    """
    women_employed, total_employed = calculate_women_counts(
        csv_path, label, granularity=granularity)
    return women_employed / total_employed


# ── Group/union count helpers (build_panel v3) ───────────────────────────────

_LABEL_COLS = [f"label{i}" for i in range(1, 6)]
_YEAR_COUNT_COLUMNS = [
    "profession", "women_n", "total_n", "women_n_unw", "total_n_unw",
]


def _read_v2_profession_csv(csv_path):
    """Read a v2 counts-schema profession CSV; raise on any other vintage."""
    df = pd.read_csv(csv_path)
    if "WeightedWomen" not in df.columns or "UnweightedN" not in df.columns:
        raise ValueError(
            f"{csv_path} is not a v2 counts-schema file — regenerate with the "
            f"v2 aggregator (v1 files are no longer supported)."
        )
    return df


def _label_union_mask(df, labels):
    """Boolean mask: rows whose label1-label5 contain ANY of `labels`.

    Row-level selection, so a row matching several labels in the set is
    included exactly once — this is what makes group counts duplication-
    safe without any seam logic.
    """
    return df[_LABEL_COLS].isin(set(labels)).any(axis=1)


def calculate_year_counts(csv_path, units):
    """
    All four counts for every panel unit, from ONE read of a year's CSV.

    Args:
        csv_path: Path to a v2 IPUMS profession CSV (one year).
        units: dict mapping unit name -> list of labels that define it.
               Ordinary units are one-label lists ({"nurse": ["nurse"]});
               collapse canonicals list their components
               ({"examiner": ["appraiser", "examiner", "investigator"]}).

    Returns:
        DataFrame with one row per unit that matched at least one CSV row
        with positive weighted total, columns:
            profession   — unit name
            women_n      — weighted women count (float; exact)
            total_n      — weighted total (float; exact)
            women_n_unw  — unweighted women respondents (int)
            total_n_unw  — unweighted respondents (int)
        Units with no matching rows are simply absent (caller skips them).

    Each unit's counts sum DISTINCT matched rows once (see
    _label_union_mask), so shared-source years and disjoint-split years
    are both handled correctly for ratios AND for respondent counts.
    """
    df = _read_v2_profession_csv(csv_path)
    rows = []
    for name, labels in units.items():
        sub = df[_label_union_mask(df, labels)]
        if sub.empty:
            continue
        total_w = float(sub["TotalEmployed"].sum())
        if total_w <= 0:
            continue
        rows.append({
            "profession": name,
            "women_n": float(sub["WeightedWomen"].sum()),
            "total_n": total_w,
            "women_n_unw": int(sub["UnweightedWomen"].sum()),
            "total_n_unw": int(sub["UnweightedN"].sum()),
        })
    return pd.DataFrame(rows, columns=_YEAR_COUNT_COLUMNS)


def calculate_group_counts(csv_path, labels):
    """
    Duplication-safe union counts for one group of labels in one year.

    Thin wrapper over calculate_year_counts for one-off/interactive use.

    Returns:
        (women_n, total_n, women_n_unw, total_n_unw)

    Raises:
        ZeroDivisionError if no rows match or the weighted total is zero.
    """
    out = calculate_year_counts(csv_path, {"_group": list(labels)})
    if out.empty:
        raise ZeroDivisionError(
            f"No rows found for labels {sorted(labels)} in {csv_path}")
    r = out.iloc[0]
    return (float(r["women_n"]), float(r["total_n"]),
            int(r["women_n_unw"]), int(r["total_n_unw"]))