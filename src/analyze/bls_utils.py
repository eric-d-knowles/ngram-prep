"""
Utilities for scraping and processing Bureau of Labor Statistics (BLS) occupation data.

This module provides functions to:
- Scrape BLS occupation demographics from web tables (HTML, 2011+) or text files (1995-2010)
- Parse and normalize occupation labels
- Extract demographic percentages from formatted CSV files

Typical usage:
    >>> from analyze.bls_utils import scrape_bls_professions_csv, calculate_women_percentage
    >>> 
    >>> # Scrape HTML data (2011+)
    >>> df = scrape_bls_professions_csv(url="https://www.bls.gov/cps/aa2015/cpsaat11.htm")
    >>> 
    >>> # Scrape text file data (2000-2010)
    >>> df = scrape_bls_professions_csv(url="https://www.bls.gov/cps/aa2010/aat11.txt")
    >>> 
    >>> # Scrape text file data (1995-1999, no Asian demographic)
    >>> df = scrape_bls_professions_csv(url="https://www.bls.gov/cps/aa1999/aat11.txt")
    >>> 
    >>> # Calculate demographic percentage for a profession
    >>> pct = calculate_women_percentage("professionsBLS_2015.csv", "nurse")
"""

from pathlib import Path
import csv
import re
import time
from io import StringIO

import pandas as pd
import requests
from requests.exceptions import RequestException


# Module-level constants
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

BLS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bls.gov/cps/",
    "Connection": "keep-alive",
}

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


# Private helper functions
def _normalize_colname(col):
    """Normalize column name to lowercase alphanumeric with spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(col).strip().lower()).strip()


def _flatten_colname(col):
    """Flatten multi-level column names into a single normalized string."""
    if isinstance(col, tuple):
        joined = " ".join(str(part) for part in col if part is not None)
        return _normalize_colname(joined)
    return _normalize_colname(col)


def _parse_numeric(value):
    """Parse numeric value from various string formats, handling common BLS conventions."""
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "--", "NA", "N/A"}:
        return None
    text = text.replace(",", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


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
        occupation: Raw occupation string from BLS table
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


def _find_target_table(tables):
    """
    Identify the main occupation demographics table from a list of parsed tables.
    
    Args:
        tables: List of pandas DataFrames from pd.read_html()
        
    Returns:
        DataFrame with standardized column names
        
    Raises:
        ValueError: If no suitable table is found
    """
    for table in tables:
        if table.shape[1] < 6:
            continue

        flattened = [_flatten_colname(c) for c in table.columns]

        occ_idx = next((i for i, c in enumerate(flattened) if "occupation" in c), 0)
        total_idx = next((i for i, c in enumerate(flattened) if "total" in c and "employed" in c), None)
        women_idx = next((i for i, c in enumerate(flattened) if "women" in c), None)
        black_idx = next((i for i, c in enumerate(flattened) if "black" in c or "african" in c), None)
        asian_idx = next((i for i, c in enumerate(flattened) if "asian" in c), None)
        hispanic_idx = next((i for i, c in enumerate(flattened) if "hispanic" in c or "latino" in c), None)

        if None in (total_idx, women_idx, black_idx, asian_idx, hispanic_idx):
            continue

        selected = table.iloc[:, [occ_idx, total_idx, women_idx, black_idx, asian_idx, hispanic_idx]].copy()
        selected.columns = [
            "Occupation",
            "TotalEmployed",
            "Women",
            "AfricanAmerican",
            "Asian",
            "HispanicLatino",
        ]
        return selected

    raise ValueError("Could not find a CPS table with Occupation/TotalEmployed/Women/Black/Asian/Hispanic columns.")


def _fetch_bls_tables(url, timeout=30, retries=1, backoff_seconds=1.5):
    """
    Fetch and parse HTML tables from a BLS URL with retry logic.
    
    Args:
        url: BLS table URL
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        backoff_seconds: Initial backoff duration (increases with each retry)
        
    Returns:
        List of pandas DataFrames
        
    Raises:
        RuntimeError: If all fetch attempts fail
    """
    session = requests.Session()
    session.headers.update(BLS_HEADERS)

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            text = response.text
            if "Access Denied" in text:
                raise RuntimeError(
                    "BLS returned an Access Denied page. "
                    "Automated requests are being blocked for this URL."
                )
            return pd.read_html(StringIO(text))
        except (RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))
            else:
                break

    raise RuntimeError(
        "Unable to retrieve BLS table. "
        "BLS may be blocking automated requests from this environment. "
        "Try again later or download the table manually and parse local HTML."
    ) from last_error


def _extract_bls_reference_year(tables):
    """
    Attempt to extract reference year from BLS table metadata.
    
    Args:
        tables: List of pandas DataFrames
        
    Returns:
        Integer year or None if not found
    """
    year_candidates = []

    for table in tables:
        for col in table.columns:
            flat = _flatten_colname(col)
            for match in re.findall(r"\b(19\d{2}|20\d{2})\b", flat):
                year_candidates.append(int(match))

    if not year_candidates:
        for table in tables:
            sample = table.head(20).astype(str).to_string()
            for match in re.findall(r"\b(19\d{2}|20\d{2})\b", sample):
                year_candidates.append(int(match))

    year_candidates = [y for y in year_candidates if 1900 <= y <= 2100]
    return max(year_candidates) if year_candidates else None


def _output_path_with_year(output_csv, reference_year, inject_year_in_filename=True):
    """
    Generate output path with year designation if appropriate.
    
    Args:
        output_csv: Base output path
        reference_year: Year to inject
        inject_year_in_filename: Whether to modify filename
        
    Returns:
        Modified path string
    """
    path = Path(output_csv)
    if not inject_year_in_filename or reference_year is None:
        return str(path)

    stem = path.stem
    year_text = str(reference_year)

    if re.search(rf"(^|[^0-9]){year_text}([^0-9]|$)", stem):
        return str(path)

    if re.search(r"BLS\d{4}", stem):
        stem = re.sub(r"BLS\d{4}", f"BLS{reference_year}", stem)
    else:
        stem = f"{stem}_{reference_year}"

    return str(path.with_name(stem + path.suffix))


def _is_text_file_url(url):
    """Check if URL points to a text file (vs HTML page)."""
    return url.lower().endswith('.txt')


def _fetch_text_file(url, timeout=30):
    """
    Fetch text file content from a BLS URL.
    
    Args:
        url: BLS text file URL
        timeout: Request timeout in seconds
        
    Returns:
        String content of the text file
        
    Raises:
        RuntimeError: If fetch fails
    """
    session = requests.Session()
    session.headers.update(BLS_HEADERS)
    
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        text = response.text
        if "Access Denied" in text:
            raise RuntimeError(
                "BLS returned an Access Denied page. "
                "Automated requests are being blocked for this URL."
            )
        return text
    except RequestException as exc:
        raise RuntimeError(
            f"Unable to retrieve BLS text file from {url}. "
            "The file may not exist or BLS may be blocking requests."
        ) from exc


def _parse_text_file_content(text_content):
    """
    Parse BLS text file formats (used 1995-2010).
    
    Handles three formats:
    - 1995-1996: Pipe-delimited (|) with Occupation | Total | Women% | Black% | Hispanic%
    - 1997-1999: Space-separated (4 columns): Total employed, Women%, Black%, Hispanic% (no Asian)
    - 2000-2010: Space-separated (5 columns): Total employed, Women%, Black%, Asian%, Hispanic%
    
    Args:
        text_content: Raw text content from BLS file
        
    Returns:
        DataFrame with standardized columns (Asian set to None if not in source)
    """
    lines = text_content.split('\n')
    
    # Extract year from header
    reference_year = None
    for line in lines[:50]:
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', line)
        if year_match:
            reference_year = int(year_match.group(1))
            break
    
    # Parse data lines
    records = []
    
    for line in lines:
        # Skip empty lines, headers, notes
        if not line.strip():
            continue
        if 'NOTE:' in line or 'Data not shown' in line:
            continue
        if 'HOUSEHOLD DATA' in line or 'ANNUAL AVERAGES' in line:
            continue
        if 'Total employed' in line or 'Percent of total' in line:
            continue
        if line.strip().startswith('Occupation'):
            continue
        if '____' in line or '----' in line:  # Skip separator lines
            continue
            
        # Try pipe-delimited format first (1995-1996)
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 5:
                occupation = parts[0].strip()
                occupation = re.sub(r'\.+$', '', occupation).strip()
                
                if not occupation or len(occupation) < 3:
                    continue
                if occupation.isupper() and len(occupation.split()) <= 4:
                    continue
                
                try:
                    records.append({
                        'Occupation': occupation,
                        'TotalEmployed': _parse_numeric(parts[1].strip()),
                        'Women': _parse_numeric(parts[2].strip()),
                        'AfricanAmerican': _parse_numeric(parts[3].strip()),
                        'Asian': None,  # Not available in 1995-1996 format
                        'HispanicLatino': _parse_numeric(parts[4].strip()),
                    })
                except (IndexError, ValueError):
                    continue
                continue
        
        # Try to match 5 numeric columns (2000-2010 space-separated: women, black, asian, hispanic)
        match_5col = re.search(
            r'^\s*(.+?)\s+'
            r'([\d,]+)\s+'
            r'([\d.]+|\-|\(1\))\s+'
            r'([\d.]+|\-|\(1\))\s+'
            r'([\d.]+|\-|\(1\))\s+'
            r'([\d.]+|\-|\(1\))\s*$',
            line
        )
        
        if match_5col:
            occupation = match_5col.group(1).strip()
            occupation = re.sub(r'\.+$', '', occupation).strip()
            
            if not occupation or len(occupation) < 3:
                continue
            if occupation.isupper() and len(occupation.split()) <= 4:
                continue
                
            records.append({
                'Occupation': occupation,
                'TotalEmployed': _parse_numeric(match_5col.group(2)),
                'Women': _parse_numeric(match_5col.group(3)),
                'AfricanAmerican': _parse_numeric(match_5col.group(4)),
                'Asian': _parse_numeric(match_5col.group(5)),
                'HispanicLatino': _parse_numeric(match_5col.group(6)),
            })
            continue
        
        # Try to match 4 numeric columns (1997-1999 space-separated: women, black, hispanic, no asian)
        match_4col = re.search(
            r'^\s*(.+?)\s+'
            r'([\d,]+)\s+'
            r'([\d.]+|\-|\(1\))\s+'
            r'([\d.]+|\-|\(1\))\s+'
            r'([\d.]+|\-|\(1\))\s*$',
            line
        )
        
        if match_4col:
            occupation = match_4col.group(1).strip()
            occupation = re.sub(r'\.+$', '', occupation).strip()
            
            if not occupation or len(occupation) < 3:
                continue
            if occupation.isupper() and len(occupation.split()) <= 4:
                continue
                
            records.append({
                'Occupation': occupation,
                'TotalEmployed': _parse_numeric(match_4col.group(2)),
                'Women': _parse_numeric(match_4col.group(3)),
                'AfricanAmerican': _parse_numeric(match_4col.group(4)),
                'Asian': None,  # Not available in 1997-1999 format
                'HispanicLatino': _parse_numeric(match_4col.group(5)),
            })
    
    if not records:
        raise ValueError("No valid occupation data found in text file")
    
    df = pd.DataFrame(records)
    df.attrs['bls_reference_year'] = reference_year
    return df


# Public API functions
def calculate_women_percentage(csv_file, profession):
    """
    Calculate percentage of profession made up of women from a BLS CSV file.
    
    This function searches through a properly-formatted BLS statistics CSV file
    for rows whose label columns (label1-label5) exactly match the specified
    profession (after normalization), and computes the weighted average
    percentage of women in that profession.
    
    Args:
        csv_file: Path to the CSV file (formatted as output by scrape_bls_professions_csv)
        profession: The profession name to search for (exact label match after normalization)
        
    Returns:
        The percentage of women in that profession (as a decimal, 0-1)
        
    Raises:
        ZeroDivisionError: If no matching professions found
        ValueError: If data is malformed
        
    Example:
        >>> pct = calculate_women_percentage("professionsBLS_2015.csv", "nurse")
        >>> print(f"Nurses: {pct*100:.1f}% women")
    """
    total_counter = 0.0
    women_counter = 0.0

    profession_norm = _normalize_label_token(str(profession).strip().lower())
    label_cols = ["label1", "label2", "label3", "label4", "label5"]

    with open(csv_file, 'r', newline='', encoding='utf-8') as br:
        reader = csv.DictReader(br)

        # Backward compatibility for non-standard CSVs lacking label columns.
        has_label_cols = all(col in (reader.fieldnames or []) for col in label_cols)

        for row in reader:
            if has_label_cols:
                row_labels = {
                    _normalize_label_token(str(row.get(col, "")).strip().lower())
                    for col in label_cols
                    if str(row.get(col, "")).strip()
                }
                is_match = profession_norm in row_labels
            else:
                # Fallback behavior for legacy files without label columns.
                line_blob = ",".join(str(v) for v in row.values())
                is_match = str(profession) in line_blob

            if not is_match:
                continue

            total_employed = float(row["TotalEmployed"])
            women_pct = float(row["Women"])
            total_counter += total_employed
            women_counter += total_employed * women_pct

    percentage = women_counter / total_counter
    return percentage / 100


def scrape_bls_professions_csv(
    url="https://www.bls.gov/cps/cpsaat11.htm",
    output_csv="professionsBLS_from_web.csv",
    min_total_employed=50,
    return_year=False,
    inject_year_in_filename=True,
    request_timeout=30,
    request_retries=1,
):
    """
    Scrape Bureau of Labor Statistics occupation demographics and save to CSV.
    
    This function fetches the Current Population Survey (CPS) table from the BLS website,
    extracts occupation names and demographic percentages (Women, African American, Asian,
    Hispanic/Latino), processes and normalizes the data, and saves it to a CSV file.
    
    Supports both HTML tables (2011+) and fixed-width text files (pre-2011).
    
    The output CSV includes tokenized occupation labels suitable for matching against
    word embedding vocabularies.
    
    Args:
        url: URL of the BLS CPS table to scrape
            Default is the main occupational demographics table (cpsaat11.htm)
            For pre-2011 data, use .txt URLs like "https://www.bls.gov/cps/aa2010/aat11.txt"
        output_csv: Path where the processed CSV should be saved
        min_total_employed: Minimum employment threshold to include an occupation (in thousands)
        return_year: If True, returns a tuple of (DataFrame, reference_year)
        inject_year_in_filename: If True, automatically adds year to output filename
        request_timeout: Per-request timeout in seconds for BLS fetches
        request_retries: Number of retries for transient fetch/parse failures
        
    Returns:
        DataFrame with columns: TotalEmployed, Women, AfricanAmerican, Asian, HispanicLatino,
                                none, label1-label5 (tokenized occupation labels)
        If return_year=True, returns (DataFrame, year) tuple
        
    Example:
        >>> # Scrape 2015 HTML data
        >>> df = scrape_bls_professions_csv(
        ...     url="https://www.bls.gov/cps/aa2015/cpsaat11.htm",
        ...     output_csv="professionsBLS.csv"
        ... )
        
        >>> # Scrape 2010 text data
        >>> df = scrape_bls_professions_csv(
        ...     url="https://www.bls.gov/cps/aa2010/aat11.txt",
        ...     output_csv="professionsBLS.csv"
        ... )
        
        >>> # Get year information
        >>> df, year = scrape_bls_professions_csv(
        ...     url="https://www.bls.gov/cps/cpsaat11.htm",
        ...     return_year=True
        ... )
        >>> print(f"Reference year: {year}")
    """
    # Determine file format and fetch data
    if _is_text_file_url(url):
        text_content = _fetch_text_file(url, timeout=request_timeout)
        raw = _parse_text_file_content(text_content)
        reference_year = raw.attrs.get('bls_reference_year')
    else:
        tables = _fetch_bls_tables(
            url,
            timeout=request_timeout,
            retries=request_retries,
        )
        reference_year = _extract_bls_reference_year(tables)
        raw = _find_target_table(tables)
    
    resolved_output_csv = _output_path_with_year(
        output_csv,
        reference_year,
        inject_year_in_filename=inject_year_in_filename,
    )

    raw["Occupation"] = raw["Occupation"].astype(str).str.strip()
    for col in ["TotalEmployed", "Women", "AfricanAmerican", "Asian", "HispanicLatino"]:
        raw[col] = raw[col].map(_parse_numeric)

    required_cols = ["TotalEmployed", "Women", "AfricanAmerican", "HispanicLatino"]
    cleaned = raw.dropna(subset=required_cols).copy()
    cleaned = cleaned[cleaned["Occupation"].ne("")]
    cleaned = cleaned[cleaned["Occupation"] != "|"]
    cleaned = cleaned[cleaned["TotalEmployed"] >= min_total_employed]

    cleaned["TotalEmployed"] = cleaned["TotalEmployed"].round().astype(int)
    cleaned = cleaned.sort_values("Women", ascending=True).reset_index(drop=True)

    label_columns = ["label1", "label2", "label3", "label4", "label5"]
    labels = cleaned["Occupation"].map(lambda x: _tokenize_occupation(x, max_tokens=5))
    label_df = pd.DataFrame(labels.tolist(), columns=label_columns, index=cleaned.index)

    out = pd.DataFrame(
        {
            "TotalEmployed": cleaned["TotalEmployed"],
            "Women": cleaned["Women"],
            "AfricanAmerican": cleaned["AfricanAmerican"],
            "Asian": cleaned["Asian"],
            "HispanicLatino": cleaned["HispanicLatino"],
            "none": "",
        }
    )
    out = pd.concat([out, label_df], axis=1)
    out = out[TARGET_COLUMNS]

    def fmt_percent(x):
        if pd.isna(x):
            return ""
        x = float(x)
        if abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
        return f"{x:.1f}".rstrip("0").rstrip(".")

    with open(resolved_output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TARGET_COLUMNS)
        for row in out.itertuples(index=False):
            writer.writerow(
                [
                    int(row.TotalEmployed),
                    fmt_percent(row.Women),
                    fmt_percent(row.AfricanAmerican),
                    fmt_percent(row.Asian),
                    fmt_percent(row.HispanicLatino),
                    "",
                    row.label1,
                    row.label2,
                    row.label3,
                    row.label4,
                    row.label5,
                ]
            )

    if reference_year is not None:
        print(f"Detected BLS reference year from page content: {reference_year}")
    else:
        print("Detected BLS reference year from page content: unknown")

    print(f"Saved {len(out)} rows to {resolved_output_csv}")

    out.attrs["bls_reference_year"] = reference_year
    out.attrs["output_csv"] = resolved_output_csv

    if return_year:
        return out, reference_year
    return out


def scrape_bls_professions_csv_batch(
    file_list,
    output_dir,
    output_basename="professionsBLS.csv",
    min_total_employed=50,
    inject_year_in_filename=True,
    continue_on_error=True,
    request_timeout=20,
    request_retries=1,
):
    """
    Scrape multiple BLS occupation URLs and return a run summary.

    Args:
        file_list: Iterable of BLS URLs to scrape
        output_dir: Directory where output CSV files will be written
        output_basename: Base filename used before year injection
        min_total_employed: Minimum employment threshold for each scrape
        inject_year_in_filename: Whether to inject detected year into output filename
        continue_on_error: If True, keep processing after an individual failure
        request_timeout: Per-request timeout in seconds for each URL
        request_retries: Number of retries for each URL

    Returns:
        pandas.DataFrame with one row per URL and columns:
        year, rows, status, output_csv, url, error

    Example:
        >>> urls = [
        ...     "https://www.bls.gov/cps/aa2010/aat11.txt",
        ...     "https://www.bls.gov/cps/aa1999/AAT11.TXT",
        ... ]
        >>> runs_df = scrape_bls_professions_csv_batch(
        ...     file_list=urls,
        ...     output_dir="/tmp/bls_scraped"
        ... )
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    runs = []
    output_csv = str(output_dir_path / output_basename)

    for url in file_list:
        try:
            df, year = scrape_bls_professions_csv(
                url=url,
                output_csv=output_csv,
                min_total_employed=min_total_employed,
                return_year=True,
                inject_year_in_filename=inject_year_in_filename,
                request_timeout=request_timeout,
                request_retries=request_retries,
            )

            runs.append(
                {
                    "url": url,
                    "year": year,
                    "rows": len(df),
                    "output_csv": df.attrs.get("output_csv"),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            runs.append(
                {
                    "url": url,
                    "year": None,
                    "rows": None,
                    "output_csv": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                break

    runs_df = pd.DataFrame(runs)

    if runs_df.empty:
        return runs_df

    summary_cols = ["year", "rows", "status", "output_csv", "url", "error"]
    return runs_df[summary_cols].sort_values(
        by=["year", "status"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
