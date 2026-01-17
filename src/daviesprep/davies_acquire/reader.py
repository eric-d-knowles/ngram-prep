"""Text file reading and parsing for Davies corpora."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "discover_text_files",
    "extract_year_from_filename",
    "extract_genre_from_filename",
    "extract_year_from_document_filename",
    "extract_text_id_from_marker",
    "read_text_file",
    "read_text_file_with_genre",
]


def discover_text_files(text_dir: Path) -> list[Path]:
    """
    Discover all text archive files in the text directory.

    Args:
        text_dir: Directory containing text zip files

    Returns:
        Sorted list of text file paths

    Example:
        >>> files = discover_text_files(Path("/data/COHA/text"))
        >>> files[0]
        Path('/data/COHA/text/text_1810s_kso.zip')
    """
    if not text_dir.exists():
        raise ValueError(f"Text directory does not exist: {text_dir}")

    # Find all .zip files matching text_*
    pattern = "text_*.zip"
    files = sorted(text_dir.glob(pattern))

    if not files:
        raise ValueError(f"No text files found in {text_dir}")

    logger.info(f"Found {len(files)} text archive files")
    return files


def extract_year_from_filename(
    filename: str,
    decade_pattern: str = r"text_(\d{4})s_",
    year_pattern: Optional[str] = None,
) -> int:
    """
    Extract year/decade from Davies corpus filename.

    For decade-based corpora (like COHA), extracts the starting year of the decade.
    For year-based corpora, extracts the specific year.

    Args:
        filename: Filename to parse (e.g., "text_1810s_kso.zip")
        decade_pattern: Regex pattern for decade extraction
        year_pattern: Regex pattern for year extraction (optional)

    Returns:
        Year as integer (e.g., 1810 for "text_1810s_kso.zip")

    Raises:
        ValueError: If no year/decade found in filename

    Example:
        >>> extract_year_from_filename("text_1810s_kso.zip")
        1810
    """
    # Try decade pattern first
    if decade_pattern:
        match = re.search(decade_pattern, filename)
        if match:
            return int(match.group(1))

    # Try year pattern if provided
    if year_pattern:
        match = re.search(year_pattern, filename)
        if match:
            return int(match.group(1))

    raise ValueError(f"Could not extract year from filename: {filename}")


def extract_genre_from_filename(filename: str) -> Optional[str]:
    """
    Extract genre code from Davies corpus text filename.

    Davies text files follow the pattern: {genre}_{year}_{doc_id}.txt
    Examples: mag_1815_552651.txt, nf_1816_747562.txt, fic_1920_123456.txt

    Args:
        filename: Filename to parse (e.g., "mag_1815_552651.txt")

    Returns:
        Genre code (e.g., "mag", "nf", "fic") or None if pattern doesn't match

    Example:
        >>> extract_genre_from_filename("mag_1815_552651.txt")
        'mag'
        >>> extract_genre_from_filename("fic_1920_123456.txt")
        'fic'
    """
    # Pattern: genre_year_docid.txt
    match = re.match(r'^([a-z]+)_\d{4}_\d+\.txt$', filename)
    if match:
        return match.group(1)
    return None


def extract_year_from_document_filename(filename: str) -> Optional[int]:
    """
    Extract specific year from Davies corpus document filename.

    Davies text files follow the pattern: {genre}_{year}_{doc_id}.txt
    Examples: mag_1815_552651.txt, nf_1816_747562.txt, fic_1920_123456.txt

    Args:
        filename: Filename to parse (e.g., "mag_1815_552651.txt")

    Returns:
        Year as integer (e.g., 1815) or None if pattern doesn't match

    Example:
        >>> extract_year_from_document_filename("mag_1815_552651.txt")
        1815
        >>> extract_year_from_document_filename("fic_1920_123456.txt")
        1920
    """
    # Pattern: genre_year_docid.txt
    match = re.match(r'^[a-z]+_(\d{4})_\d+\.txt$', filename)
    if match:
        return int(match.group(1))
    return None


def extract_text_id_from_marker(content: str) -> Optional[int]:
    """
    Extract textID from marker at start of Davies corpus text.

    Davies corpora include a document identifier marker at the start of each text file:
    - Most corpora: @@textID (e.g., @@552651)
    - GloWbE corpus: ##textID (e.g., ##703)
    
    The marker may be preceded by whitespace (e.g., "\\r\\n@@552651").

    Args:
        content: Text content containing marker

    Returns:
        TextID as integer (e.g., 552651) or None if marker not found

    Example:
        >>> extract_text_id_from_marker("@@552651\\nSome text...")
        552651
        >>> extract_text_id_from_marker("##703\\nSome text...")
        703
    """
    # Strip leading whitespace and check for markers
    stripped = content.lstrip()
    
    # Check for @@ marker (Movies, TV, COCA, COHA, etc.)
    if stripped.startswith('@@'):
        lines = stripped.split('\n', 1)
        marker_line = lines[0]
        match = re.match(r'@@(\d+)', marker_line)
        if match:
            return int(match.group(1))
    
    # Check for ## marker (GloWbE)
    if stripped.startswith('##'):
        lines = stripped.split('\n', 1)
        marker_line = lines[0]
        match = re.match(r'##(\d+)', marker_line)
        if match:
            return int(match.group(1))
    
    return None


def read_text_file(
    zip_path: Path,
    year: int,
) -> Iterator[Tuple[int, str]]:
    """
    Read and yield document text from a Davies corpus zip file.

    Each zip contains multiple text documents. This function yields
    the text content of each document along with its year.

    Args:
        zip_path: Path to zip file
        year: Year associated with this file

    Yields:
        Tuples of (year, document_text)

    Example:
        >>> for year, text in read_text_file(Path("text_1810s.zip"), 1810):
        ...     print(f"Year {year}: {len(text)} chars")
    """
    logger.debug(f"Reading {zip_path.name} (year={year})")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all .txt files in the archive
            txt_files = [f for f in zf.namelist() if f.endswith('.txt')]

            if not txt_files:
                logger.warning(f"No .txt files found in {zip_path.name}")
                return

            for txt_file in txt_files:
                try:
                    # Read file content
                    with zf.open(txt_file) as f:
                        content = f.read().decode('utf-8', errors='replace')

                    # Skip document ID markers (e.g., "@@552651")
                    # These appear at the start of COHA text files
                    if content.startswith('@@'):
                        # Remove the marker line
                        lines = content.split('\n', 1)
                        if len(lines) > 1:
                            content = lines[1]

                    # Only yield if there's actual content
                    if content.strip():
                        yield year, content

                except Exception as e:
                    logger.warning(f"Error reading {txt_file} from {zip_path.name}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error opening {zip_path}: {e}")
        raise


def read_text_file_with_genre(
    zip_path: Path,
    year: int,
    metadata_loader=None,
) -> Iterator[Tuple[int, str, Optional[str]]]:
    """
    Read and yield document text with genre metadata from a Davies corpus zip file.

    Each zip contains multiple text documents. If metadata_loader is provided,
    uses authoritative metadata; otherwise falls back to filename parsing.

    Args:
        zip_path: Path to zip file
        year: Year associated with this file (for fallback)
        metadata_loader: Optional DaviesMetadataLoader for authoritative metadata lookup

    Yields:
        Tuples of (year, document_text, genre_code)

    Example:
        >>> for year, text, genre in read_text_file_with_genre(Path("text_1810s.zip"), 1810):
        ...     print(f"Year {year}, Genre {genre}: {len(text)} chars")
        Year 1815, Genre mag: 66444 chars
    """
    logger.debug(f"Reading {zip_path.name} (year={year})")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all .txt files in the archive
            txt_files = [f for f in zf.namelist() if f.endswith('.txt')]

            if not txt_files:
                logger.warning(f"No .txt files found in {zip_path.name}")
                return

            for txt_file in txt_files:
                try:
                    # Read file content first to extract textID
                    with zf.open(txt_file) as f:
                        content = f.read().decode('utf-8', errors='replace')

                    # Extract textID from @@ marker
                    text_id = extract_text_id_from_marker(content)
                    
                    # Determine year and genre: prioritize metadata, then filename, then fallback
                    if metadata_loader and text_id:
                        doc_year, genre = metadata_loader.get_year_and_genre(text_id)
                        if doc_year is None:
                            # Fallback to filename-based extraction
                            filename = Path(txt_file).name
                            genre = extract_genre_from_filename(filename)
                            doc_year = extract_year_from_document_filename(filename)
                            if doc_year is None:
                                doc_year = year
                    else:
                        # Fallback: extract from filename (COHA-style)
                        filename = Path(txt_file).name
                        genre = extract_genre_from_filename(filename)
                        doc_year = extract_year_from_document_filename(filename)
                        if doc_year is None:
                            doc_year = year

                    # Skip document ID markers (e.g., "@@552651")
                    # These appear at the start of Davies text files
                    if content.startswith('@@'):
                        # Remove the marker line
                        lines = content.split('\n', 1)
                        if len(lines) > 1:
                            content = lines[1]

                    # Only yield if there's actual content
                    if content.strip():
                        yield doc_year, content, genre

                except Exception as e:
                    logger.warning(f"Error reading {txt_file} from {zip_path.name}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error opening {zip_path}: {e}")
        raise
