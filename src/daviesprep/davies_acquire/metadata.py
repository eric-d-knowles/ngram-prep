"""
Metadata loader for Davies corpora.

Loads metadata from sources.zip files (Excel for COHA, TSV for Movies)
to provide authoritative year and genre information indexed by textID.
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DaviesMetadataLoader:
    """
    Loads and indexes metadata from Davies corpus sources files.
    
    Each corpus has a sources.zip file containing metadata about all documents.
    This loader provides a unified interface to look up year and genre by textID.
    """
    
    def __init__(self, corpus_path: Path):
        """
        Initialize metadata loader for a Davies corpus.
        
        Args:
            corpus_path: Path to corpus directory (e.g., /path/to/COHA)
                        Should contain sources.zip file
        """
        self.corpus_path = Path(corpus_path)
        self.corpus_name = self.corpus_path.name
        self.metadata: Dict[int, Dict] = {}  # textID -> {year, genre, ...}
        self._loaded = False
        
    def load(self) -> bool:
        """
        Load metadata from sources.zip file.
        
        Returns:
            True if metadata loaded successfully, False otherwise
        """
        if self._loaded:
            return True
            
        # Try multiple naming patterns for sources file
        possible_names = [
            "sources.zip",                          # Standard (COHA)
            f"sources_{self.corpus_name}.zip",      # Named variant (Movies)
            f"sources_{self.corpus_name.lower()}.zip",  # Lowercase variant
        ]
        
        sources_zip = None
        for name in possible_names:
            candidate = self.corpus_path / name
            if candidate.exists():
                sources_zip = candidate
                break
        
        if sources_zip is None:
            logger.warning(f"Metadata file not found in {self.corpus_path}. Tried: {', '.join(possible_names)}")
            return False
        
        try:
            with zipfile.ZipFile(sources_zip, 'r') as zf:
                # Determine format by checking file contents
                file_list = zf.namelist()
                
                if any(f.endswith('.xlsx') for f in file_list):
                    self._load_excel_metadata(zf)
                elif any(f.endswith('.txt') for f in file_list):
                    self._load_tsv_metadata(zf)
                else:
                    logger.error(f"Unknown metadata format in {sources_zip}")
                    return False
            
            self._loaded = True
            logger.info(f"Loaded metadata for {len(self.metadata)} documents from {self.corpus_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading metadata from {sources_zip}: {e}")
            return False
    
    def _load_excel_metadata(self, zf: zipfile.ZipFile) -> None:
        """Load metadata from Excel file (COHA format)."""
        try:
            import openpyxl
        except ImportError:
            logger.error("openpyxl required for Excel metadata - install with: pip install openpyxl")
            return
        
        # Find Excel file
        excel_files = [f for f in zf.namelist() if f.endswith('.xlsx')]
        if not excel_files:
            logger.error("No Excel file found in sources.zip")
            return
        
        excel_file = excel_files[0]
        
        with zf.open(excel_file) as f:
            wb = openpyxl.load_workbook(f)
            ws = wb.active
            
            # Expected columns: textID, # words, genre, year, title, author, ...
            # Row 1 is header
            for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
                if not row[0]:  # Skip empty rows
                    continue
                
                text_id = int(row[0])
                # Column indices (0-based): 0=textID, 1=words, 2=genre, 3=year, 4=title, 5=author
                words = row[1] if len(row) > 1 else None
                genre = row[2] if len(row) > 2 else None
                year = row[3] if len(row) > 3 else None
                title = row[4] if len(row) > 4 else None
                author = row[5] if len(row) > 5 else None
                
                self.metadata[text_id] = {
                    'year': int(year) if year else None,
                    'genre': str(genre).lower() if genre else None,
                    'title': title,
                    'author': author,
                    'words': words,
                }
    
    def _load_tsv_metadata(self, zf: zipfile.ZipFile) -> None:
        """Load metadata from TSV file.
        
        Supports multiple formats:
        1. Movies/TV format: textID  fileID  #words  genre  year  ...
           - Uses fileID for indexing (ID that appears in @@ markers)
        2. COCA format: textID  year  genre  ...
           - Uses textID for indexing (simpler format)
        3. GloWbE format: textID  #words  country genre  URL  title
           - Uses textID for indexing, no year field (web corpus)
        """
        # Find text file
        txt_files = [f for f in zf.namelist() if f.endswith('.txt')]
        if not txt_files:
            logger.error("No text file found in sources.zip")
            return
        
        txt_file = txt_files[0]
        
        with zf.open(txt_file) as f:
            lines = f.read().decode('latin-1', errors='replace').split('\n')
        
        # Parse header
        if not lines:
            logger.error("Empty metadata file")
            return
        
        header = lines[0].strip().lower()
        
        # Detect format based on header or first data line
        if 'fileid' in header or 'textid\tfileid' in header:
            # Movies/TV format with separate fileID column (has header)
            self._load_tsv_movies_format(lines)
        elif 'country' in header and 'genre' in header:
            # GloWbE format: textID, #words, country genre, ... (has header)
            self._load_tsv_glowbe_format(lines)
        elif header[0].isdigit():
            # COCA format: No header, starts with data (textID year genre ...)
            # Check if second field is a year (4 digits)
            parts = header.split('\t')
            if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 4:
                self._load_tsv_coca_format(lines)
            else:
                logger.warning(f"Unknown TSV format. First line: {header[:100]}")
        else:
            logger.warning(f"Unknown TSV format. Header: {header[:100]}")
    
    def _load_tsv_movies_format(self, lines: list) -> None:
        """Load Movies/TV format TSV with fileID column."""
        # Skip header line and separator line (dashes)
        data_start = 2
        for i, line in enumerate(lines[1:4]):
            if all(c in '- \t\r' for c in line.strip()):
                data_start = i + 2
                break
        
        # Parse data rows using regex
        import re
        for line in lines[data_start:]:
            line = line.strip()
            if not line or all(c in '- \t\r' for c in line):
                continue
            
            try:
                # Pattern: textID(digits) fileID(digits) #words(digits) genre(...) year(4digits) ...
                match = re.match(
                    r'(\d+)\s+(\d+)\s+(\d+)\s+([^0-9]+?)\s+(\d{4})\s+(.+)$',
                    line
                )
                
                if match:
                    text_id = int(match.group(1))
                    file_id = int(match.group(2))  # The ID in @@ markers
                    words = int(match.group(3))
                    genre_str = match.group(4).strip()
                    year = int(match.group(5))
                    
                    genre = genre_str.lower() if genre_str else None
                    
                    # Index by fileID (the @@ marker ID), not textID
                    self.metadata[file_id] = {
                        'year': year,
                        'genre': genre,
                        'words': words,
                        'text_id': text_id,
                    }
            
            except (ValueError, AttributeError) as e:
                logger.debug(f"Error parsing metadata line: {line[:60]}... - {e}")
                continue
    
    def _load_tsv_coca_format(self, lines: list) -> None:
        """Load COCA format TSV: textID, year, genre, ...
        
        COCA has NO header line - data starts immediately at line 0.
        """
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('\t')
                if len(parts) >= 3:
                    text_id = int(parts[0])
                    year = int(parts[1])
                    genre = parts[2].lower() if parts[2] else None
                    
                    # Index by textID (same as @@ marker ID for COCA)
                    self.metadata[text_id] = {
                        'year': year,
                        'genre': genre,
                    }
            
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing COCA line: {line[:60]}... - {e}")
                continue
    
    def _load_tsv_glowbe_format(self, lines: list) -> None:
        """Load GloWbE format TSV: textID, #words, country genre, URL, title."""
        # Skip header line (line 0) and separator line (line 1)
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('\t')
                if len(parts) >= 3:
                    text_id = int(parts[0])
                    words = int(parts[1]) if parts[1].isdigit() else None
                    # Country and genre combined in one field (e.g., "AU B ")
                    country_genre = parts[2].strip() if len(parts) > 2 else None
                    
                    # GloWbE is a web corpus with no year, assign a placeholder
                    # or extract from URL if possible
                    year = 2012  # GloWbE was collected around 2012-2013
                    
                    # Index by textID (same as ## marker ID for GloWbE)
                    self.metadata[text_id] = {
                        'year': year,
                        'genre': country_genre,
                        'words': words,
                    }
            
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing GloWbE line: {line[:60]}... - {e}")
                continue
    
    def get_year_and_genre(self, text_id: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Look up year and genre for a given textID.
        
        Args:
            text_id: Document ID from @@ marker
            
        Returns:
            Tuple of (year, genre) or (None, None) if not found
        """
        if not self._loaded:
            return None, None
        
        if text_id not in self.metadata:
            logger.warning(f"TextID {text_id} not found in metadata")
            return None, None
        
        metadata = self.metadata[text_id]
        return metadata.get('year'), metadata.get('genre')
    
    def get_full_metadata(self, text_id: int) -> Optional[Dict]:
        """
        Get full metadata record for a textID.
        
        Args:
            text_id: Document ID from @@ marker
            
        Returns:
            Dictionary with year, genre, title, author, etc., or None if not found
        """
        if not self._loaded:
            return None
        
        return self.metadata.get(text_id)
