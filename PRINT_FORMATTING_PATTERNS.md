# Print Formatting Patterns in lexichron

This document details the print output formatting patterns found in training and acquisition functions across the `/scratch/edk202/lexichron/src` directory.

---

## Overview

The codebase uses a hierarchical formatting system with three main levels:
1. **Major Banners** (━ style) - Top-level pipeline/process headers
2. **Section Headers** (═ style) - Configuration and category sections  
3. **Subsections** (─ style) - Detailed configuration items
4. **Data rows** - Key-value pairs with aligned spacing

---

## 1. Word2Vec Training Display (`src/train/word2vec/display.py`)

### Print Statements & Output Structure

#### Training Header
```python
def print_training_header(start_time, db_path, model_dir, log_dir, max_parallel_models, grid_params):
    lines = [
        "WORD2VEC MODEL TRAINING",
        "━" * LINE_WIDTH,  # 100 chars of ━
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,  # 100 chars of ═
        f"Database:             {db_path_str}",
        f"Model directory:      {model_dir_str}",
        f"Log directory:        {log_dir_str}",
        f"Parallel models:      {max_parallel_models}",
        "",
        grid_params,
        "",
    ]
    print("\n".join(lines), flush=True)
```

**Output Example:**
```
WORD2VEC MODEL TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start Time: 2026-01-08 10:30:45

Configuration
════════════════════════════════════════════════════════════════════════════════════════
Database:             /path/to/database.db
Model directory:      /path/to/models
Log directory:        /path/to/logs
Parallel models:      4

Grid parameters...
```

**Key Pattern:**
- **Major Title**: All caps, plain text
- **Top Separator**: `━` (BOX DRAWINGS HEAVY HORIZONTAL) × 100
- **Timestamp**: ISO format `YYYY-MM-DD HH:MM:SS`
- **Section Title**: Mixed case
- **Section Separator**: `═` (BOX DRAWINGS DOUBLE HORIZONTAL) × 100
- **Data Rows**: Label padded to fixed width (22-24 chars), then value
- **Spacing**: Blank lines between major sections

#### Completion Banner
```python
def print_completion_banner(model_dir, total_tasks):
    lines = [
        "",
        "Training Complete",
        "═" * LINE_WIDTH,
        f"Models trained:       {total_tasks}",
        f"Model directory:      {model_dir_str}",
        "━" * LINE_WIDTH,
        "",
    ]
    print("\n".join(lines), flush=True)
```

**Output Example:**
```

Training Complete
════════════════════════════════════════════════════════════════════════════════════════
Models trained:       12
Model directory:      /path/to/models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```

#### Alignment Header
```python
def print_alignment_header(start_time, model_dir, output_dir, anchor_year, num_models, 
                          weighted_alignment, stability_method=None, ...):
    lines = [
        "",
        "WORD2VEC MODEL NORMALIZATION & ALIGNMENT",
        "━" * LINE_WIDTH,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,
        f"Model directory:      {model_dir_str}",
        f"Output directory:     {output_dir_str}",
        f"Anchor year:          {anchor_year}",
        f"Total models:         {num_models}",
    ]
    if workers is not None:
        lines.append(f"Parallel workers:     {workers}")
    
    lines.append("")
    lines.append("Alignment Method")
    lines.append("─" * LINE_WIDTH)
    
    if weighted_alignment:
        lines.append(f"Type:                 Weighted Procrustes")
        if stability_method:
            lines.append(f"Stability metric:     {stability_method}")
        if include_frequency is not None:
            freq_status = "Yes" if include_frequency else "No"
            lines.append(f"Include frequency:    {freq_status}")
        if include_frequency and frequency_weight is not None:
            lines.append(f"Frequency weight:     {frequency_weight:.2f} ({int(frequency_weight*100)}% frequency, {int((1-frequency_weight)*100)}% stability)")
    else:
        lines.append(f"Type:                 Unweighted Procrustes (all shared vocabulary)")
```

**Output Example:**
```

WORD2VEC MODEL NORMALIZATION & ALIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start Time: 2026-01-08 10:30:45

Configuration
════════════════════════════════════════════════════════════════════════════════════════
Model directory:      /path/to/models
Output directory:     /path/to/output
Anchor year:          1900
Total models:         50
Parallel workers:     8

Alignment Method
──────────────────────────────────────────────────────────────────────────────────────────
Type:                 Weighted Procrustes
Stability metric:     pairwise_stability
Include frequency:    Yes
Frequency weight:     0.70 (70% frequency, 30% stability)
```

**Key Pattern for Subsections:**
- **Subsection Title**: Mixed case
- **Subsection Separator**: `─` (BOX DRAWINGS LIGHT HORIZONTAL) × 100
- **Data Rows**: Same padded format as above

---

## 2. N-Gram Acquisition Pipeline (`src/ngramprep/ngram_acquire/reporter.py`)

### Print Statements & Output Structure

```python
def print_pipeline_header(start_time, page_url, db_path, start_idx, end_idx, total_files,
                         files_to_get, files_to_skip, workers, write_batch_size, 
                         ngram_size, ngram_type, overwrite_db, open_type):
    print(format_banner("N-GRAM ACQUISITION PIPELINE", style="━"))
    print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
    print()
    print(format_banner("Download Configuration"))
    print(f"Ngram repo:           {truncate_path_to_fit(page_url, 'Ngram repo:           ')}")
    print(f"DB path:              {truncate_path_to_fit(db_path, 'DB path:              ')}")
    print(f"File range:           {start_idx} to {end_idx}")
    print(f"Total files:          {total_files}")
    print(f"Files to get:         {files_to_get}")
    print(f"Skipping:             {files_to_skip}")
    print(f"Download workers:     {workers}")
    print(f"Batch size:           {write_batch_size:,}")
    print(f"Ngram size:           {ngram_size}")
    print(f"Ngram type:           {ngram_type}")
    print(f"Overwrite DB:         {overwrite_db}")
    print(f"DB Profile:           {open_type}")
    print()
    print(format_banner("Download Progress"))

def print_final_summary(start_time, end_time, success, failure, written, batches, 
                       uncompressed_bytes):
    total_runtime = end_time - start_time
    ok = len(success)
    bad = len(failure)
    # ... calculations for time_per_file, fph, mb_per_sec
```

**Output Example:**
```
N-GRAM ACQUISITION PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start Time: 2026-01-08 10:30:45

Download Configuration
════════════════════════════════════════════════════════════════════════════════════════
Ngram repo:           https://storage.googleapis.com/books/ngrams
DB path:              /data/ngrams.db
File range:           0 to 100
Total files:          500
Files to get:         100
Skipping:             0
Download workers:     4
Batch size:           10,000
Ngram size:           1
Ngram type:           all
Overwrite DB:         False
DB Profile:           fast

Download Progress
════════════════════════════════════════════════════════════════════════════════════════
```

**Key Pattern:**
- Uses `format_banner()` helper function for consistency
- Numeric values use thousands separators (e.g., `10,000`)
- Paths are truncated with `truncate_path_to_fit()` to maintain line width
- Same 100-character line width standard

---

## 3. Progress Display Classes (`src/ngramprep/utilities/progress.py`)

### ProgressDisplay Class Methods

```python
class ProgressDisplay:
    def __init__(self, width: int = 100):
        self.width = width

    def print_banner(self, title: str, style: str = "═", include_blank: bool = False) -> None:
        if include_blank:
            print()
        print(title)
        print(style * self.width)

    def print_section(self, title: str, style: str = "—") -> None:
        print()
        print(title)
        print(style * self.width)

    def print_config_items(self, items: Dict[str, Any], indent: str = "") -> None:
        if not items:
            return
        max_key_len = max(len(str(k)) for k in items.keys())
        for key, value in items.items():
            padding = " " * (max_key_len - len(str(key)))
            print(f"{indent}{key}:{padding} {value}")

    def print_summary_box(self, title: str, items: Dict[str, Any], box_width: Optional[int] = None) -> None:
        if box_width is None:
            box_width = self.width
        
        title_width = len(f" {title} ")
        item_widths = [len(f" {key}: {value} ") for key, value in items.items()]
        max_content_width = max([title_width] + item_widths)
        actual_width = max(box_width, max_content_width + 2)
        
        print("┌" + "─" * (actual_width - 2) + "┐")
        
        title_text = f" {title} "
        padding = " " * (actual_width - len(title_text) - 2)
        print("│" + title_text + padding + "│")
        
        print("├" + "─" * (actual_width - 2) + "┤")
        
        for key, value in items.items():
            item_text = f" {key}: {value} "
            padding = " " * (actual_width - len(item_text) - 2)
            print("│" + item_text + padding + "│")
        
        print("└" + "─" * (actual_width - 2) + "┘")
```

**Output Examples:**

Banner:
```
Configuration
════════════════════════════════════════════════════════════════════════════════════════
```

Section:
```

Configuration
────────────────────────────────────────────────────────────────────────────────────────
```

Config Items (auto-aligned):
```
database: /path/to/db
workers:  4
timeout:  30
```

Summary Box:
```
┌──────────────────────────────┐
│ Processing Complete          │
├──────────────────────────────┤
│ Items processed: 1,000,000   │
│ Elapsed time: 2h 15m         │
│ Throughput: 123.4 items/sec  │
└──────────────────────────────┘
```

### Formatting Methods

```python
@staticmethod
def format_rate(count: int, elapsed_seconds: float, unit: str = "items") -> str:
    """Format a processing rate."""
    if elapsed_seconds <= 0:
        return f"0 {unit}/sec"
    rate = count / elapsed_seconds
    if rate >= 1000:
        return f"{rate:,.0f} {unit}/sec"
    elif rate >= 10:
        return f"{rate:.1f} {unit}/sec"
    else:
        return f"{rate:.2f} {unit}/sec"

@staticmethod
def format_percentage(numerator: int, denominator: int) -> str:
    """Format a percentage."""
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"
```

**Output Examples:**
- `4,000 items/sec` (rate ≥ 1000)
- `123.5 items/sec` (rate 10-1000)
- `2.34 items/sec` (rate < 10)
- `75.0%` (percentage to 1 decimal place)

---

## 4. N-Gram Filter Pipeline (`src/ngramprep/ngram_filter/pipeline/progress.py`)

### Progress Banner with Aligned Columns

```python
def print_phase_banner() -> None:
    """Print the pipeline phase 1 banner and headers."""
    field_width = 14
    fields = ["items", "kept%", "workers", "units", "rate", "elapsed"]
    line = "─"
    print('\n' + ''.join(f"{field:^{field_width}}" for field in fields))
    print(''.join(f"{line*field_width:^{field_width}}" for field in fields))
```

**Output Example:**
```

        items         kept%       workers        units         rate       elapsed
──────────────────────────────────────────────────────────────────────────────────────
```

**Key Pattern:**
- **Field Width**: 14 characters per column
- **Alignment**: Center-aligned using `^` format specifier
- **Separators**: Each column separator is the separator character repeated field_width times
- **Column Header**: Uses field names

### Progress Formatter Class

```python
class ProgressFormatter:
    @staticmethod
    def format_count(count: int) -> str:
        """Format a count with K/M/B suffixes to 2 decimal places."""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}B"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.2f}K"
        else:
            return str(count)

    @staticmethod
    def format_rate(items_per_second: float) -> str:
        """Format a rate for display (e.g., '1.2k/s', '850/s')."""
        if items_per_second >= 1000:
            return f"{items_per_second / 1000:.1f}k/s"
        return f"{items_per_second:.0f}/s"

    @staticmethod
    def format_elapsed_time(seconds: float) -> str:
        """Format elapsed time for display."""
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes:02d}m"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs:02d}s"
        else:
            return f"{seconds:.0f}s"
```

**Output Examples:**
- **Counts**: `1.23K`, `4.56M`, `7.89B`
- **Rates**: `1.2k/s`, `850/s`
- **Times**: `2h15m`, `5m30s`, `45s`

---

## 5. N-Gram Pivot Pipeline (`src/ngramprep/ngram_pivot/pipeline/progress.py`)

### Pivot Pipeline Banner

```python
def print_phase_banner() -> None:
    """Print the pipeline phase banner and headers."""
    field_width = 16
    fields = ["ngrams", "exp", "units", "rate", "elapsed"]
    line = "─"
    print('\n' + ''.join(f"{field:^{field_width}}" for field in fields))
    print(''.join(f"{line*field_width:^{field_width}}" for field in fields))
```

**Output Example:**
```

         ngrams           exp          units          rate        elapsed
────────────────────────────────────────────────────────────────────────────────
```

**Key Pattern:**
- **Field Width**: 16 characters per column (wider than filter pipeline)
- **Alignment**: Center-aligned
- **Purpose**: Different fields track ngram processing vs filtering

---

## 6. Display Utilities (`src/ngramprep/utilities/display.py`)

### Helper Functions

```python
def format_bytes(num_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

def truncate_path_to_fit(path: Union[Path, str], prefix: str, total_width: int = 100) -> str:
    """Truncate path to fit within total_width including prefix."""
    path_str = str(path)
    max_path_length = total_width - len(prefix)
    
    if len(path_str) <= max_path_length:
        return path_str
    
    if max_path_length < 4:
        return "..."
    
    return "..." + path_str[-(max_path_length - 3):]

def format_banner(title: str, width: int = 100, style: str = "═") -> str:
    """Create a formatted banner with title and separator line."""
    return f"{title}\n{style * width}"

def format_section_header(title: str, width: int = 100, style: str = "—") -> str:
    """Create a formatted section header (lighter style than banner)."""
    return format_banner(title, width, style)
```

**Output Examples:**
- **Bytes**: `1.00 KB`, `1.46 MB`, `5.00 GB`, `2.50 PB`
- **Path Truncation**: `/long/path` → `/long/path` (if fits) or `...to/file.db` (if truncated)
- **Banner**: 
  ```
  Pipeline Start
  ════════════════════════════════════════════════════════════════════════════════════════
  ```
- **Section Header**: 
  ```
  Configuration
  ────────────────────────────────────────────────────────────────────────────────────────
  ```

---

## Summary of Formatting Patterns

### Separator Characters
| Character | Name | Usage |
|-----------|------|-------|
| `━` | BOX DRAWINGS HEAVY HORIZONTAL | Major section separators (e.g., pipeline headers) |
| `═` | BOX DRAWINGS DOUBLE HORIZONTAL | Configuration/main section headers |
| `─` | BOX DRAWINGS LIGHT HORIZONTAL | Subsection headers and column separators |
| `│` | BOX DRAWINGS LIGHT VERTICAL | Box borders |
| `┌` `┐` `└` `┘` | Box corners | Summary boxes |
| `├` `┤` | Box intersections | Summary box separators |

### Data Row Formatting
- **Padded Labels**: Fixed-width label field (typically 22-26 characters)
- **Separator**: Colon followed by spaces
- **Values**: Right-aligned with consistent spacing
- **Example**: `Database:             /path/to/database.db`

### Column Formatting
- **Field Width**: 14 chars (filter) or 16 chars (pivot)
- **Alignment**: Center-aligned (`^` format specifier)
- **Spacing**: Maintained by joining formatted fields

### Number Formatting
- **Thousands Separator**: Used for large numbers (e.g., `10,000`)
- **Rates**: Abbreviated with `k/s` or `/s` (e.g., `1.2k/s`)
- **Percentages**: One decimal place (e.g., `75.0%`)
- **Sizes**: Two decimal places (e.g., `1.46 MB`)
- **Compressed/Abbreviated Numbers**: K/M/B with 2 decimal places (e.g., `1.23K`)

### Standard Line Width
- Default width: **100 characters**
- Used consistently across all separators and banners

### Hierarchy
1. **Major Banner** (━, style, all caps) - Top-level process
2. **Section Header** (═, mixed case) - Configuration/category
3. **Subsection** (─, mixed case) - Detailed settings
4. **Data Rows** - Key-value pairs with aligned spacing
5. **Blank Lines** - Used to separate sections

### Time Formatting
- **Timestamps**: ISO format `YYYY-MM-DD HH:MM:SS`
- **Durations**: Human-readable (e.g., `2h15m`, `5m30s`, `45s`)

