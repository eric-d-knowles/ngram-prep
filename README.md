# ngram-prep

**Scalable tools to prepare Google Books Ngrams data for linguistic and cultural analysis**

Process n-grams from the Google Books corpus using multiple CPUs. Ideal for large corpora consisting of millions and billions of entries. Provides efficient pipelines for filtering, transforming, and organizing n-gram data prior to analysis.

## Capabilities

- **Automated data acquisition:** Download and organize Google Books n-gram datasets (1-grams through 5-grams, multiple languages)
- **Flexible text processing:** Apply linguistic transformations (lemmatization, stopword removal, case normalization)
- **Temporal analysis support:** Reorganize data to a format suitable for time-series analysis 
  - FROM: `n-gram → (year1, count1, volumes1) (year2, count2, volumes2) ... (yearn, countn, volumesn)`
  - TO:
    - `[year1] n-gram → (count1, volumes1)`
    - `[year2] n-gram → (count2, volumes2)`
    - `...`
    - `[year3] n-gram → (countn, volumesn)`
- **High-throughput architecture:** Parallel processing with automatic load balancing, progress tracking, and resume capability
- **Research-friendly storage:** Fast key-value database (RocksDB) with efficient compression for long-term storage

## Workflow

This package provides three sequential pipeline stages:

1. **Acquire**: Download raw n-gram files from Google Books Ngrams and store in a queryable database
2. **Filter**: Apply linguistic transformations (cleaning, lemmatization, stopword removal) to focus on relevant data
3. **Pivot**: Reorganize data by year for time-series analysis (e.g., query all 5-grams containing "climate change" in 1995)

## System Requirements

- HPC cluster or workstation with multiple CPU cores (30+ cores recommended)
- Large amount of RAM (80+ GB recommended)
- Fast local storage (NVMe SSD recommended)
- Several TB of disk space for processing and storing very large corpora
- Settings can be tuned for fewer resources, but at the cost of processing speed

## Installation

```bash
pip install git+https://github.com/eric-d-knowles/ngram-prep.git
```

**Dependencies:**
- Python 3.11+
- `rocks-shim` for database access
- `nltk` (for stopwords and lemmatization)
- Common HPC systems: Works with Slurm job schedulers

## Quick Start

#### Setup

```python
# Auto-reload edited scripts (useful during development)
%load_ext autoreload
%autoreload 2

# NLTK resources for text processing
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Import ngram_prep modules
from pathlib import Path
from ngram_prep.ngram_acquire import download_and_ingest_to_rocksdb
from ngram_prep.ngram_acquire.logger import setup_logger
from ngram_prep.ngram_filter import PipelineConfig, FilterConfig, build_processed_db
from ngram_prep.ngram_pivot import run_pivot_pipeline
from ngram_prep.ngram_pivot.config import PipelineConfig as PivotConfig
from ngram_prep.utilities.peek import db_head, db_peek, db_peek_prefix

# Setup logging (optional but recommended)
setup_logger(
    db_path="/data/ngrams/5grams.db",
    console=False,
    rotate=True,
    max_bytes=100_000_000,
    backup_count=5,
    force=True
)
```

#### Step 1: Download and ingest 5-grams

```python
download_and_ingest_to_rocksdb(
    ngram_size=5,
    repo_release_id="20200217",
    repo_corpus_id="eng",
    db_path_stub="/data/ngrams",
    file_range=(0, 19422),           # Process all files (or specify subset for testing)
    random_seed=76,                  # Randomize download order
    workers=30,
    use_threads=False,
    ngram_type="tagged",             # Include part-of-speech tags
    overwrite_db=False,              # Set True to start fresh
    write_batch_size=1_000_000,
    open_type="write:packed24",
    compact_after_ingest=True
)
```

#### Step 2: Filter and clean the data

```python
# Set up paths
src_db = Path("/data/ngrams/5grams.db")
dst_db = src_db.parent / "5grams_processed.db"
tmp_dir = src_db.parent / "processing_tmp"

# Optional: use a whitelist from unigrams to filter vocabulary
whitelist_path = Path("/data/ngrams/1grams_processed.db/whitelist.txt")

# Configure filtering
filter_config = FilterConfig(
    stop_set=stopwords,
    lemma_gen=lemmatizer,
    whitelist_path=whitelist_path    # Optional: pre-filter vocabulary
)

# Configure pipeline
pipeline_config = PipelineConfig(
    src_db=src_db,
    dst_db=dst_db,
    tmp_dir=tmp_dir,
    num_workers=40,
    num_initial_work_units=80,
    work_unit_claim_order="sequential",
    max_split_depth=100,
    split_check_interval_s=45.0,
    mode="restart",                  # Or "resume" to continue interrupted jobs
    progress_every_s=15.0,
    max_items_per_bucket=10_000_000,
    max_bytes_per_bucket=512 * 1024 * 1024,
    ingest_num_readers=40,
    ingest_batch_items=5_000_000,
    ingest_queue_size=1
)

build_processed_db(pipeline_config, filter_config)
```

#### Step 3: Pivot for temporal analysis (optional)

```python
# This transforms the data structure from:
#   Key: "climate change mitigation" → Values: (1995, 42, 38), (1996, 58, 51), ... (year, count, volumes)
# To:
#   Key: "[1995] climate change mitigation" → Value: (42, 38)
#   Key: "[1996] climate change mitigation" → Value: (58, 51)

pipeline_cfg = PivotConfig(
    src_db=Path("/data/ngrams/5grams_processed.db"),
    dst_db=Path("/data/ngrams/5grams_pivoted.db"),
    tmp_dir=Path("/data/ngrams/pivot_tmp"),
    num_workers=30,
    num_initial_work_units=40,
    max_split_depth=100,
    work_unit_claim_order="random",
    split_check_interval_s=15.0,
    progress_every_s=600.0,
    mode="restart",
    max_items_per_bucket=50_000_000,
    max_bytes_per_bucket=5 * 1024 * 1024 * 1024,
    num_ingest_readers=3,
    ingest_buffer_shards=1
)

run_pivot_pipeline(pipeline_cfg)
```

#### Querying the results

```python
db_path = "/data/ngrams/5grams_pivoted.db"

# View sample entries
db_head(db_path, n=5)

# Search for specific phrases in a given year
db_peek(db_path, start_key="[2002] your search term", n=5)

# Find all 5-grams with a prefix
db_peek_prefix(db_path, prefix="[2018] your prefix", n=10)
```

## Configuration Reference

### Filter Pipeline (`ngram_filter.PipelineConfig`)

**Core parameters:**
- `src_db`, `dst_db`: Input and output database paths
- `tmp_dir`: Working directory for temporary files
- `num_workers`: Number of parallel processing workers (default: 8, adjust based on CPU cores)
- `mode`: `"resume"` (continue interrupted jobs), `"restart"` (start fresh), or `"reprocess"` (rerun all)

**Performance tuning:**
- `num_initial_work_units`: Initial data partitions for load balancing (default: same as `num_workers`)
- `max_split_depth`: How much to subdivide work for load balancing (default: 5)
- `compact_after_ingest`: Optimize final database size (recommended: `True` for long-term storage)
- `progress_every_s`: Progress update interval in seconds (increase for Slurm jobs)

**Advanced options:**
- `writer_read_profile`, `writer_write_profile`: Database I/O optimization profiles
- `ingest_num_readers`, `ingest_queue_size`: Control memory vs. throughput during final merge
- `output_whitelist_path`: Generate frequency list of retained n-grams

### Filter Configuration (`ngram_filter.FilterConfig`)

**Text transformations (all optional):**
- `lowercase`: Normalize to lowercase (recommended for most analyses)
- `alpha_only`: Remove punctuation and numbers (keep only letters)
- `filter_short`: Remove very short tokens (default: < 3 characters)
- `filter_stops`: Remove common function words (requires `stop_set`)
- `apply_lemmatization`: Reduce words to root forms—e.g., "running"→"run" (requires `lemma_gen`)

**Resources:**
- `stop_set`: Set of stopwords to remove (e.g., `set(stopwords.words("english"))` from NLTK)
- `lemma_gen`: Lemmatizer object (e.g., `WordNetLemmatizer()` from NLTK)
- `whitelist_path`: Pre-filter input using existing vocabulary list (saves processing time)

### Pivot Pipeline (`ngram_pivot.PipelineConfig`)

**Core parameters:**
- `src_db`, `dst_db`: Input and output database paths
- `tmp_dir`: Working directory for temporary files
- `num_workers`: Number of parallel workers (typically higher than filter, e.g., 30)
- `mode`: `"resume"`, `"restart"`, or `"reprocess"`

**Performance tuning:**
- `num_initial_work_units`: Initial partitions (often higher than workers, e.g., 40-80)
- `max_split_depth`: Maximum subdivisions for load balancing (default: 5, increase for very large datasets)
- `enable_ingest`: Merge temporary files into final database (set `False` to inspect intermediate outputs)
- `compact_after_ingest`: Optimize database size (recommended for production use)

## Output Files

After running the pipelines, you'll have:

- **Final database** (`dst_db`): Query-ready RocksDB containing your processed n-grams
- **Frequency whitelist** (optional): Text file listing retained n-grams with occurrence counts (useful for documenting your corpus)
- **Compressed archive** (optional): Use `common_db.compress_db()` for efficient long-term storage and transfer

**Temporary files** (can be deleted after completion):
- `tmp_dir/worker_outputs/`: Intermediate processing shards
- `tmp_dir/work_tracker.db`: Progress tracking database (useful for debugging interrupted jobs)

## Monitoring Progress

### Real-time Progress Display

The pipelines print periodic updates showing:

```
     ngrams           exp            units           splits           rate          elapsed
────────────────────────────────────────────────────────────────────────────────────────────────
    128.56M          42.3x          310·24·1237        1260          214.2k/s         10m00s
```

**What each column means:**
- **ngrams/items**: Total records processed so far
- **exp/kept%**: Data expansion ratio (pivot) or percentage of n-grams retained after filtering
- **units**: Work distribution status as `pending·processing·completed` (shows load balancing)
- **splits**: Number of times work was subdivided to balance load across workers
- **rate**: Processing throughput (n-grams per second)
- **elapsed**: Total time since pipeline started

### Checking Status Manually

For long-running jobs on HPC clusters, check progress from the command line:

```bash
# From your tmp_dir location
python3 -c "
import sqlite3
conn = sqlite3.connect('work_tracker.db')
cur = conn.cursor()
results = cur.execute('SELECT status, COUNT(*) FROM work_units GROUP BY status').fetchall()
status_dict = dict(results)
total = sum(status_dict.values())
completed = status_dict.get('completed', 0)
print(f'Progress: {completed}/{total} work units completed ({100*completed/total:.1f}%)')
print(f'Pending: {status_dict.get(\"pending\", 0)}, Processing: {status_dict.get(\"processing\", 0)}')
"
```

### Understanding the Two-Stage Architecture

Both filter and pivot pipelines work in two phases:

1. **Processing stage**: Workers divide the input data into chunks, process them in parallel, and write results to temporary files (`tmp_dir/worker_outputs/`)
2. **Ingestion stage**: Temporary files are merged into the final database using parallel streaming (controlled by `enable_ingest`)

This design enables:
- **Resume capability**: Interrupted jobs pick up where they left off
- **Load balancing**: Work units automatically split when some workers finish early
- **Memory efficiency**: Large datasets don't need to fit in RAM