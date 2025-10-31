# ngram-prep

**Scalable tools to prepare Google Books Ngrams data for semantic analysis**

Process n-grams from the Google Books data repository using multiple CPUs. Ideal for large corpora consisting of millions or billions of ngrams. Provides efficient pipelines for filtering, transforming, and organizing n-gram data prior to analysis.

## Capabilities

- **Data acquisition:** Download n-gram datasets (1-grams, 2-grams, 3-grams, 4-grams, or 5-grams). Immediately ingest into a queryable RockDB database. 
- **Language support**. The pipeline work with any language supported by Google Books Ngrams: English, Chinese (simplified), French, German, Hebrew, Italian, Russian, and Spanish. 
- **Configurable processing:** Apply any or all of the following transformations: case normalization, stopword removal, short word removal, non-alphabetic token removal, and lemmatization. Discarded tokens are replaced in the corpus with `<UNK>`.
- **Whitelist creation:** Output the top-N most frequent unigrams (1-grams) for creating an eligible vocabulary list. Then use this list to efficiently filter multigrams (2- through 5-grams). Processing multigrams using a whitelist reduces processing time by bypassing other filters, ensures correctly spelled words, and, when used in conjunction with case normalization, discards many proper nouns (e.g., "jackson" will be discarded).
- **Temporal analysis support:** Reorganize data to a format suitable for time-series analyses:
  - BEFORE: `n-gram → (year1, count1, volumes1) (year2, count2, volumes2) ... (yearn, countn, volumesn)`
  - AFTER:
    - `[year1] n-gram → (count1, volumes1)`
    - `[year2] n-gram → (count2, volumes2)`
    - `...`
    - `[year3] n-gram → (countn, volumesn)`
- **High-throughput architecture:** Parallel processing with automatic load balancing, progress tracking, and resume capability.
- **Research-friendly storage:** Fast key-value database (RocksDB) quickly queries even very large datasets.

## Workflow

Three modules are provided for downloading and processing n-grams:

1. `ngram_acuire`: Fetch raw n-gram files from the Google Books repository and store in a RocksDB database for fast querying.
2. `ngram_filter`: Apply linguistic transformations (e.g., case normalization, lemmatization, and stopword removal) to focus on relevant data
3. `ngram_pivot`: Reorganize data from "wide" (per-ngram) to "long" (per-year) format for time-series analysis.

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

**Python Dependencies (installed automatically):**
- Python 3.11+
- `rocks-shim` for database access
- `Cython` for compilation
- `numpy` for Cython extensions
- `spacy` for lemmatization
- `spacy-lookups-data` for lemma lookup tables
- `stop-words` for stopword filtering
- `pyenchant` for spell checking
- `setproctitle` for process naming
- `tqdm` for progress display

**System Dependencies:**
- `libenchant` C library for spell checking (likely already installed on HPC clusters)
  - Ubuntu/Debian: `apt-get install libenchant-2-dev`
  - RHEL/CentOS: `yum install enchant-devel`
  - macOS: `brew install enchant`
- Compatible HPC systems: Works with Slurm job schedulers

## Quick Start

See the `notebooks/` directory for complete examples:

- **`download_unigrams.ipynb`** - Download and ingest 1-grams, apply filtering, generate vocabulary whitelist
- **`download_multigrams.ipynb`** - Download and filter 2-grams through 5-grams using whitelist
- **`pivot_unigrams.ipynb`** - Reorganize 1-grams for time-series analysis
- **`pivot_multigrams.ipynb`** - Reorganize multi-grams for temporal analysis
- **`train_word2vec.ipynb`** - Train word embeddings on processed n-grams

### Basic Usage

```python
from pathlib import Path
from ngram_prep.ngram_acquire import download_and_ingest_to_rocksdb
from ngram_prep.ngram_filter import PipelineConfig, FilterConfig, build_processed_db
from ngram_prep.ngram_pivot import run_pivot_pipeline
from ngram_prep.ngram_pivot.config import PipelineConfig as PivotConfig

# Step 1: Download and ingest n-grams
download_and_ingest_to_rocksdb(
    ngram_size=1,
    repo_release_id="20200217",
    repo_corpus_id="eng",
    db_path_stub="/data/ngrams",
    workers=30
)

# Step 2: Filter and clean
pipeline_config = PipelineConfig(
    src_db=Path("/data/ngrams/1grams.db"),
    dst_db=Path("/data/ngrams/1grams_processed.db"),
    tmp_dir=Path("/data/ngrams/tmp"),
    num_workers=40,
    mode="restart"
)

filter_config = FilterConfig(
    lowercase=True,
    filter_short=True,
    alpha_only=True
)

build_processed_db(pipeline_config, filter_config)

# Step 3: Pivot for time-series analysis (optional)
pivot_config = PivotConfig(
    src_db=Path("/data/ngrams/1grams_processed.db"),
    dst_db=Path("/data/ngrams/1grams_pivoted.db"),
    tmp_dir=Path("/data/ngrams/pivot_tmp"),
    num_workers=30,
    mode="restart"
)

run_pivot_pipeline(pivot_config)
```

See notebooks for detailed configuration options, temporal analysis workflows, and querying examples.

## Configuration Reference

### Filter Pipeline (`ngram_filter.PipelineConfig`)

**Core parameters:**
- `src_db`, `dst_db`: Input and output database paths
- `tmp_dir`: Working directory for temporary files
- `num_workers`: Number of parallel processing workers (default: 8, adjust based on CPU cores)
- `mode`: `"resume"` (continue interrupted jobs), `"restart"` (start fresh)

**Performance tuning:**
- `num_initial_work_units`: Initial data partitions for load balancing (default: same as `num_workers`)
- `max_split_depth`: How much to subdivide work for load balancing (default: 5)
- `compact_after_ingest`: Optimize final database size (recommended: `True` for long-term storage)
- `progress_every_s`: Progress update interval in seconds

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