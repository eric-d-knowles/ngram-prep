# ngram-prep

End-to-end tools to acquire, filter, and store Google Books n‑grams at scale.

## Features
- **High‑throughput acquisition:** Parallel download of n‑gram files and direct ingestion into RocksDB
- **Resumable filtering pipeline:** Parallel processing with intelligent partitioning, work tracking, buffered writes, and progress reporting
- **Storage optimized for downstream use:** Shard outputs merged via streaming ingestion into a single RocksDB; optional post‑ingest compaction and whitelist generation
- **Flexible configuration:** Control concurrency, buffer sizes, batch sizes, I/O profiles, and ingestion behavior

## Installation

```bash pip install -e .```

## Quick Start

### 1. Download and ingest n‑grams

```
from ngram_acquire.pipeline.orchestrate import download_and_ingest_to_rocksdb
from ngram_acquire.pipeline.logger import setup_logger

download_and_ingest_to_rocksdb(
    ngram_size = 5,
    repo_release_id = "20200217",
    repo_corpus_id = "eng",
    db_path="/path/to/5grams.db",
    workers = 32,
    ngram_type = "tagged",
    open_type = "write:packed24",
    overwrite = True
)
```

### 2. Filter and merge into a final database

```
from nltk.corpus import stopwords; stopwords = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer; lemmatizer = WordNetLemmatizer()

from ngram_filter.config import PipelineConfig, FilterConfig
from ngram_filter.pipeline.orchestrator import build_processed_db

pcfg = PipelineConfig(
    src_db=Path("/path/to/5grams.db"),
    dst_db=Path("/path/to/5grams_processed.db"),
    tmp_dir=Path("/path/to/tmp"),
    readers=8,
    enable_ingest=True,
)

fcfg = FilterConfig(
    stop_set=stopwords,
    lemma_gen=lemmatizer,
    whitelist_path=wht_path
)

build_processed_db(pcfg, fcfg)
```

## Configuration Overview

`PipelineConfig` (selected):
- `src_db`, `dst_db`, `tmp_dir`: Input/output DBs and working directory
- `readers`, `work_units_per_reader`: Parallelism and work unit granularity
- `partitioning_sample_rate`, `prefix_length`: Key‑space partitioning
- `writer_read_profile`, `writer_write_profile`, `writer_disable_wal`: Filtering I/O
- `ingestors`, `ingest_*`: Streaming merge parameters
- `mode`, `force_cache_use`, `enable_ingest`, `enable_compact`, `delete_after_ingest`: Pipeline controls
- `progress_every_s`: Progress reporter cadence

`FilterConfig` (selected):
- `opt_lower`, `opt_alpha`, `opt_shorts`, `opt_stops`, `opt_lemmas`: Transformation switches
- `stop_set`, `lemma_gen`: Optional stopwords and lemmatizer
- `whitelist_path`, `whitelist_min_count`, `whitelist_top_n`: Input whitelist controls
- `vocab_path`: Optional vocabulary for shared memory‑mapped lookups

## Outputs
- **Final processed DB:** `dst_db`
- **Temporary artifacts:** `tmp_dir/worker_outputs` (shard DBs), `tmp_dir/work_tracker.db` (progress-tracking SQLite DB)
- **Optional whitelist:** Generated if configured

## Monitoring
Progress can be printed periodically (`progress_every_s`). Work status is tracked in `tmp_dir/work_tracker.db` (SQLite).
