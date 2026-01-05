# lexichron

**Scalable tools for temporal linguistic analysis using Google Books Ngrams and Davies Corpora**

A comprehensive platform for semantic change research combining data preparation, text processing, and word embedding analysis. Supports both Google Ngrams (1-5 grams) and Mark Davies' corpora (COHA, COCA, etc.). Ideal for large datasets consisting of millions or billions of text examples. Provides efficient pipelines for acquiring, filtering, transforming, and organizing raw data—and for training and evaluating `word2vec` models to track semantic change over time.

While `lexichron` can be tuned to run on systems with fewer CPUs and less RAM, the package truly shines on High Performance Computing (HPC) or cloud infrastructures. Processing pipelines that might take days or weeks on a laptop can be completed in hours on a cluster or cloud platform.

## Citation

If you use lexichron in your research, please cite it:

```bibtex
@software{knowles2026lexichron,
  author = {Knowles, Eric D.},
  title = {lexichron: Scalable tools for temporal linguistic analysis},
  year = {2026},
  url = {https://github.com/eric-d-knowles/lexichron},
  version = {0.1.0}
}
```

Alternatively, you can click "Cite this repository" in the GitHub sidebar for additional citation formats.

## Capabilities

### Data Preparation
- **Data acquisition:** Download n-gram datasets (1- through 5-grams) or access Davies corpora. (Davies datasets must be licensed and downloaded by the user.) Immediately ingest data into a queryable RockDB database.
- **Language support**. N-gram pipelines support English, Chinese (simplified), French, German, Hebrew, Italian, Russian, and Spanish.
- **Configurable processing:** Apply any or all of the following transformations: case normalization, stopword removal, short word removal, non-alphabetic token removal, and lemmatization. Discarded tokens are replaced in the corpus with `<UNK>`.
- **Whitelist creation:** Output the top-N most frequent unigrams, applying optional spell-checking, then use this whitelist to efficiently filter text examples. Spell-checking discards proper nouns when used in conjunction with case normalization (e.g., "Jackson" and "Einstein" would be discarded). A year range can be defined to ensure that the whitelist contains only tokens found in all specified years. 
- **Bigram hyphenation:** Automatically convert semantically interesting bigrams into hyphenated unigrams (e.g., "working class" → "working-class", "nuclear family" → "nuclear-family"), preserving multiword concepts as single tokens for downstream analysis.
- **Token immunity:** Define tokens that should always be preserved during filtering, immune to exclusion rules. Useful for domain-specific terms, names or proper nouns, historical keywords, or particular multiword expressions that you want to ensure remain in your corpus regardless of other filtering criteria.
- **Temporal analysis support:** Reorganize n-gram data into a format suitable for time-series analyses:
  - BEFORE: `n-gram → (year1, count1, volumes1) (year2, count2, volumes2) ... (yearn, countn, volumesn)`
  - AFTER:
    - `[year1] n-gram → (count1, volumes1)`
    - `[year2] n-gram → (count2, volumes2)`
    - `...`
    - `[year3] n-gram → (countn, volumesn)`
- **High-throughput architecture:** Parallel processing with load balancing, progress tracking, and resume capability in the event of interruption.
- **Research-friendly storage:** Fast key-value database (RocksDB) quickly queries even enormous datasets.

### Model Training and Evaluation
- **Word embeddings:** Train `word2vec` models on the processed n-grams using `gensim`'s implementation. Optionally use `corpus_file` mode to enable fast, multithreaded training and training multiple years at once. Easily adjust model hyperparameters:
  - `approach`: use skip-gram or continuous bag-of-words (CBOW) architectures
  - `vector_size`: the number vector dimensions (features) to extract
  - `window size`: the width of the context window
  - `min_count`: the minimum frequency of words to include in the model
  - `weight_by`: downweight common ngrams by frequency or document count
- **Evaluation:** Evaluate the performance of the trained model using standard intrinsic tests of similarity and analogy performance. Plot the results for visual comparison of model quality. Use mixed-model regression to quantify the impact of different hyperparameters on model performance across years.

## Workflow

The toolkit provides two parallel pipelines for different data sources:

### Google Ngrams Pipeline

1. **`ngram_acquire`**: Fetch raw n-gram files (1-5 grams) from the Google Books repository and store in a RocksDB database for fast querying.
2. **`ngram_filter`**: Apply linguistic transformations (case normalization, lemmatization, stopword removal, spell-checking, bigram hyphenation) to prepare data. Optionally generate vocabulary whitelists.
3. **`ngram_pivot`**: Reorganize data from "wide" (per-ngram) to "long" (per-year) format for time-series analysis.
4. **`ngram_analyze`**: Track semantic drift and similarity changes across time using trained word embeddings.

### Davies Corpora Pipeline

1. **`davies_acquire`**: Ingest Davies corpus files (COHA, COCA, etc.) with genre and year information into RocksDB.
2. **`davies_filter`**: Apply the same filtering and preprocessing transformations as ngram_filter for consistency.

### Model Training

**`train/word2vec`**: Train per-year word2vec models, evaluate across intrinsic benchmarks, align models across years, and analyze hyperparameter impact via regression.

## System Requirements

- HPC cluster or workstation with multiple CPU cores (30+ cores recommended)
- Large amount of RAM (80+ GB recommended)
- Fast local storage (NVMe SSD recommended)
- Several TB of disk space for processing and storing very large corpora
- Settings can be tuned for fewer resources, but at the cost of processing speed

## Installation

### Prerequisites

- Git
- Conda or Miniconda

### Conda Setup (Recommended)

**Step 1: Clone the repository**

```bash
git clone https://github.com/eric-d-knowles/lexichron.git
cd lexichron
```

**Step 2: Create and activate the conda environment**

```bash
# Create the conda environment from the provided environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate lexichron
```

**Step 2b: Install hunspell dictionaries for spell-checking**

The environment includes `enchant` for spell-checking support, but you also need hunspell dictionaries:

```bash
# Minimal installation (English only)
conda install -c conda-forge enchant hunspell-en

# Or install all languages supported by the ngram pipeline (recommended)
conda install -c conda-forge enchant hunspell-en hunspell-fr hunspell-de hunspell-he hunspell-it hunspell-ru hunspell-es hunspell-zh
```

**Step 3: Install the package**

```bash
# Install in editable mode from the project root
pip install -e .
```

The `-e` flag installs the package in editable mode, so changes to the source code are immediately available.

**Automatic initialization:** On first import, the package will automatically download any missing spaCy language models and configure enchant library paths.

**Step 4: Register the Jupyter kernel**

To use the notebooks in the `notebooks/` directory, register the conda environment as a Jupyter kernel:

```bash
# Register the kernel (while the conda environment is active)
python -m ipykernel install --user --name=lexichron --display-name="Python (lexichron)"
```

Now when you launch Jupyter, you'll be able to select the "Python (lexichron)" kernel for your notebooks.

### Alternative: Apptainer Container (For HPC)

If you prefer a containerized approach for HPC clusters with GPU support:

```bash
# Clone the repository first
git clone https://github.com/eric-d-knowles/lexichron.git
cd lexichron

# Build the container on a compute node
./build/build_container.sh

# Install the package (run once, from the project root)
apptainer exec --nv lexichron.sif pip install -e .

# Run notebooks or scripts with the container
apptainer exec --nv lexichron.sif jupyter notebook
apptainer exec --nv lexichron.sif python your_script.py
```

The container includes CUDA 12.6.2, cuDNN, and all system dependencies pre-installed.

## Quick Start

See the `notebooks/` directory for complete workflow examples:

### Google Ngrams Workflows

- **`eng_unigrams_workflow.ipynb`** - Download and ingest 1-grams, apply filtering and preprocessing, generate vocabulary whitelist (English)
- **`eng_multigrams_workflow.ipynb`** - Download and filter 2-5 grams using whitelist (English)
- **`rus_unigrams_workflow.ipynb`** - Same as English unigrams but for Russian
- **`rus_multigrams_workflow.ipynb`** - Same as English multigrams but for Russian
- **`ngrams_change_analysis_workflow.ipynb`** - Analyze semantic drift and track meaning changes over time

### Davies Corpora Workflows

- **`davies_acquisition_workflow.ipynb`** - Ingest Davies corpus files with genre and year information
- **`coha_training_workflow.ipynb`** - Train word2vec models on COHA corpus data
- **`coha_change_analysis_workflow.ipynb`** - Analyze semantic change in historical English (COHA)

### Model Training & Evaluation

- **`training_workflow.ipynb`** - Train word embeddings on processed n-grams
- **`ngram_training_workflow.ipynb`** - End-to-end word2vec training pipeline for n-grams

### Basic Usage Example

```python
from pathlib import Path
from ngramprep.ngram_acquire import download_and_ingest_to_rocksdb
from ngramprep.ngram_filter import PipelineConfig, FilterConfig, build_processed_db
from ngramprep.ngram_pivot import run_pivot_pipeline
from ngramprep.ngram_pivot.config import PipelineConfig as PivotConfig

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

For detailed configuration options, see the docstrings in `ngramprep.ngram_filter.config`, `ngramprep.ngram_pivot.config`, and `daviesprep.davies_filter.config`, or refer to the example notebooks.

## Output Files

After running the pipelines, you'll have:

- **Final database** (`dst_db`): Query-ready RocksDB containing your processed n-grams
- **Frequency whitelist** (optional): Text file listing retained n-grams with occurrence counts (useful for documenting your corpus)
- **Compressed archive** (optional): Use `common_db.compress_db()` for efficient long-term storage and transfer

**Temporary files** (can be deleted after completion):
- `tmp_dir/worker_outputs/`: Intermediate processing shards
- `tmp_dir/work_tracker.db`: Progress tracking database (useful for debugging interrupted jobs)

## Advanced: Monitoring and Architecture

### Real-time Progress Display

The ngram_filter and ngram_pivot pipelines print periodic updates showing:

```
      items         kept%         workers         units          rate          elapsed
──────────────────────────────────────────────────────────────────────────────────────────
    128.56M         85.4%          8/40          10·24·1237     214.2k/s        10m00s
```

**Column meanings:**
- **items**: Total records processed so far
- **kept%**: Percentage of n-grams retained after filtering (100% for pivot)
- **workers**: Active workers / total workers (shows load distribution)
- **units**: Work distribution status as `pending·processing·completed` (shows load balancing)
- **rate**: Processing throughput (records per second)
- **elapsed**: Total time since pipeline started

### Two-Stage Pipeline Architecture

The ngram_filter and ngram_pivot pipelines use a two-phase design for memory efficiency and fault tolerance:

1. **Processing stage**: Workers divide the input data into chunks, process them in parallel, and write results to temporary files (`tmp_dir/worker_outputs/`)
2. **Ingestion stage**: Temporary files are merged into the final database using parallel streaming

This design enables:
- **Resume capability**: Interrupted jobs pick up where they left off
 - **Load balancing**: Work units are pre-balanced via density-based sampling; workers steal remaining units as they finish
- **Balanced work units**: Density-based sampling scans the corpus to estimate token frequency distributions, then partitions work so each unit has similar total token mass, reducing straggler workers and keeping throughput consistent
- **Memory efficiency**: Large datasets don't need to fit in RAM
- **Predictable resource usage**: Memory consumption is bounded regardless of corpus size

*Note: Davies acquisition pipelines use simpler direct ingestion and do not employ the two-stage architecture.*

## Support and Maintenance

This project is provided as-is for research and development purposes. While issues and pull requests are welcome, there is no guarantee of response time or ongoing maintenance. The code is shared in the spirit of open science, but support is provided on a best-effort basis only.

For critical production use, consider forking and maintaining your own version.