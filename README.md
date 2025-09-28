# ngram-prep

End-to-end tools to acquire, filter, and store Google Books n‑grams at scale.

Features:
- High‑throughput acquisition: parallel download of n‑gram archives and direct ingestion into RocksDB.
- Resumable filtering pipeline: multi‑process processing with intelligent partitioning, work tracking, buffered writes, and progress reporting.
- Storage optimized for analytics: shard outputs merged via streaming ingestion into a single RocksDB; optional post‑ingest compaction.
- Flexible configuration: control concurrency, buffer sizes, batch sizes, I/O profiles, and ingestion behavior via a single pipeline config.

## Installation
Install in editable mode:
