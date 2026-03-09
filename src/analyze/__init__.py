"""
General-purpose analysis tools for word embeddings.

This module provides analysis tools that work with any word2vec embeddings,
regardless of whether they come from Google N-grams, Davies corpora (COHA/COCA),
or other sources.
"""

__all__ = [
    "aggregate_ipums_professions_csv",
    "aggregate_ipums_professions_csv_batch",
    "fetch_ipums_microdata_cps",
    "fetch_and_aggregate_ipums_professions_csv",
]