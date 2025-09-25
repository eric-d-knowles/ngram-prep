# ngram_filter/partitioning/__init__.py
"""
Work unit partitioning strategies for ngram processing.
"""

from .intelligent import (
    IntelligentPartitioner,
    create_intelligent_work_units,
)

__all__ = [
    "IntelligentPartitioner",
    "create_intelligent_work_units",
]