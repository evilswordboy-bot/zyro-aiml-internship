"""
Top-level batch processing interface module.
Exposes BatchProcessor and get_batch_processor for parallel fault-tolerant processing.
"""

from services.batch import BatchProcessor, get_batch_processor

__all__ = ["BatchProcessor", "get_batch_processor"]
