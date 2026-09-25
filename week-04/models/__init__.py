"""
Models package for AI Document Intelligence & Workflow Platform.
"""

from .document import (
    DocumentRecord,
    STATUS_PROCESSED,
    STATUS_NEEDS_REVIEW,
    STATUS_FAILED,
    TYPE_INVOICE,
    TYPE_RESUME,
    TYPE_OTHER,
    ALL_DOCUMENT_TYPES,
    ALL_STATUSES
)

__all__ = [
    "DocumentRecord",
    "STATUS_PROCESSED",
    "STATUS_NEEDS_REVIEW",
    "STATUS_FAILED",
    "TYPE_INVOICE",
    "TYPE_RESUME",
    "TYPE_OTHER",
    "ALL_DOCUMENT_TYPES",
    "ALL_STATUSES"
]
