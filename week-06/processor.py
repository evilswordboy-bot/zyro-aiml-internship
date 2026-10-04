"""
Top-level document processor interface module.
Exposes DocumentProcessor and get_document_processor.
"""

from services.document_processor import DocumentProcessor, get_document_processor

__all__ = ["DocumentProcessor", "get_document_processor"]
