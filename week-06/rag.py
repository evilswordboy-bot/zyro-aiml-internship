"""
Top-level Section 13 bridge module for RAG & AI Document Assistant.
Provides direct access to RAGService, get_rag_service, and RAGEngine.
"""

from services.rag import RAGService, get_rag_service
from src.rag_engine import RAGEngine, NO_INFO_RESPONSE

__all__ = [
    "RAGService",
    "get_rag_service",
    "RAGEngine",
    "NO_INFO_RESPONSE",
]
