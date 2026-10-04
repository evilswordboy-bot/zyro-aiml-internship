"""
Service wrapper for RAG Engine and Document AI Assistant.
Coordinates database document loading, index management, and grounded query execution.
Week 6: RAG & Document AI Integration.
"""

from typing import Dict, Any, Optional, List
from src.rag_engine import RAGEngine, NO_INFO_RESPONSE
from database.db import DatabaseManager, get_db
import config


class RAGService:
    """High-level service managing RAG indexing and question answering."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db()
        self.engine = RAGEngine(
            chunk_size_words=config.RAG_CHUNK_SIZE_WORDS,
            chunk_overlap_words=config.RAG_CHUNK_OVERLAP_WORDS,
            similarity_threshold=config.RAG_SIMILARITY_THRESHOLD,
            top_k=config.RAG_TOP_K
        )
        self.indexed_doc_count = 0

    def sync_index_from_db(self) -> int:
        """Loads non-failed documents from SQLite and builds the vector index."""
        docs = self.db.get_all_documents_for_rag(limit=500)
        chunks_count = self.engine.build_index(docs)
        self.indexed_doc_count = len(docs)
        return chunks_count

    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Answers a user query using the grounded RAG engine.
        Ensures index is synchronized before answering.
        """
        if not self.engine.is_indexed:
            self.sync_index_from_db()

        return self.engine.answer_question(
            query=question,
            top_k=top_k,
            custom_threshold=config.RAG_SIMILARITY_THRESHOLD
        )

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the RAG index."""
        return {
            "is_indexed": self.engine.is_indexed,
            "total_chunks": len(self.engine.chunks),
            "documents_indexed": self.indexed_doc_count,
            "similarity_threshold": config.RAG_SIMILARITY_THRESHOLD,
            "top_k": config.RAG_TOP_K
        }


_rag_singleton: Optional[RAGService] = None


def get_rag_service(db_manager: Optional[DatabaseManager] = None) -> RAGService:
    """Returns singleton RAGService instance."""
    global _rag_singleton
    if db_manager is not None:
        return RAGService(db_manager)
    if _rag_singleton is None:
        _rag_singleton = RAGService()
    return _rag_singleton
