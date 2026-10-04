"""
Unit and integration tests for RAG Engine and Document AI Assistant.
Week 6: Grounded QA, Source Attribution, and Anti-Hallucination Guardrails (Section 7, Tests 13 & 14).
"""

import pytest
from src.rag_engine import RAGEngine, NO_INFO_RESPONSE
from services.rag import RAGService
from database.db import DatabaseManager
import tempfile
import os


@pytest.fixture
def sample_corpus():
    """Provides authentic document samples for RAG testing."""
    return [
        {
            "id": 1,
            "original_filename": "invoice_techcorp.pdf",
            "document_type": "Invoice",
            "text_preview": "TechCorp Solutions Inc. Invoice INV-2026-001. Bill to Global Enterprises. Total Amount: $ 1,450.00 for Cloud Architecture Consulting.",
            "raw_text": "TechCorp Solutions Inc. Invoice INV-2026-001. Bill to Global Enterprises. Total Amount: $ 1,450.00 for Cloud Architecture Consulting."
        },
        {
            "id": 2,
            "original_filename": "resume_priya.pdf",
            "document_type": "Resume",
            "text_preview": "Priya Sharma Senior ML Engineer. Email: priya.sharma@aimlconsulting.io. Skills: Python, Scikit-learn, TensorFlow, PyTorch, Kubernetes. 5 years experience.",
            "raw_text": "Priya Sharma Senior ML Engineer. Email: priya.sharma@aimlconsulting.io. Skills: Python, Scikit-learn, TensorFlow, PyTorch, Kubernetes. 5 years experience."
        },
        {
            "id": 3,
            "original_filename": "nda_agreement.pdf",
            "document_type": "Other",
            "text_preview": "Mutual Non-Disclosure Agreement (NDA) between Alpha Corp and Beta Systems. The duration and term of obligation is three years from execution date.",
            "raw_text": "Mutual Non-Disclosure Agreement (NDA) between Alpha Corp and Beta Systems. The duration and term of obligation is three years from execution date."
        }
    ]


def test_chunking_mechanism():
    """Verifies text is chunked with proper word sizing and overlap."""
    engine = RAGEngine(chunk_size_words=20, chunk_overlap_words=5)
    long_text = "word " * 55
    chunks = engine.chunk_document(doc_id=10, filename="test.pdf", doc_type="Other", text=long_text)

    assert len(chunks) >= 3
    assert all(c.doc_id == 10 for c in chunks)
    assert all(c.filename == "test.pdf" for c in chunks)
    assert chunks[0].chunk_id == 0
    assert chunks[1].chunk_id == 1


def test_build_index_and_retrieve(sample_corpus):
    """Verifies vector space indexing and cosine similarity retrieval."""
    engine = RAGEngine()
    count = engine.build_index(sample_corpus)
    assert count == 3
    assert engine.is_indexed is True

    # Retrieve relevant chunk for cloud consulting
    results = engine.retrieve("Cloud Architecture Consulting", top_k=2)
    assert len(results) >= 1
    top_chunk, score = results[0]
    assert "TechCorp" in top_chunk.text
    assert top_chunk.filename == "invoice_techcorp.pdf"
    assert score > 0.20


def test_grounded_answer_with_source_attribution(sample_corpus):
    """Test 13: Grounded question answering returns accurate answer and source citations."""
    engine = RAGEngine()
    engine.build_index(sample_corpus)

    q = "What are the technical skills of Priya Sharma?"
    res = engine.answer_question(q)

    assert res["answered"] is True
    assert res["status"] == "GROUNDED_ANSWER"
    assert len(res["sources"]) > 0
    assert "resume_priya.pdf" in res["sources"][0]["filename"]
    assert "Python" in res["answer"] or "TensorFlow" in res["answer"]
    assert res["confidence"] > 0.15


def test_insufficient_information_guardrail(sample_corpus):
    """Test 14: Unanswerable question strictly returns insufficient information without hallucinating."""
    engine = RAGEngine()
    engine.build_index(sample_corpus)

    unrelated_q = "What is the orbital velocity of the International Space Station in low Earth orbit?"
    res = engine.answer_question(unrelated_q)

    assert res["answered"] is False
    assert res["status"] == "INSUFFICIENT_INFORMATION"
    assert res["answer"] == NO_INFO_RESPONSE
    assert len(res["sources"]) == 0


def test_empty_corpus_handling():
    """Answering against an empty corpus safely returns insufficient info."""
    engine = RAGEngine()
    engine.build_index([])
    res = engine.answer_question("Where is the invoice?")
    assert res["answered"] is False
    assert res["answer"] == NO_INFO_RESPONSE


def test_rag_service_with_database(sample_corpus):
    """Verifies RAGService automatically syncs documents from SQLite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_rag.db")
        db = DatabaseManager(db_path)
        for doc in sample_corpus:
            db.add_document({
                "original_filename": doc["original_filename"],
                "stored_filename": doc["original_filename"],
                "document_type": doc["document_type"],
                "file_hash": f"hash_{doc['id']}",
                "status": "Completed",
                "current_status": "Completed",
                "file_path": f"storage/{doc['original_filename']}",
                "text_preview": doc["text_preview"],
                "raw_text": doc["raw_text"]
            })

        service = RAGService(db_manager=db)
        status = service.get_status()
        assert status["total_chunks"] == 0  # not indexed yet

        res = service.ask("What is the NDA duration?")
        assert res["answered"] is True
        assert "three years" in res["answer"].lower()
        assert service.get_status()["documents_indexed"] == 3
