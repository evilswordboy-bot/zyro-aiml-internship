"""
Unit tests for SQLite Database Layer (database/db.py).
"""

import os
import tempfile
import pytest
from database.db import DatabaseManager


@pytest.fixture
def temp_db():
    """Provides a temporary, isolated SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_documents.db")
        db = DatabaseManager(db_path=db_path)
        yield db


def test_add_and_get_document(temp_db):
    doc_data = {
        "original_filename": "sample_invoice.pdf",
        "stored_filename": "invoice_20260925_123456_a1b2c3.pdf",
        "document_type": "Invoice",
        "upload_date": "2026-09-25T10:00:00",
        "company": "Acme Corp",
        "invoice_number": "INV-9901",
        "total_amount": "$ 500.00",
        "file_path": "invoices/invoice_20260925_123456_a1b2c3.pdf",
        "text_preview": "Acme Corp Invoice Total $500",
        "file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "status": "Processed",
        "status_reason": "All fields identified",
        "metadata_json": {"model": "Logistic Regression"}
    }
    doc_id = temp_db.add_document(doc_data)
    assert doc_id > 0

    retrieved = temp_db.get_document_by_id(doc_id)
    assert retrieved is not None
    assert retrieved["original_filename"] == "sample_invoice.pdf"
    assert retrieved["company"] == "Acme Corp"
    assert retrieved["invoice_number"] == "INV-9901"
    assert retrieved["total_amount"] == "$ 500.00"
    assert retrieved["file_hash"] == doc_data["file_hash"]
    assert retrieved["metadata"]["model"] == "Logistic Regression"


def test_duplicate_lookup_by_hash(temp_db):
    test_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    doc_data = {
        "original_filename": "contract.pdf",
        "stored_filename": "other_20260925_123456_abcdef.pdf",
        "document_type": "Other",
        "file_path": "other/contract.pdf",
        "text_preview": "Non-disclosure agreement",
        "file_hash": test_hash,
        "status": "Processed"
    }
    temp_db.add_document(doc_data)

    existing = temp_db.get_document_by_hash(test_hash)
    assert existing is not None
    assert existing["original_filename"] == "contract.pdf"

    non_existing = temp_db.get_document_by_hash("0000000000000000000000000000000000000000000000000000000000000000")
    assert non_existing is None


def test_multi_field_search(temp_db):
    temp_db.add_document({
        "original_filename": "alpha_bill.pdf",
        "stored_filename": "invoice_1.pdf",
        "document_type": "Invoice",
        "company": "Alpha Tech",
        "invoice_number": "INV-100",
        "file_path": "invoices/inv1.pdf",
        "text_preview": "Alpha Tech server bill",
        "file_hash": "hash1",
        "status": "Processed"
    })
    temp_db.add_document({
        "original_filename": "beta_resume.pdf",
        "stored_filename": "resume_1.pdf",
        "document_type": "Resume",
        "candidate_name": "Beta Developer",
        "file_path": "resumes/res1.pdf",
        "text_preview": "Python and Machine Learning engineer",
        "file_hash": "hash2",
        "status": "Processed"
    })

    # Search by company
    res1 = temp_db.search_documents(search_query="Alpha")
    assert len(res1) == 1
    assert res1[0]["company"] == "Alpha Tech"

    # Search by candidate name
    res2 = temp_db.search_documents(search_query="Beta Developer")
    assert len(res2) == 1
    assert res2[0]["candidate_name"] == "Beta Developer"

    # Search by text preview content
    res3 = temp_db.search_documents(search_query="Machine Learning")
    assert len(res3) == 1
    assert res3[0]["document_type"] == "Resume"

    # Filter by document_type
    res_inv = temp_db.search_documents(doc_type="Invoice")
    assert len(res_inv) == 1
    assert res_inv[0]["document_type"] == "Invoice"

    # Filter by non-existent status
    res_none = temp_db.search_documents(status="Failed")
    assert len(res_none) == 0


def test_kpi_statistics(temp_db):
    temp_db.add_document({
        "original_filename": "inv.pdf",
        "stored_filename": "inv.pdf",
        "document_type": "Invoice",
        "file_path": "invoices/inv.pdf",
        "text_preview": "inv",
        "file_hash": "h1",
        "status": "Processed"
    })
    temp_db.add_document({
        "original_filename": "res.pdf",
        "stored_filename": "res.pdf",
        "document_type": "Resume",
        "file_path": "resumes/res.pdf",
        "text_preview": "res",
        "file_hash": "h2",
        "status": "Needs Review"
    })
    temp_db.add_document({
        "original_filename": "bad.pdf",
        "stored_filename": "bad.pdf",
        "document_type": "Other",
        "file_path": "other/bad.pdf",
        "text_preview": "bad",
        "file_hash": "h3",
        "status": "Failed"
    })

    stats = temp_db.get_statistics()
    assert stats["total"] == 3
    assert stats["invoices"] == 1
    assert stats["resumes"] == 1
    assert stats["other"] == 1
    assert stats["processed"] == 1
    assert stats["needs_review"] == 1
    assert stats["failed"] == 1
