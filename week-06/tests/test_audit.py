"""
Unit tests for AuditService (Week 5).
Validates immutable audit logging, event retrieval, chronological ordering, and timeline feeds.
"""

import os
import tempfile
import pytest
from database.db import DatabaseManager
from services.audit import (
    AuditService,
    ACTION_UPLOADED,
    ACTION_PROCESSING_STARTED,
    ACTION_CLASSIFIED,
    ACTION_EXTRACTED,
    ACTION_VALIDATION_PASSED,
    ACTION_ROUTED_TO_COMPLETED,
    ACTION_REVIEWER_APPROVED
)


@pytest.fixture
def audit_setup():
    """Builds isolated DatabaseManager and AuditService."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_audit.db")
        db = DatabaseManager(db_path=db_path)
        audit = AuditService(db)
        yield db, audit


def test_log_and_retrieve_document_history(audit_setup):
    db, audit = audit_setup

    doc_id = db.add_document({
        "original_filename": "sample_invoice.pdf",
        "stored_filename": "sample_invoice.pdf",
        "document_type": "Invoice",
        "status": "New",
        "file_hash": "audit_hash_1"
    })

    # Record full lifecycle sequence
    audit.record_event(doc_id, ACTION_UPLOADED, None, "New", "File uploaded")
    audit.record_event(doc_id, ACTION_PROCESSING_STARTED, "New", "Processing", "OCR started")
    audit.record_event(doc_id, ACTION_CLASSIFIED, "Processing", "Processing", "Classified as Invoice")
    audit.record_event(doc_id, ACTION_EXTRACTED, "Processing", "Processing", "Fields extracted")
    audit.record_event(doc_id, ACTION_VALIDATION_PASSED, "Processing", "Processing", "Valid fields")
    audit.record_event(doc_id, ACTION_ROUTED_TO_COMPLETED, "Processing", "Completed", "Rules verified")

    history = audit.get_document_history(doc_id)
    assert len(history) == 6
    assert history[0]["action"] == ACTION_UPLOADED
    assert history[-1]["action"] == ACTION_ROUTED_TO_COMPLETED
    assert history[-1]["new_status"] == "Completed"


def test_recent_timeline_joined_data(audit_setup):
    db, audit = audit_setup

    doc_id1 = db.add_document({
        "original_filename": "doc1.pdf",
        "stored_filename": "doc1.pdf",
        "document_type": "Invoice",
        "status": "Completed",
        "file_hash": "audit_hash_2"
    })
    doc_id2 = db.add_document({
        "original_filename": "doc2.pdf",
        "stored_filename": "doc2.pdf",
        "document_type": "Resume",
        "status": "Completed",
        "file_hash": "audit_hash_3"
    })

    audit.record_event(doc_id1, ACTION_UPLOADED, None, "New", "Uploaded doc1")
    audit.record_event(doc_id2, ACTION_UPLOADED, None, "New", "Uploaded doc2")

    recent = audit.get_recent_timeline(limit=10)
    assert len(recent) == 2
    # Check that document metadata is joined properly
    filenames = [r["original_filename"] for r in recent]
    assert "doc1.pdf" in filenames
    assert "doc2.pdf" in filenames
