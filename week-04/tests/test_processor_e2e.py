"""
Integration tests for DocumentProcessor end-to-end workflow (Task 3, 7, 8).
"""

import os
import tempfile
import pytest
from database.db import DatabaseManager
from services.file_storage import FileStorageManager
from services.document_processor import DocumentProcessor
from src.classifier import DocumentClassifier
from src.utils import load_dataset_from_dir, generate_rich_sample_pdfs


@pytest.fixture
def integrated_processor():
    """Builds a temporary DocumentProcessor with isolated DB and Storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage_dir = os.path.join(tmpdir, "storage")
        samples_dir = os.path.join(tmpdir, "samples")

        db = DatabaseManager(db_path=db_path)
        storage = FileStorageManager(storage_dir=storage_dir)

        # Train a light classifier on project dataset
        data_train = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train")
        train_texts, train_labels = load_dataset_from_dir(data_train)
        clf = DocumentClassifier()
        if train_texts:
            clf.train(train_texts, train_labels)

        proc = DocumentProcessor(db_manager=db, storage_manager=storage, classifier=clf)

        # Generate sample files
        generate_rich_sample_pdfs(samples_dir)

        yield proc, samples_dir, db, storage


def test_e2e_invoice_workflow_and_persistence(integrated_processor):
    proc, samples_dir, db, storage = integrated_processor
    inv_path = os.path.join(samples_dir, "invoice_1.pdf")
    with open(inv_path, "rb") as f:
        file_bytes = f.read()

    # 1. First upload: Brand new file
    res = proc.process_and_store(file_bytes, "invoice_1.pdf")
    assert res["success"] is True
    assert res["is_duplicate"] is False
    assert res["document_type"] == "Invoice"
    assert res["status"] == "Processed"
    assert res["fields"]["Invoice Number"] == "INV-2026-001"
    assert res["fields"]["Total Amount"] == "$ 1,450.00"

    doc_id = res["document_id"]
    assert doc_id > 0

    # Verify physical file exists in storage
    stored_path = storage.get_full_path(res["file_path"])
    assert os.path.isfile(stored_path)

    # Verify database record exists
    db_record = db.get_document_by_id(doc_id)
    assert db_record is not None
    assert db_record["company"] == "TechCorp Solutions Inc."
    assert db_record["status"] == "Processed"

    # 2. Second upload: Identical file -> DUPLICATE DETECTION
    dup_res = proc.process_and_store(file_bytes, "invoice_1_copy.pdf")
    assert dup_res["success"] is True
    assert dup_res["is_duplicate"] is True
    assert dup_res["document"]["id"] == doc_id
    assert "Duplicate document detected" in dup_res["message"]

    # Verify no second record was created in database
    all_docs = db.get_all_documents()
    assert len(all_docs) == 1


def test_e2e_missing_field_needs_review(integrated_processor):
    proc, samples_dir, db, storage = integrated_processor
    missing_path = os.path.join(samples_dir, "invoice_missing_total.pdf")
    with open(missing_path, "rb") as f:
        file_bytes = f.read()

    res = proc.process_and_store(file_bytes, "invoice_missing_total.pdf")
    assert res["success"] is True
    assert res["is_duplicate"] is False
    assert res["status"] == "Needs Review"
    assert "Missing critical invoice field(s): Total Amount" in res["status_reason"]

    # Check that database also saved status as 'Needs Review'
    db_doc = db.get_document_by_id(res["document_id"])
    assert db_doc["status"] == "Needs Review"
