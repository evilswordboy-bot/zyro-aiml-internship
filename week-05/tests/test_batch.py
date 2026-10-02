"""
Unit tests for BatchProcessor (Week 5).
Validates multi-document processing, progress tracking, and critical per-document fault isolation.
"""

import os
import tempfile
import pytest
from database.db import DatabaseManager
from services.file_storage import FileStorageManager
from services.document_processor import DocumentProcessor
from services.batch import BatchProcessor
from src.classifier import DocumentClassifier
from src.utils import generate_rich_sample_pdfs, load_dataset_from_dir


@pytest.fixture
def batch_setup():
    """Builds isolated DocumentProcessor and BatchProcessor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_batch.db")
        storage_dir = os.path.join(tmpdir, "storage")
        samples_dir = os.path.join(tmpdir, "samples")

        db = DatabaseManager(db_path=db_path)
        storage = FileStorageManager(storage_dir=storage_dir)

        data_train = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train")
        train_texts, train_labels = load_dataset_from_dir(data_train)
        clf = DocumentClassifier()
        if train_texts:
            clf.train(train_texts, train_labels)

        processor = DocumentProcessor(db_manager=db, storage_manager=storage, classifier=clf)
        batch_proc = BatchProcessor(processor=processor)

        generate_rich_sample_pdfs(samples_dir)

        yield batch_proc, samples_dir, db


def test_batch_processing_success(batch_setup):
    batch_proc, samples_dir, db = batch_setup

    files = []
    for name in ["invoice_1.pdf", "resume_1.pdf", "meeting_minutes.pdf"]:
        path = os.path.join(samples_dir, name)
        with open(path, "rb") as f:
            files.append((name, f.read()))

    progress_events = []

    def on_progress(idx, total, fname, status):
        progress_events.append((idx, total, fname, status))

    summary = batch_proc.process_batch(files, progress_callback=on_progress)

    assert summary["total"] == 3
    assert len(summary["results"]) == 3
    assert summary["failed"] == 0
    assert len(progress_events) > 0


def test_batch_isolated_error_fault_tolerance(batch_setup):
    """
    CRITICAL REQUIREMENT:
    Verify that an error in one document (e.g. corrupt byte payload or invalid format)
    never halts, stops, or crashes processing of subsequent documents in the batch.
    """
    batch_proc, samples_dir, db = batch_setup

    with open(os.path.join(samples_dir, "invoice_1.pdf"), "rb") as f:
        good_invoice = f.read()

    with open(os.path.join(samples_dir, "resume_1.pdf"), "rb") as f:
        good_resume = f.read()

    # Corrupt / invalid document payload
    corrupt_pdf = b"%PDF-1.4 Corrupted content with no EOF marker or catalog dictionary"

    batch_files = [
        ("first_good_invoice.pdf", good_invoice),
        ("corrupt_doc.pdf", corrupt_pdf),
        ("second_good_resume.pdf", good_resume)
    ]

    summary = batch_proc.process_batch(batch_files)

    # The batch must complete all 3 items without crashing
    assert summary["total"] == 3
    assert len(summary["results"]) == 3

    # First and third documents must have processed successfully
    res1 = summary["results"][0]
    res2 = summary["results"][1]
    res3 = summary["results"][2]

    assert res1["success"] is True
    assert res1["filename"] == "first_good_invoice.pdf"

    # Second document caught by error boundary
    assert res2["success"] is False
    assert res2["filename"] == "corrupt_doc.pdf"
    assert res2["status"] == "Failed"

    # Third document was not blocked by second document's failure
    assert res3["success"] is True
    assert res3["filename"] == "second_good_resume.pdf"
