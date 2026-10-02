"""
Comprehensive Week 5 Acceptance Test Matrix & End-to-End Workflow Verification.
Tests 15-document evaluation matrix, state transitions, validation rules,
review queue interactions, audit timeline reconstruction, and batch fault isolation.
"""

import os
import io
import pytest
from PIL import Image, ImageDraw

from database.db import DatabaseManager
from services.file_storage import FileStorageManager
from services.document_processor import DocumentProcessor
from services.workflow import WorkflowEngine
from services.audit import AuditService
from services.batch import BatchProcessor
from src.classifier import DocumentClassifier
from src.ocr_processor import OCRProcessor
from src.utils import generate_rich_sample_pdfs, load_dataset_from_dir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
DB_PATH = os.path.join(DATA_DIR, "documents.db")


def create_scanned_receipt_image() -> bytes:
    """Generates an authentic synthetic scanned receipt image (PNG)."""
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "TAX INVOICE RECEIPT", fill=(0, 0, 0))
    draw.text((20, 60), "Invoice Number: REC-2026-99", fill=(0, 0, 0))
    draw.text((20, 100), "Company: Metro Retailers Ltd", fill=(0, 0, 0))
    draw.text((20, 140), "Total Amount: $ 420.00", fill=(0, 0, 0))
    draw.text((20, 180), "Date: 2026-06-15", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def week5_platform():
    """Initializes isolated database, storage, classifier, workflow, audit, and processor."""
    generate_rich_sample_pdfs(SAMPLES_DIR)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_week5.db")
        storage_dir = os.path.join(tmpdir, "storage")
        db = DatabaseManager(db_path=db_path)
        storage = FileStorageManager(storage_dir=storage_dir)
        audit = AuditService(db)
        workflow = WorkflowEngine(db, audit)

        clf = DocumentClassifier()
        clf.load_model(os.path.join(BASE_DIR, "models", "classifier.pkl"))
        if not clf.is_trained:
            train_texts, train_labels = load_dataset_from_dir(os.path.join(DATA_DIR, "train"))
            if train_texts:
                clf.train(train_texts, train_labels)

        ocr_engine = OCRProcessor()
        processor = DocumentProcessor(
            db_manager=db,
            storage_manager=storage,
            classifier=clf,
            ocr_engine=ocr_engine,
            audit_service=audit,
            workflow_engine=workflow
        )
        batch_proc = BatchProcessor(processor=processor)

        yield db, storage, audit, workflow, processor, batch_proc


# -----------------------------------------------------------------------------
# Document Matrix Tests (15 Documents / Cases)
# -----------------------------------------------------------------------------

def test_01_valid_invoice_completed(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "invoice_1.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "invoice_1.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Invoice"
    assert res["document"]["status"] in ("Completed", "Processed")
    assert res["document"]["invoice_number"] == "INV-2026-001"


def test_02_invoice_with_currency_completed(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "invoice_2.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "invoice_2.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Invoice"
    assert res["document"]["status"] in ("Completed", "Processed")


def test_03_invoice_missing_total_needs_review(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "invoice_missing_total.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "invoice_missing_total.pdf")
    assert res["success"] is True
    assert res["document"]["status"] == "Needs Review"
    assert res["document"]["total_amount"] == "Not Found"


def test_04_invoice_missing_number_needs_review(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "invoice_missing_number.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "invoice_missing_number.pdf")
    assert res["success"] is True
    assert res["document"]["status"] == "Needs Review"
    assert res["document"]["invoice_number"] == "Not Found"


def test_05_invoice_freelance_completed(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "invoice_freelance.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "invoice_freelance.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Invoice"
    assert res["document"]["status"] in ("Completed", "Processed")


def test_06_valid_resume_completed(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "resume_1.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "resume_1.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Resume"
    assert res["document"]["status"] in ("Completed", "Processed")
    assert res["document"]["candidate_name"] == "Alex Smith"


def test_07_resume_missing_phone_needs_review(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "resume_missing_phone.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "resume_missing_phone.pdf")
    assert res["success"] is True
    assert res["document"]["status"] == "Needs Review"
    assert res["document"]["candidate_phone"] == "Not Found"


def test_08_resume_invalid_email_needs_review(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "resume_invalid_email.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "resume_invalid_email.pdf")
    assert res["success"] is True
    assert res["document"]["status"] == "Needs Review"


def test_09_resume_senior_dev_completed(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "resume_senior_dev.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "resume_senior_dev.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Resume"
    assert res["document"]["status"] in ("Completed", "Processed")


def test_10_meeting_minutes_completed(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "meeting_minutes.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "meeting_minutes.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Other"
    assert res["document"]["status"] in ("Completed", "Processed")


def test_11_project_proposal_completed(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "project_proposal.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "project_proposal.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Other"
    assert res["document"]["status"] in ("Completed", "Processed")


def test_12_nda_agreement_completed(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "nda_agreement.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "nda_agreement.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Other"
    assert res["document"]["status"] in ("Completed", "Processed")


def test_13_scanned_receipt_image(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    img_bytes = create_scanned_receipt_image()
    res = processor.process_and_store(img_bytes, "receipt_test.png")
    assert res["document"] is not None
    assert res["document"]["stored_filename"].endswith(".png")


def test_14_duplicate_suppression(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "invoice_freelance.pdf"), "rb") as f:
        data = f.read()
    dup_res = processor.process_and_store(data, "invoice_freelance_copy.pdf")
    assert dup_res["success"] is True
    assert dup_res["is_duplicate"] is True


def test_15_human_review_approval(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "invoice_missing_number.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "invoice_missing_number.pdf")
    doc_id = res["document"]["id"]

    # Approve through workflow engine
    ok, msg = workflow.approve_document(doc_id, reviewer_note="Manually confirmed PO number with vendor.")
    assert ok is True

    doc = db.get_document_by_id(doc_id)
    assert doc["status"] in ("Completed", "Processed")

    # Verify audit trail
    history = audit.get_document_history(doc_id)
    actions = [h["action"] for h in history]
    assert "REVIEWER_APPROVED" in actions
    assert "WORKFLOW_COMPLETED" in actions


def test_16_human_review_rejection_mandatory_note(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    with open(os.path.join(SAMPLES_DIR, "resume_invalid_email.pdf"), "rb") as f:
        data = f.read()
    res = processor.process_and_store(data, "resume_invalid_email.pdf")
    doc_id = res["document"]["id"]

    # Attempt rejection without note -> MUST BE BLOCKED
    ok_empty, err = workflow.reject_document(doc_id, reviewer_note="")
    assert ok_empty is False
    assert "mandatory" in err.lower()

    # Rejection with note -> SUCCEEDS
    ok_valid, msg = workflow.reject_document(doc_id, reviewer_note="Invalid email domain violates applicant policy.")
    assert ok_valid is True

    doc = db.get_document_by_id(doc_id)
    assert doc["status"] == "Rejected"

    history = audit.get_document_history(doc_id)
    assert any(h["action"] == "REVIEWER_REJECTED" for h in history)


def test_17_batch_processing_fault_isolation(week5_platform):
    db, storage, audit, workflow, processor, batch_proc = week5_platform

    with open(os.path.join(SAMPLES_DIR, "meeting_minutes.pdf"), "rb") as f:
        good_doc = f.read()

    corrupt_data = b"NOT A VALID FILE CONTENT 12345"

    batch_items = [
        ("batch_valid_doc.pdf", good_doc),
        ("batch_corrupt_doc.pdf", corrupt_data)
    ]

    summary = batch_proc.process_batch(batch_items)
    assert summary["total"] == 2
    assert len(summary["results"]) == 2
    # First item processed, second failed safely without halting
    assert summary["results"][0]["filename"] == "batch_valid_doc.pdf"
    assert summary["results"][1]["filename"] == "batch_corrupt_doc.pdf"


def test_18_audit_history_timeline_integrity(week5_platform):
    db, storage, audit, workflow, processor, _ = week5_platform
    recent_logs = audit.get_recent_timeline(limit=25)
    assert len(recent_logs) > 0
    # Every audit entry must have timestamp, document_id, action, and new_status
    for entry in recent_logs:
        assert entry["document_id"] is not None
        assert entry["action"] is not None
        assert entry["timestamp"] is not None
        assert entry["new_status"] is not None
