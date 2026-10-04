"""
Complete End-to-End Acceptance Test Matrix for ZYROO AI/ML Internship Week 6.
Directly implements and validates all 14 mandatory production acceptance test scenarios:
  Test 1  — Clean Digital PDF
  Test 2  — Scanned Document (OCR Fallback)
  Test 3  — Supported Document Types (Invoice, Resume, Other)
  Test 4  — Missing Information
  Test 5  — Duplicate Document
  Test 6  — Unsupported File
  Test 7  — Corrupted Document
  Test 8  — Low Confidence
  Test 9  — Financial Validation (Arithmetic Mismatch)
  Test 10 — Invalid Workflow Transition
  Test 11 — Mixed Batch Processing
  Test 12 — Restart Persistence
  Test 13 — Document AI Assistant (Grounded QA with Citations)
  Test 14 — Insufficient Information Handling (No-Hallucination Guardrail)
"""

import os
import io
import tempfile
import pytest
from PIL import Image, ImageDraw

from database.db import DatabaseManager
from services.file_storage import FileStorageManager
from services.document_processor import DocumentProcessor
from services.workflow import WorkflowEngine, InvalidStateTransitionError
from services.audit import AuditService
from services.batch import BatchProcessor
from services.rag import RAGService
from src.classifier import DocumentClassifier
from src.ocr_processor import OCRProcessor
from src.utils import generate_rich_sample_pdfs, load_dataset_from_dir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def create_scanned_receipt_image() -> bytes:
    """Generates an authentic synthetic scanned tax receipt image (PNG)."""
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
def week6_platform():
    """Initializes isolated test harness for Week 6 Acceptance Matrix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples_dir = os.path.join(tmpdir, "samples")
        generate_rich_sample_pdfs(samples_dir)

        db_path = os.path.join(tmpdir, "week6_platform.db")
        storage_dir = os.path.join(tmpdir, "storage")

        db = DatabaseManager(db_path=db_path)
        storage = FileStorageManager(storage_dir=storage_dir)
        audit = AuditService(db)
        workflow = WorkflowEngine(db, audit)

        clf = DocumentClassifier()
        clf.load_model(os.path.join(BASE_DIR, "models", "classifier.pkl"))
        if not clf.is_trained:
            train_texts, train_labels = load_dataset_from_dir(os.path.join(DATA_DIR, "train"))
            clf.train(train_texts, train_labels)

        ocr_eng = OCRProcessor()
        processor = DocumentProcessor(
            db_manager=db,
            storage_manager=storage,
            classifier=clf,
            ocr_engine=ocr_eng,
            audit_service=audit,
            workflow_engine=workflow
        )
        batch_processor = BatchProcessor(processor=processor)
        rag_service = RAGService(db_manager=db)

        yield {
            "tmpdir": tmpdir,
            "samples_dir": samples_dir,
            "db_path": db_path,
            "db": db,
            "storage": storage,
            "audit": audit,
            "workflow": workflow,
            "processor": processor,
            "batch_processor": batch_processor,
            "rag_service": rag_service
        }


# ==============================================================================
# 14 MANDATORY WEEK 6 ACCEPTANCE TESTS
# ==============================================================================

def test_scenario_01_clean_digital_pdf(week6_platform):
    """Test 1: Upload a clean digital PDF and verify successful straight-through processing."""
    samples_dir = week6_platform["samples_dir"]
    proc = week6_platform["processor"]

    with open(os.path.join(samples_dir, "invoice_1.pdf"), "rb") as f:
        pdf_bytes = f.read()

    res = proc.process_and_store(pdf_bytes, "invoice_1.pdf")
    assert res["success"] is True
    assert res["document_type"] == "Invoice"
    assert res["status"] in ("Approved", "Completed")
    assert res["has_anomalies"] is False
    assert res["processing_time_ms"] > 0.0


def test_scenario_02_scanned_document_ocr(week6_platform):
    """Test 2: Upload a scanned document and verify OCR computer vision fallback."""
    proc = week6_platform["processor"]
    scanned_bytes = create_scanned_receipt_image()

    res = proc.process_and_store(scanned_bytes, "receipt_scanned.png")
    assert res["document"] is not None
    assert res["document"]["stored_filename"].endswith(".png")
    if proc.ocr_engine.tesseract_available:
        assert res["success"] is True
        assert res["ocr_used"] is True
    else:
        # Graceful handling when host environment lacks Tesseract system binary
        assert res["document"]["status"] in ("Failed", "Needs Review")


def test_scenario_03_supported_document_types(week6_platform):
    """Test 3: Verify processing of all supported types: Invoice, Resume, Other (Proposal/PO/Contract)."""
    samples_dir = week6_platform["samples_dir"]
    proc = week6_platform["processor"]

    # 1. Invoice
    with open(os.path.join(samples_dir, "invoice_2.pdf"), "rb") as f:
        res_inv = proc.process_and_store(f.read(), "invoice_2.pdf")
    assert res_inv["document_type"] == "Invoice"

    # 2. Resume
    with open(os.path.join(samples_dir, "resume_1.pdf"), "rb") as f:
        res_res = proc.process_and_store(f.read(), "resume_1.pdf")
    assert res_res["document_type"] == "Resume"

    # 3. Other / Proposal / PO
    with open(os.path.join(samples_dir, "project_proposal.pdf"), "rb") as f:
        res_prop = proc.process_and_store(f.read(), "project_proposal.pdf")
    assert res_prop["document_type"] == "Other"


def test_scenario_04_missing_information(week6_platform):
    """Test 4: Upload document with missing required fields -> verify validation flags them and routes to Needs Review."""
    samples_dir = week6_platform["samples_dir"]
    proc = week6_platform["processor"]

    with open(os.path.join(samples_dir, "invoice_missing_number.pdf"), "rb") as f:
        res = proc.process_and_store(f.read(), "invoice_missing_number.pdf")

    assert res["success"] is True
    assert res["status"] == "Needs Review"
    assert "Invoice Number" in res["missing_fields"]
    assert res["validation"]["is_valid"] is False


def test_scenario_05_duplicate_document(week6_platform):
    """Test 5: Re-upload identical document -> verify cryptographic duplicate suppression."""
    samples_dir = week6_platform["samples_dir"]
    proc = week6_platform["processor"]

    with open(os.path.join(samples_dir, "invoice_1.pdf"), "rb") as f:
        pdf_bytes = f.read()

    # Re-upload invoice_1.pdf
    res_dup = proc.process_and_store(pdf_bytes, "invoice_1_duplicate_upload.pdf")
    assert res_dup["success"] is True
    assert res_dup["is_duplicate"] is True
    assert "Duplicate document detected" in res_dup["message"]


def test_scenario_06_unsupported_file(week6_platform):
    """Test 6: Upload unsupported file type -> verify graceful rejection without crash."""
    proc = week6_platform["processor"]
    dummy_exe = b"MZ\x90\x00\x03\x00\x00\x00"

    res = proc.process_and_store(dummy_exe, "malicious_payload.exe")
    assert res["success"] is False
    assert "Unsupported file format" in res["error"]
    assert res["status"] == "Failed"


def test_scenario_07_corrupted_document(week6_platform):
    """Test 7: Upload corrupted/unreadable document -> verify safe error handling and Failed status."""
    proc = week6_platform["processor"]
    corrupted_pdf = b"%PDF-1.4\n%GARBAGE_BYTES_CORRUPTED_STREAM_EOF"

    res = proc.process_and_store(corrupted_pdf, "corrupted_document.pdf")
    assert res["success"] is False
    assert res["status"] == "Failed"
    assert "Unreadable" in res["document"]["document_type"] or "error" in res


def test_scenario_08_low_confidence(week6_platform):
    """Test 8: Test document with low classification confidence -> routed to Needs Review."""
    workflow = week6_platform["workflow"]

    val_clean = {"is_valid": True, "reasons": [], "missing_fields": [], "invalid_fields": []}
    target_st, action, reason = workflow.evaluate_rules(
        doc_type="Invoice",
        validation_result=val_clean,
        confidence=0.45,  # below 0.70 threshold
        text_valid=True
    )
    assert target_st == "Needs Review"
    assert "confidence (45.0%) is below approval threshold" in reason


def test_scenario_09_financial_validation_arithmetic_mismatch(week6_platform):
    """Test 9: Upload invoice with Subtotal + Tax != Total -> verify arithmetic anomaly and routing to Needs Review."""
    samples_dir = week6_platform["samples_dir"]
    proc = week6_platform["processor"]

    with open(os.path.join(samples_dir, "invoice_arithmetic_mismatch.pdf"), "rb") as f:
        pdf_bytes = f.read()

    res = proc.process_and_store(pdf_bytes, "invoice_arithmetic_mismatch.pdf")
    assert res["success"] is True
    assert res["has_anomalies"] is True
    assert any(a["type"] == "ARITHMETIC_MISMATCH" for a in res["anomalies"])
    assert res["status"] == "Needs Review"
    assert "Arithmetic mismatch" in res["status_reason"]


def test_scenario_10_invalid_workflow_transition(week6_platform):
    """Test 10: Attempt disallowed state transition (e.g. Rejected -> Approved) -> verify system prevents it."""
    workflow = week6_platform["workflow"]
    db = week6_platform["db"]

    doc_id = db.add_document({
        "original_filename": "terminal_doc.pdf",
        "stored_filename": "terminal_doc.pdf",
        "document_type": "Invoice",
        "file_hash": "terminal_hash_test_10",
        "status": "Rejected",
        "current_status": "Rejected",
        "file_path": "invoices/terminal_doc.pdf",
        "text_preview": "Terminal state test"
    })

    # Attempt illegal jump: Rejected -> Approved
    ok, err_msg = workflow.transition_document(doc_id, "Approved", "UNAUTHORIZED_JUMP")
    assert ok is False
    assert "Blocked invalid transition" in err_msg


def test_scenario_11_mixed_batch_processing(week6_platform):
    """Test 11: Submit mixed batch of valid, missing fields, and corrupted files -> all isolated properly."""
    samples_dir = week6_platform["samples_dir"]
    batch_proc = week6_platform["batch_processor"]

    with open(os.path.join(samples_dir, "resume_senior_dev.pdf"), "rb") as f:
        valid_bytes = f.read()
    with open(os.path.join(samples_dir, "invoice_missing_number.pdf"), "rb") as f:
        missing_bytes = f.read()
    corrupt_bytes = b"BROKEN_BYTES"

    batch_files = [
        ("batch_valid_resume.pdf", valid_bytes),
        ("batch_missing_inv.pdf", missing_bytes),
        ("batch_corrupt_file.pdf", corrupt_bytes)
    ]

    res = batch_proc.process_batch(batch_files)
    assert res["total"] == 3
    assert res["failed"] >= 1  # Corrupt file failed safely
    assert (res["needs_review"] >= 1 or res["duplicates"] >= 1)  # Missing field routed or duplicate detected
    assert res["completed"] + res["needs_review"] + res["duplicates"] >= 2  # Non-corrupt files processed successfully


def test_scenario_12_restart_persistence(week6_platform):
    """Test 12: Re-instantiate database manager pointing to same SQLite file -> verify all data and audit logs persist."""
    db_path = week6_platform["db_path"]

    # Open fresh database connection directly from disk
    restarted_db = DatabaseManager(db_path=db_path)
    kpis = restarted_db.get_statistics()

    assert kpis["total"] > 0
    assert kpis["total_audit_events"] > 0
    recent = restarted_db.get_recent_documents(limit=5)
    assert len(recent) > 0


def test_scenario_13_document_ai_assistant_grounded_qa(week6_platform):
    """Test 13: Ask AI assistant question answerable from indexed documents -> verified grounded answer with source citations."""
    rag_service = week6_platform["rag_service"]
    rag_service.sync_index_from_db()

    q = "What is the invoice number and total amount for Quantum Dynamics Corp?"
    ans = rag_service.ask(q)

    assert ans["answered"] is True
    assert ans["status"] == "GROUNDED_ANSWER"
    assert len(ans["sources"]) > 0
    assert "INV-2026-ARITH" in ans["answer"] or "1,850.00" in ans["answer"] or "Quantum Dynamics" in ans["answer"]


def test_scenario_14_insufficient_information_guardrail(week6_platform):
    """Test 14: Ask question with unavailable information -> assistant must clearly state insufficient info without hallucinating."""
    rag_service = week6_platform["rag_service"]

    q_unknown = "What is the warp core coolant temperature of the Millennium Falcon in hyperspace?"
    ans = rag_service.ask(q_unknown)

    assert ans["answered"] is False
    assert ans["status"] == "INSUFFICIENT_INFORMATION"
    assert "The available documents do not contain sufficient information to answer this question." in ans["answer"]
    assert len(ans["sources"]) == 0
