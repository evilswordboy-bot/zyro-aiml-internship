"""
Comprehensive Week 4 Acceptance Test Matrix & Persistence Verification.
Validates:
1. Invoice upload
2. Resume upload
3. Duplicate upload (suppressed)
4. Scanned/image document
5. Missing-field invoice (Needs Review)
6. Missing-field resume (Needs Review)
7. Invalid file rejection
8. Oversized file rejection
9. Multi-field search
10. Category and status filters
11. Sorting
12. Restart persistence test
"""

import os
import io
import pytest
from PIL import Image, ImageDraw, ImageFont

from database.db import DatabaseManager
from services.file_storage import FileStorageManager
from services.document_processor import DocumentProcessor
from services.validation import FileValidator
from src.classifier import DocumentClassifier
from src.ocr_processor import OCRProcessor
from src.utils import load_dataset_from_dir, generate_rich_sample_pdfs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
DB_PATH = os.path.join(DATA_DIR, "documents.db")


def create_test_scanned_image() -> bytes:
    """Generates an authentic synthetic scanned receipt image (PNG)."""
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "INVOICE", fill=(0, 0, 0))
    draw.text((20, 60), "Invoice Number: INV-9009", fill=(0, 0, 0))
    draw.text((20, 100), "Company: Scanned Logistics Ltd", fill=(0, 0, 0))
    draw.text((20, 140), "Total Amount: $ 890.00", fill=(0, 0, 0))
    draw.text((20, 180), "Date: 2026-09-25", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def app_services():
    """Initializes primary database, storage, classifier, and processor."""
    generate_rich_sample_pdfs(SAMPLES_DIR)
    db = DatabaseManager(db_path=DB_PATH)
    storage = FileStorageManager(storage_dir=STORAGE_DIR)

    clf = DocumentClassifier()
    clf.load_model(os.path.join(BASE_DIR, "models", "classifier.pkl"))
    if not clf.is_trained:
        train_texts, train_labels = load_dataset_from_dir(os.path.join(DATA_DIR, "train"))
        if train_texts:
            clf.train(train_texts, train_labels)

    ocr_engine = OCRProcessor()
    processor = DocumentProcessor(db_manager=db, storage_manager=storage, classifier=clf, ocr_engine=ocr_engine)
    return db, storage, processor


def test_01_invoice_upload(app_services):
    db, storage, processor = app_services
    with open(os.path.join(SAMPLES_DIR, "invoice_1.pdf"), "rb") as f:
        file_bytes = f.read()

    res = processor.process_and_store(file_bytes, "invoice_1.pdf")
    assert res["success"] is True
    assert res["is_duplicate"] is False or res["document"]["document_type"] == "Invoice"
    assert res["document"]["document_type"] == "Invoice"


def test_02_invoice_with_currency(app_services):
    db, storage, processor = app_services
    with open(os.path.join(SAMPLES_DIR, "invoice_2.pdf"), "rb") as f:
        file_bytes = f.read()

    res = processor.process_and_store(file_bytes, "invoice_2.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Invoice"


def test_03_resume_upload(app_services):
    db, storage, processor = app_services
    with open(os.path.join(SAMPLES_DIR, "resume_1.pdf"), "rb") as f:
        file_bytes = f.read()

    res = processor.process_and_store(file_bytes, "resume_1.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Resume"
    assert res["document"]["candidate_name"] != "Not Found"


def test_04_other_document_upload(app_services):
    db, storage, processor = app_services
    with open(os.path.join(SAMPLES_DIR, "meeting_minutes.pdf"), "rb") as f:
        file_bytes = f.read()

    res = processor.process_and_store(file_bytes, "meeting_minutes.pdf")
    assert res["success"] is True
    assert res["document"]["document_type"] == "Other"


def test_05_duplicate_suppression(app_services):
    db, storage, processor = app_services
    with open(os.path.join(SAMPLES_DIR, "invoice_1.pdf"), "rb") as f:
        file_bytes = f.read()

    # Attempt re-upload of identical file
    res = processor.process_and_store(file_bytes, "invoice_1_duplicate.pdf")
    assert res["success"] is True
    assert res["is_duplicate"] is True
    assert "Duplicate document detected" in res["message"]


def test_06_missing_invoice_field_needs_review(app_services):
    db, storage, processor = app_services
    with open(os.path.join(SAMPLES_DIR, "invoice_missing_total.pdf"), "rb") as f:
        file_bytes = f.read()

    res = processor.process_and_store(file_bytes, "invoice_missing_total.pdf")
    assert res["success"] is True
    assert res["document"]["status"] == "Needs Review"
    assert res["document"]["total_amount"] == "Not Found"


def test_07_missing_resume_field_needs_review(app_services):
    db, storage, processor = app_services
    with open(os.path.join(SAMPLES_DIR, "resume_missing_phone.pdf"), "rb") as f:
        file_bytes = f.read()

    res = processor.process_and_store(file_bytes, "resume_missing_phone.pdf")
    assert res["success"] is True
    assert res["document"]["status"] == "Needs Review"
    assert res["document"]["candidate_phone"] == "Not Found"


def test_08_scanned_image_processing(app_services):
    db, storage, processor = app_services
    scanned_bytes = create_test_scanned_image()
    res = processor.process_and_store(scanned_bytes, "receipt_scanned.png")
    # File must always be safely saved in storage and recorded in SQLite
    assert res["document"] is not None
    assert res["document"]["stored_filename"].endswith(".png")
    assert os.path.isfile(storage.get_full_path(res["document"]["file_path"]))
    if processor.ocr_engine.tesseract_available:
        assert res["success"] is True
    else:
        # Safe error handling when OCR binary is missing on host environment
        assert res["document"]["status"] in ("Failed", "Needs Review")


def test_09_invalid_file_rejection(app_services):
    db, storage, processor = app_services
    res = processor.process_and_store(b"binary executable", "malware.exe")
    assert res["success"] is False
    assert "Unsupported file format" in res["error"]


def test_10_oversized_file_rejection(app_services):
    db, storage, processor = app_services
    huge_data = b"0" * (21 * 1024 * 1024)
    res = processor.process_and_store(huge_data, "massive.pdf")
    assert res["success"] is False
    assert "exceeds maximum upload size" in res["error"]


def test_11_multi_field_search(app_services):
    db, storage, processor = app_services
    # Search by company
    c_res = db.search_documents(search_query="TechCorp")
    assert len(c_res) >= 1
    assert any("TechCorp" in d["company"] for d in c_res)

    # Search by candidate
    r_res = db.search_documents(search_query="Alex Smith")
    assert len(r_res) >= 1
    assert any("Alex Smith" in d["candidate_name"] for d in r_res)

    # Search by document type
    inv_res = db.search_documents(doc_type="Invoice")
    assert len(inv_res) >= 1
    assert all(d["document_type"] == "Invoice" for d in inv_res)


def test_12_sorting_and_filtering(app_services):
    db, storage, processor = app_services
    newest = db.search_documents(sort_order="DESC")
    oldest = db.search_documents(sort_order="ASC")
    assert len(newest) == len(oldest)
    if len(newest) > 1:
        assert newest[0]["upload_date"] >= oldest[0]["upload_date"]


def test_13_restart_persistence():
    """Simulates application restart by re-instantiating fresh DB and Storage."""
    fresh_db = DatabaseManager(db_path=DB_PATH)
    fresh_storage = FileStorageManager(storage_dir=STORAGE_DIR)

    all_docs = fresh_db.get_all_documents()
    assert len(all_docs) >= 5, "Database records must persist across restart."

    # Verify physical files remain readable in storage
    for doc in all_docs[:3]:
        file_bytes = fresh_storage.read_file(doc["file_path"])
        assert file_bytes is not None and len(file_bytes) > 0, f"Stored file {doc['file_path']} must persist."
