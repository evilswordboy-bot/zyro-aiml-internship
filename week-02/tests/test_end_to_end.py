"""
End-to-end integration tests for Zyroo AI Document Intelligence Platform.
Executes the full pipeline:
Validation -> Extraction -> Classification -> Information Parsing -> Results Verification
on all sample documents generated for Task 2.
"""

import pytest
from pathlib import Path
from src.validator import validate_document
from src.extractor import DocumentExtractor
from src.classifier import HybridClassifier
from src.parser import DocumentParser
from src.sample_generator import generate_samples
from src.config import SAMPLES_DIR


@pytest.fixture(scope="session", autouse=True)
def ensure_samples():
    generate_samples()


@pytest.fixture
def pipeline():
    return {
        "extractor": DocumentExtractor(),
        "classifier": HybridClassifier(),
        "parser": DocumentParser()
    }


def test_e2e_invoice_001(pipeline):
    path = SAMPLES_DIR / "invoice_001.pdf"
    file_bytes = path.read_bytes()

    # Step 1: Validate
    val = validate_document(file_bytes, "invoice_001.pdf")
    assert val.is_valid is True
    assert val.file_type == "PDF"

    # Step 2: Extract
    ext = pipeline["extractor"].extract_from_bytes(file_bytes, "invoice_001.pdf")
    assert ext.char_count > 100
    assert ext.page_count == 1

    # Step 3: Classify
    cls = pipeline["classifier"].classify(ext.text)
    assert cls.document_type == "Invoice"
    assert cls.confidence >= 0.85

    # Step 4: Parse
    parse = pipeline["parser"].parse(ext.text, cls.document_type)
    assert parse.fields["Invoice Number"].value == "INV-1024"
    assert parse.fields["Date"].value == "02-09-2026"
    assert "ABC Technologies" in parse.fields["Company Name"].value
    assert "125,000" in parse.fields["Total Amount"].value


def test_e2e_invoice_002(pipeline):
    path = SAMPLES_DIR / "invoice_002.pdf"
    file_bytes = path.read_bytes()

    val = validate_document(file_bytes, "invoice_002.pdf")
    assert val.is_valid is True

    ext = pipeline["extractor"].extract_from_bytes(file_bytes, "invoice_002.pdf")
    assert ext.char_count > 50

    cls = pipeline["classifier"].classify(ext.text)
    assert cls.document_type == "Invoice"

    parse = pipeline["parser"].parse(ext.text, cls.document_type)
    assert parse.fields["Invoice Number"].value == "INV-2026-889"
    assert "4,850.00" in parse.fields["Total Amount"].value


def test_e2e_resume_001(pipeline):
    path = SAMPLES_DIR / "resume_001.pdf"
    file_bytes = path.read_bytes()

    val = validate_document(file_bytes, "resume_001.pdf")
    assert val.is_valid is True

    ext = pipeline["extractor"].extract_from_bytes(file_bytes, "resume_001.pdf")
    assert ext.char_count > 500

    cls = pipeline["classifier"].classify(ext.text)
    assert cls.document_type == "Resume"

    parse = pipeline["parser"].parse(ext.text, cls.document_type)
    assert parse.fields["Name"].value == "Alex Morgan"
    assert parse.fields["Email"].value == "alex.morgan@example.com"
    assert "+1-555-019-2834" in parse.fields["Phone"].value
    assert len(parse.all_skills_flat) >= 5


def test_e2e_sample_other(pipeline):
    path = SAMPLES_DIR / "sample_other.pdf"
    file_bytes = path.read_bytes()

    val = validate_document(file_bytes, "sample_other.pdf")
    assert val.is_valid is True

    ext = pipeline["extractor"].extract_from_bytes(file_bytes, "sample_other.pdf")
    cls = pipeline["classifier"].classify(ext.text)
    assert cls.document_type == "Other"
