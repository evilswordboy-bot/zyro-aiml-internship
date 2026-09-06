"""
Unit tests for text extraction module (src/extractor.py).
Tests:
- PDF text extraction from bytes
- Page count and char count reporting
- Unsupported format handling
- Corrupted PDF handling
"""

import pytest
from pathlib import Path
from src.extractor import DocumentExtractor
from src.config import SAMPLES_DIR


@pytest.fixture
def extractor():
    return DocumentExtractor()


def test_extract_invoice_pdf(extractor):
    pdf_path = SAMPLES_DIR / "invoice_001.pdf"
    assert pdf_path.exists(), "Sample invoice_001.pdf should exist"
    
    bytes_data = pdf_path.read_bytes()
    res = extractor.extract_from_bytes(bytes_data, "invoice_001.pdf")

    assert res.page_count == 1
    assert res.char_count > 100
    assert "INV-1024" in res.text
    assert "ABC TECHNOLOGIES" in res.text
    assert res.used_ocr is False
    assert "Successful" in res.extraction_status


def test_extract_resume_pdf(extractor):
    pdf_path = SAMPLES_DIR / "resume_001.pdf"
    assert pdf_path.exists(), "Sample resume_001.pdf should exist"

    bytes_data = pdf_path.read_bytes()
    res = extractor.extract_from_bytes(bytes_data, "resume_001.pdf")

    assert res.page_count == 1
    assert "Alex Morgan" in res.text
    assert "alex.morgan@example.com" in res.text
    assert "PyTorch" in res.text


def test_extract_unsupported_format(extractor):
    res = extractor.extract_from_bytes(b"some binary data", "doc.unsupported")
    assert res.text == ""
    assert "Unsupported" in res.extraction_status
    assert len(res.warnings) > 0


def test_extract_corrupted_pdf(extractor):
    res = extractor.extract_from_bytes(b"%PDF-corrupted-garbage", "corrupt.pdf")
    assert "Error" in res.extraction_status or res.char_count == 0
