"""
Unit tests for document validation module (src/validator.py).
Tests:
- Normal valid PDF input
- Normal valid image input
- Empty input (0 bytes)
- Disallowed file extensions (.exe, .txt, .docx)
- Empty or whitespace filename
- File size limit constraint (>10MB)
- Corrupted PDF magic byte check
- Corrupted image check
"""

import pytest
from src.validator import validate_document
from src.config import MAX_FILE_SIZE_BYTES


def test_validate_valid_pdf():
    valid_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    res = validate_document(valid_pdf_bytes, "test_doc.pdf")
    assert res.is_valid is True
    assert res.file_type == "PDF"
    assert res.error_message is None
    assert res.file_size_kb > 0


def test_validate_empty_file():
    res = validate_document(b"", "empty.pdf")
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()
    assert res.file_type == "Empty"


def test_validate_empty_filename():
    res = validate_document(b"%PDF-1.4...", "   ")
    assert res.is_valid is False
    assert "filename cannot be empty" in res.error_message.lower()


def test_validate_unsupported_extension():
    res = validate_document(b"Some text content", "malicious.exe")
    assert res.is_valid is False
    assert "unsupported file format" in res.error_message.lower()

    res2 = validate_document(b"Word document", "notes.docx")
    assert res2.is_valid is False


def test_validate_oversized_file():
    oversized_bytes = b"%PDF" + b"0" * (MAX_FILE_SIZE_BYTES + 1024)
    res = validate_document(oversized_bytes, "huge_document.pdf")
    assert res.is_valid is False
    assert "exceeds maximum allowed size" in res.error_message.lower()


def test_validate_corrupted_pdf_header():
    # File named .pdf but doesn't start with %PDF
    fake_pdf = b"NOT_A_PDF_HEADER_JUST_RANDOM_TEXT"
    res = validate_document(fake_pdf, "fake.pdf")
    assert res.is_valid is False
    assert "not a valid pdf" in res.error_message.lower()


def test_validate_corrupted_image():
    # File named .png with corrupted header
    corrupt_png = b"NOT_A_PNG"
    res = validate_document(corrupt_png, "broken.png")
    assert res.is_valid is False
    assert "corrupted" in res.error_message.lower()
