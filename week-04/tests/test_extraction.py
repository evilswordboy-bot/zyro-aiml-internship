"""
Unit Tests for Information Extraction and Document Reading Modules
"""

import pytest
from src.extractor import DocumentExtractor, NOT_FOUND
from src.document_reader import DocumentReader


def test_invoice_field_extraction_complete():
    text = """TechCorp Solutions Inc.
Invoice Number: INV-2026-001
Date: 15/03/2026
Bill To: Global Enterprises Ltd.
Total Amount: $ 1,450.00"""

    res = DocumentExtractor.extract_invoice_fields(text)
    fields = res["fields"]

    assert fields["Invoice Number"] == "INV-2026-001"
    assert fields["Date"] == "15/03/2026"
    assert "TechCorp" in fields["Company Name"]
    assert "1,450.00" in fields["Total Amount"]
    assert len(res["missing_fields"]) == 0
    assert res["fields_found_count"] == 4


def test_invoice_missing_total_handling():
    text = """Beacon Creative Agency
Invoice #: BCA-2026-88
Date: 14-08-2026
Billed To: Solstice Brands
Services: Brand Identity Design
Billing Status: Hours pending final client audit and reconciliation."""

    res = DocumentExtractor.extract_invoice_fields(text)
    fields = res["fields"]

    assert fields["Invoice Number"] == "BCA-2026-88"
    assert fields["Date"] == "14-08-2026"
    assert "Beacon" in fields["Company Name"]
    assert fields["Total Amount"] == NOT_FOUND  # Truthful missing field
    assert "Total Amount" in res["missing_fields"]
    assert res["fields_found_count"] == 3


def test_resume_field_extraction_complete():
    text = """Alex Smith
Email: alex.smith@email.com | Phone: +91 98765 43210 | Bengaluru, India
PROFESSIONAL SUMMARY
Results-driven AI/ML Engineer with 3+ years experience.
SKILLS & TECHNICAL EXPERTISE
Python, SQL, Streamlit, PyMuPDF, Scikit-learn, PyTorch, Docker, Git"""

    res = DocumentExtractor.extract_resume_fields(text)
    fields = res["fields"]

    assert fields["Name"] == "Alex Smith"
    assert fields["Email"] == "alex.smith@email.com"
    assert "+91 98765 43210" in fields["Phone"]
    assert "Python" in fields["Skills"]
    assert len(res["missing_fields"]) == 0


def test_resume_missing_phone_handling():
    text = """Rachel Adams
Email: rachel.adams@uxdesign.net | San Francisco, CA
SUMMARY
Senior UX Designer passionate about design systems.
SKILLS & TOOLS
Figma, React, TypeScript, HTML5, CSS3, Git"""

    res = DocumentExtractor.extract_resume_fields(text)
    fields = res["fields"]

    assert fields["Name"] == "Rachel Adams"
    assert fields["Email"] == "rachel.adams@uxdesign.net"
    assert fields["Phone"] == NOT_FOUND  # Truthful missing field
    assert "Phone" in res["missing_fields"]
    assert "Figma" in fields["Skills"]


def test_document_reader_empty_pdf():
    res = DocumentReader.read_pdf(b"")
    assert res["is_empty"] is True
    assert res["error"] is not None


def test_document_reader_corrupt_pdf():
    res = DocumentReader.read_pdf(b"not a real pdf content")
    assert res["is_empty"] is True
    assert "Failed to open" in res["error"]
