"""
Unit tests for DocumentValidator (Week 5).
Validates structural, numerical, pattern, and presence checks for Invoices, Resumes, and Other documents.
"""

import pytest
from services.validation import DocumentValidator, NOT_FOUND


def test_validate_invoice_valid_complete():
    fields = {
        "Invoice Number": "INV-2026-999",
        "Company Name": "Apex Systems Inc.",
        "Total Amount": "$ 2,450.00",
        "Date": "2026-05-20"
    }
    res = DocumentValidator.validate_invoice(fields, text="Sample text")
    assert res["is_valid"] is True
    assert len(res["missing_fields"]) == 0
    assert len(res["invalid_fields"]) == 0
    assert "Invoice Number" in res["passed_fields"]
    assert "Total Amount" in res["passed_fields"]
    assert "Company Name" in res["passed_fields"]


def test_validate_invoice_missing_number():
    fields = {
        "Invoice Number": NOT_FOUND,
        "Company Name": "Apex Systems Inc.",
        "Total Amount": "$ 500.00"
    }
    res = DocumentValidator.validate_invoice(fields)
    assert res["is_valid"] is False
    assert "Invoice Number" in res["missing_fields"]
    assert any("Invoice Number" in r for r in res["reasons"])


def test_validate_invoice_invalid_amount():
    fields = {
        "Invoice Number": "INV-101",
        "Company Name": "Alpha Corp",
        "Total Amount": "INVALID_CURRENCY"
    }
    res = DocumentValidator.validate_invoice(fields)
    assert res["is_valid"] is False
    assert any(inv["field"] == "Total Amount" for inv in res["invalid_fields"])


def test_validate_invoice_currency_parsing():
    # Various real-world currency formats
    assert DocumentValidator.parse_numeric_amount("$ 1,234.56") == 1234.56
    assert DocumentValidator.parse_numeric_amount("Rs. 78,500") == 78500.0
    assert DocumentValidator.parse_numeric_amount("€ 999.00") == 999.0
    assert DocumentValidator.parse_numeric_amount("4500") == 4500.0
    assert DocumentValidator.parse_numeric_amount("-50.00") == -50.0
    assert DocumentValidator.parse_numeric_amount(NOT_FOUND) is None


def test_validate_resume_valid_complete():
    fields = {
        "Name": "Jane Doe",
        "Email": "jane.doe@techvibe.io",
        "Phone": "+1 (555) 234-5678",
        "Skills": "Python, Machine Learning, PyTorch, Docker"
    }
    res = DocumentValidator.validate_resume(fields, text="Full resume text content")
    assert res["is_valid"] is True
    assert len(res["missing_fields"]) == 0
    assert len(res["invalid_fields"]) == 0
    assert "Candidate Name" in res["passed_fields"]
    assert "Email Address" in res["passed_fields"]
    assert "Phone Number" in res["passed_fields"]
    assert "Skills" in res["passed_fields"]


def test_validate_resume_invalid_email():
    fields = {
        "Name": "John Smith",
        "Email": "john.smith.missing.domain",
        "Phone": "9876543210",
        "Skills": "Java, Spring, SQL"
    }
    res = DocumentValidator.validate_resume(fields)
    assert res["is_valid"] is False
    assert any(inv["field"] == "Email Address" for inv in res["invalid_fields"])


def test_validate_resume_missing_phone():
    fields = {
        "Name": "Sarah Connor",
        "Email": "sarah.connor@cyberdyne.org",
        "Phone": NOT_FOUND,
        "Skills": "Robotics, AI Security"
    }
    res = DocumentValidator.validate_resume(fields)
    assert res["is_valid"] is False
    assert "Phone Number" in res["missing_fields"]


def test_validate_resume_missing_skills():
    fields = {
        "Name": "Sarah Connor",
        "Email": "sarah.connor@cyberdyne.org",
        "Phone": "9876543210",
        "Skills": NOT_FOUND
    }
    res = DocumentValidator.validate_resume(fields)
    assert res["is_valid"] is False
    assert "Skills" in res["missing_fields"]


def test_validate_other_document():
    # Valid other document with sufficient text
    valid_res = DocumentValidator.validate_document(
        "Other", {}, text="This is a project design document outlining system architecture and microservices."
    )
    assert valid_res["is_valid"] is True

    # Invalid other document with insufficient text
    short_res = DocumentValidator.validate_document("Other", {}, text="Too short")
    assert short_res["is_valid"] is False
