"""
Unit tests for information extraction parser (src/parser.py).
Tests:
- Invoice field extraction (Invoice Number, Date, Company Name, Total Amount)
- Resume field extraction (Name, Email, Phone, Skills)
- Missing field handling and status tracking
- Completeness score calculation
"""

import pytest
from src.parser import DocumentParser


@pytest.fixture
def parser():
    return DocumentParser()


def test_parse_complete_invoice(parser):
    text = """
    ABC Technologies Global
    TAX INVOICE
    Invoice Number: INV-1024
    Date: 02-09-2026
    Bill To: ABC Technologies Global
    Description: Machine Learning Consulting
    Total Amount: PKR 125,000
    """
    res = parser.parse(text, "Invoice")
    assert res.document_type == "Invoice"
    assert res.fields["Invoice Number"].found is True
    assert res.fields["Invoice Number"].value == "INV-1024"
    assert res.fields["Date"].found is True
    assert res.fields["Date"].value == "02-09-2026"
    assert res.fields["Company Name"].found is True
    assert "ABC Technologies" in res.fields["Company Name"].value
    assert res.fields["Total Amount"].found is True
    assert "125,000" in res.fields["Total Amount"].value
    assert res.completeness_score == 100.0


def test_parse_invoice_missing_fields(parser):
    text = """
    TAX INVOICE
    Date: 10-10-2026
    Total Amount: $500.00
    """
    res = parser.parse(text, "Invoice")
    assert res.fields["Invoice Number"].found is False
    assert res.fields["Invoice Number"].value == "Not found"
    assert res.fields["Date"].found is True
    assert res.completeness_score < 100.0


def test_parse_complete_resume(parser):
    text = """
    Alex Morgan
    Email: alex.morgan@example.com
    Phone: +1-555-019-2834
    Professional Summary: Senior ML Engineer
    Technical Skills: Python, PyTorch, Scikit-Learn, Docker, Kubernetes, FastAPI, SQL
    Education: BS Computer Science
    """
    res = parser.parse(text, "Resume")
    assert res.document_type == "Resume"
    assert res.fields["Name"].found is True
    assert res.fields["Name"].value == "Alex Morgan"
    assert res.fields["Email"].found is True
    assert res.fields["Email"].value == "alex.morgan@example.com"
    assert res.fields["Phone"].found is True
    assert "+1-555-019-2834" in res.fields["Phone"].value
    assert res.fields["Skills"].found is True
    assert "Python" in res.all_skills_flat
    assert "PyTorch" in res.all_skills_flat
    assert "Docker" in res.all_skills_flat
    assert res.completeness_score == 100.0


def test_parse_other_document(parser):
    text = "Weekly engineering sync discussion on caching layers and database indexing."
    res = parser.parse(text, "Other")
    assert res.document_type == "Other"
    assert "Status" in res.fields
