"""
Unit and integration tests for AnomalyDetector engine.
Week 6: Anomaly Detection, Financial Consistency, and Quality Checks.
"""

import pytest
from services.anomaly import (
    AnomalyDetector,
    ANOMALY_ARITHMETIC_MISMATCH,
    ANOMALY_SUSPICIOUS_AMOUNT,
    ANOMALY_FUTURE_DATE,
    ANOMALY_LOW_TEXT_QUALITY,
    ANOMALY_DUPLICATE_CONTENT,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM
)
from database.db import DatabaseManager
import tempfile
import os


def test_invoice_arithmetic_clean():
    """Clean invoice where Subtotal + Tax = Total should produce no arithmetic anomaly."""
    fields = {"Total Amount": "$ 1,100.00"}
    text = "Subtotal: $ 1,000.00\nTax: $ 100.00\nTotal Amount: $ 1,100.00"
    res = AnomalyDetector.detect_anomalies("Invoice", fields, text)
    assert res["has_anomalies"] is False
    assert len(res["anomalies"]) == 0


def test_invoice_arithmetic_mismatch():
    """Invoice where Subtotal + Tax != Total should trigger ANOMALY_ARITHMETIC_MISMATCH with HIGH severity."""
    fields = {"Total Amount": "$ 1,850.00"}
    text = "Subtotal: $ 1,000.00\nTax: $ 100.00\nTotal Amount: $ 1,850.00"
    res = AnomalyDetector.detect_anomalies("Invoice", fields, text)
    assert res["has_anomalies"] is True
    assert res["has_high_severity"] is True

    types = [a["type"] for a in res["anomalies"]]
    assert ANOMALY_ARITHMETIC_MISMATCH in types
    mismatch = [a for a in res["anomalies"] if a["type"] == ANOMALY_ARITHMETIC_MISMATCH][0]
    assert mismatch["severity"] == SEVERITY_HIGH
    assert mismatch["details"]["difference"] == 750.0


def test_suspicious_negative_and_zero_amount():
    """Negative or zero invoice amount triggers high-severity anomaly."""
    res_neg = AnomalyDetector.detect_anomalies("Invoice", {"Total Amount": "-$ 500.00"}, "Invoice total: -$ 500.00")
    assert res_neg["has_anomalies"] is True
    assert any(a["type"] == ANOMALY_SUSPICIOUS_AMOUNT for a in res_neg["anomalies"])

    res_zero = AnomalyDetector.detect_anomalies("Invoice", {"Total Amount": "$ 0.00"}, "Invoice total: $ 0.00")
    assert res_zero["has_anomalies"] is True
    assert any(a["type"] == ANOMALY_SUSPICIOUS_AMOUNT for a in res_zero["anomalies"])


def test_suspicious_extreme_amount():
    """Amounts over $1,000,000 trigger a high-value transaction warning."""
    fields = {"Total Amount": "$ 5,250,000.00"}
    res = AnomalyDetector.detect_anomalies("Invoice", fields, "Invoice Total: $ 5,250,000.00")
    assert res["has_anomalies"] is True
    assert any(a["type"] == ANOMALY_SUSPICIOUS_AMOUNT and a["severity"] == SEVERITY_MEDIUM for a in res["anomalies"])


def test_future_date_detection():
    """Dates far into the future (> 365 days) trigger FUTURE_DATE anomaly."""
    text = "Invoice Date: 2035-12-01\nTotal: $ 500.00"
    res = AnomalyDetector.detect_anomalies("Invoice", {"Total Amount": "$ 500.00"}, text)
    assert res["has_anomalies"] is True
    assert any(a["type"] == ANOMALY_FUTURE_DATE for a in res["anomalies"])


def test_low_text_quality_repetition():
    """Repetitive garbage characters trigger LOW_TEXT_QUALITY anomaly."""
    text = "Invoice Number: INV-01\n" + "x" * 50 + "\nTotal Amount: $ 100.00"
    res = AnomalyDetector.detect_anomalies("Invoice", {"Total Amount": "$ 100.00"}, text)
    assert res["has_anomalies"] is True
    assert any(a["type"] == ANOMALY_LOW_TEXT_QUALITY for a in res["anomalies"])


def test_duplicate_hash_anomaly():
    """Detects existing document collision when DB is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = DatabaseManager(db_path)
        db.add_document({
            "original_filename": "first_doc.pdf",
            "stored_filename": "first_doc.pdf",
            "document_type": "Invoice",
            "file_hash": "dummyhash12345678",
            "status": "Completed",
            "file_path": "invoices/first_doc.pdf",
            "text_preview": "Some text"
        })

        res = AnomalyDetector.detect_anomalies(
            doc_type="Invoice",
            fields={"Total Amount": "$ 250.00"},
            text="Clean text",
            file_hash="dummyhash12345678",
            db_manager=db
        )
        assert res["has_anomalies"] is True
        assert any(a["type"] == ANOMALY_DUPLICATE_CONTENT for a in res["anomalies"])
