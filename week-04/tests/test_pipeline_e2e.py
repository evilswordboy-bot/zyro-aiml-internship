"""
End-to-end integration test of the full document analysis pipeline.
"""

import os
import pytest
from app import initialize_system, process_document
from src.extractor import NOT_FOUND

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")


@pytest.fixture(scope="module")
def system_components():
    clf, ocr_engine, eval_results = initialize_system()
    return clf, ocr_engine, eval_results


def test_pipeline_normal_invoice(system_components):
    clf, ocr_engine, _ = system_components
    path = os.path.join(SAMPLES_DIR, "invoice_1.pdf")
    with open(path, "rb") as f:
        data = f.read()

    res = process_document(
        file_bytes=data,
        filename="invoice_1.pdf",
        file_extension="pdf",
        classifier=clf,
        ocr_engine=ocr_engine,
        active_model_name="Logistic Regression",
        ocr_settings={"preprocess": True}
    )

    assert res["document_type"] == "Invoice"
    assert res["confidence"] != "Not Available"
    assert len(res["stages"]) == 6
    assert res["fields"]["Invoice Number"] == "INV-2026-001"
    assert "1,450.00" in res["fields"]["Total Amount"]
    assert len(res["missing_fields"]) == 0


def test_pipeline_invoice_missing_total(system_components):
    clf, ocr_engine, _ = system_components
    path = os.path.join(SAMPLES_DIR, "invoice_missing_total.pdf")
    with open(path, "rb") as f:
        data = f.read()

    res = process_document(
        file_bytes=data,
        filename="invoice_missing_total.pdf",
        file_extension="pdf",
        classifier=clf,
        ocr_engine=ocr_engine,
        active_model_name="Logistic Regression",
        ocr_settings={"preprocess": True}
    )

    assert res["document_type"] == "Invoice"
    assert res["fields"]["Total Amount"] == NOT_FOUND
    assert "Total Amount" in res["missing_fields"]


def test_pipeline_normal_resume(system_components):
    clf, ocr_engine, _ = system_components
    path = os.path.join(SAMPLES_DIR, "resume_1.pdf")
    with open(path, "rb") as f:
        data = f.read()

    res = process_document(
        file_bytes=data,
        filename="resume_1.pdf",
        file_extension="pdf",
        classifier=clf,
        ocr_engine=ocr_engine,
        active_model_name="Logistic Regression",
        ocr_settings={"preprocess": True}
    )

    assert res["document_type"] == "Resume"
    assert res["fields"]["Name"] == "Alex Smith"
    assert res["fields"]["Email"] == "alex.smith@email.com"
    assert "+91 98765 43210" in res["fields"]["Phone"]
    assert len(res["missing_fields"]) == 0


def test_pipeline_resume_missing_phone(system_components):
    clf, ocr_engine, _ = system_components
    path = os.path.join(SAMPLES_DIR, "resume_missing_phone.pdf")
    with open(path, "rb") as f:
        data = f.read()

    res = process_document(
        file_bytes=data,
        filename="resume_missing_phone.pdf",
        file_extension="pdf",
        classifier=clf,
        ocr_engine=ocr_engine,
        active_model_name="Logistic Regression",
        ocr_settings={"preprocess": True}
    )

    assert res["document_type"] == "Resume"
    assert res["fields"]["Phone"] == NOT_FOUND
    assert "Phone" in res["missing_fields"]


def test_pipeline_other_meeting_minutes(system_components):
    clf, ocr_engine, _ = system_components
    path = os.path.join(SAMPLES_DIR, "meeting_minutes.pdf")
    with open(path, "rb") as f:
        data = f.read()

    res = process_document(
        file_bytes=data,
        filename="meeting_minutes.pdf",
        file_extension="pdf",
        classifier=clf,
        ocr_engine=ocr_engine,
        active_model_name="Logistic Regression",
        ocr_settings={"preprocess": True}
    )

    assert res["document_type"] == "Other"
    assert len(res["stages"]) == 6
