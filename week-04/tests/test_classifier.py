"""
Unit Tests for Document Classifier Module
"""

import os
import pytest
from src.classifier import DocumentClassifier, CLASS_INVOICE, CLASS_RESUME, CLASS_OTHER
from src.utils import load_dataset_from_dir


@pytest.fixture(scope="module")
def trained_classifier():
    clf = DocumentClassifier()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, "data", "train")
    texts, labels = load_dataset_from_dir(train_dir)
    clf.train(texts, labels)
    return clf


def test_classifier_invoice_prediction(trained_classifier):
    invoice_text = "Tax Invoice Invoice Number: INV-9901 Date: 2026-05-10 Total Amount Due: $ 2,500.00 Description Qty Unit Price"
    pred = trained_classifier.predict(invoice_text, model_name="Logistic Regression")
    assert pred["document_type"] == CLASS_INVOICE
    assert pred["confidence"] != "Not Available"
    assert pred["confidence_val"] is not None
    assert pred["confidence_val"] > 0.35  # Plurality threshold in 3-class distribution


def test_classifier_resume_prediction(trained_classifier):
    resume_text = "Alex Smith Resume Email: alex@test.com Phone: +91 9876543210 Skills: Python, Streamlit, PyTorch, SQL Education VTU"
    pred = trained_classifier.predict(resume_text, model_name="Logistic Regression")
    assert pred["document_type"] == CLASS_RESUME
    assert pred["confidence_val"] is not None
    assert pred["confidence_val"] > 0.35


def test_classifier_other_prediction(trained_classifier):
    other_text = "Quarterly Strategic Management Meeting Minutes. Discussed project deliverables, timeline updates, and sprint retrospectives."
    pred = trained_classifier.predict(other_text, model_name="Logistic Regression")
    assert pred["document_type"] == CLASS_OTHER


def test_rule_based_baseline(trained_classifier):
    sample = "INVOICE Bill To Client Inc Total: $ 500.00"
    pred = trained_classifier.predict(sample, model_name="Rule-Based Baseline")
    assert pred["document_type"] == CLASS_INVOICE
    assert pred["confidence"] == "Not Available"  # Rule-based honestly reports Not Available


def test_model_persistence(trained_classifier, tmp_path):
    save_file = str(tmp_path / "test_model.pkl")
    trained_classifier.save_model(save_file)
    assert os.path.isfile(save_file)

    new_clf = DocumentClassifier()
    loaded = new_clf.load_model(save_file)
    assert loaded is True
    assert new_clf.is_trained is True

    test_pred = new_clf.predict("Invoice No: 123 Total Amount: $100")
    assert test_pred["document_type"] == CLASS_INVOICE
