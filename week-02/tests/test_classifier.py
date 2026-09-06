"""
Unit tests for document classification module (src/classifier.py).
Tests:
- Rule-based classifier on invoice, resume, and general text
- ML TF-IDF classifier predictions and explanations
- Hybrid ensemble consensus and fallbacks
- Low score threshold mapping to 'Other'
"""

import pytest
from src.classifier import RuleBasedClassifier, MLDocumentClassifier, HybridClassifier
from src.config import DOC_TYPE_INVOICE, DOC_TYPE_RESUME, DOC_TYPE_OTHER


@pytest.fixture
def rule_clf():
    return RuleBasedClassifier()


@pytest.fixture
def ml_clf():
    return MLDocumentClassifier()


@pytest.fixture
def hybrid_clf():
    return HybridClassifier()


def test_rule_based_invoice(rule_clf):
    text = "Tax Invoice # 9921 Date 01-01-2026 Total Amount Due PKR 50,000 Bill To Acme Corp"
    res = rule_clf.classify(text)
    assert res.document_type == DOC_TYPE_INVOICE
    assert res.confidence > 0.6
    assert res.invoice_score > res.resume_score
    assert len(res.top_keywords) > 0


def test_rule_based_resume(rule_clf):
    text = "Curriculum Vitae Jane Doe Skills Python Docker Work Experience Software Engineer Education BS Computer Science"
    res = rule_clf.classify(text)
    assert res.document_type == DOC_TYPE_RESUME
    assert res.confidence > 0.6
    assert res.resume_score > res.invoice_score


def test_rule_based_other(rule_clf):
    text = "The quick brown fox jumps over the lazy dog. Today we will discuss gardening tips and plant watering."
    res = rule_clf.classify(text)
    assert res.document_type == DOC_TYPE_OTHER


def test_ml_classifier_invoice(ml_clf):
    text = "Commercial Invoice INV-4001 Total Amount Due $5,000 Payment terms Net 30 days"
    res = ml_clf.classify(text)
    assert res.document_type == DOC_TYPE_INVOICE
    assert res.confidence > 0.5
    assert len(res.top_ml_features) > 0


def test_ml_classifier_resume(ml_clf):
    text = "Alex Smith Resume Professional Experience Machine Learning Engineer Education University of Oxford"
    res = ml_clf.classify(text)
    assert res.document_type == DOC_TYPE_RESUME
    assert res.confidence > 0.5


def test_hybrid_classifier_consensus(hybrid_clf):
    text = "Tax Invoice INV-1024 Date 02-09-2026 Total Amount PKR 125,000 Bill to ABC Technologies"
    res = hybrid_clf.classify(text)
    assert res.document_type == DOC_TYPE_INVOICE
    assert res.confidence >= 0.8
    assert res.confidence_label == "High confidence"
    assert "Consensus" in res.explanation or "Invoice" in res.explanation
