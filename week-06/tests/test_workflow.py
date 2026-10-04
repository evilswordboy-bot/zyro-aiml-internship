"""
Unit tests for WorkflowEngine (Week 5).
Validates state machine transitions, invalid transition blocking, decision rule evaluation,
and human review approval / rejection with mandatory note enforcement.
"""

import os
import tempfile
import pytest
from database.db import DatabaseManager
from services.audit import AuditService
from services.workflow import (
    WorkflowEngine,
    STATUS_NEW,
    STATUS_PROCESSING,
    STATUS_NEEDS_REVIEW,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
)


@pytest.fixture
def workflow_setup():
    """Builds isolated DatabaseManager, AuditService, and WorkflowEngine."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_workflow.db")
        db = DatabaseManager(db_path=db_path)
        audit = AuditService(db)
        engine = WorkflowEngine(db, audit)
        yield db, audit, engine


def test_state_transition_validation():
    # Valid transitions
    assert WorkflowEngine.can_transition(STATUS_NEW, STATUS_PROCESSING)[0] is True
    assert WorkflowEngine.can_transition(STATUS_PROCESSING, STATUS_COMPLETED)[0] is True
    assert WorkflowEngine.can_transition(STATUS_PROCESSING, STATUS_NEEDS_REVIEW)[0] is True
    assert WorkflowEngine.can_transition(STATUS_NEEDS_REVIEW, STATUS_APPROVED)[0] is True
    assert WorkflowEngine.can_transition(STATUS_NEEDS_REVIEW, STATUS_REJECTED)[0] is True
    assert WorkflowEngine.can_transition(STATUS_APPROVED, STATUS_COMPLETED)[0] is True
    assert WorkflowEngine.can_transition(STATUS_REJECTED, STATUS_PROCESSING)[0] is True
    assert WorkflowEngine.can_transition(STATUS_FAILED, STATUS_PROCESSING)[0] is True

    # Blocked invalid transitions
    assert WorkflowEngine.can_transition(STATUS_NEW, STATUS_COMPLETED)[0] is False
    assert WorkflowEngine.can_transition(STATUS_NEW, STATUS_APPROVED)[0] is False
    assert WorkflowEngine.can_transition(STATUS_COMPLETED, STATUS_APPROVED)[0] is False
    assert WorkflowEngine.can_transition(STATUS_REJECTED, STATUS_COMPLETED)[0] is False


def test_evaluate_rules_decision_logic(workflow_setup):
    _, _, engine = workflow_setup

    # 1. Unreadable / empty document -> Failed
    status, action, reason = engine.evaluate_rules(
        doc_type="Invoice",
        validation_result={"is_valid": True},
        confidence=0.95,
        text_valid=False
    )
    assert status == STATUS_FAILED

    # 2. Validation failed -> Needs Review
    status, action, reason = engine.evaluate_rules(
        doc_type="Invoice",
        validation_result={"is_valid": False, "reasons": ["Missing Total Amount"]},
        confidence=0.92,
        text_valid=True
    )
    assert status == STATUS_NEEDS_REVIEW

    # 3. Confidence below 0.70 threshold -> Needs Review
    status, action, reason = engine.evaluate_rules(
        doc_type="Invoice",
        validation_result={"is_valid": True},
        confidence=0.65,
        text_valid=True
    )
    assert status == STATUS_NEEDS_REVIEW
    assert "below approval threshold" in reason

    # 4. Valid and high confidence -> Completed
    status, action, reason = engine.evaluate_rules(
        doc_type="Invoice",
        validation_result={"is_valid": True},
        confidence=0.88,
        text_valid=True
    )
    assert status == STATUS_COMPLETED


def test_approve_document_flow(workflow_setup):
    db, audit, engine = workflow_setup

    # Insert a document in 'Needs Review'
    doc_id = db.add_document({
        "original_filename": "test_inv.pdf",
        "stored_filename": "test_inv.pdf",
        "document_type": "Invoice",
        "status": STATUS_NEEDS_REVIEW,
        "current_status": STATUS_NEEDS_REVIEW,
        "file_hash": "hash_123"
    })

    ok, msg = engine.approve_document(doc_id, reviewer_note="Manually verified by finance manager.")
    assert ok is True

    # Document must now be Completed
    doc = db.get_document_by_id(doc_id)
    assert doc["status"] == STATUS_COMPLETED
    assert doc["current_status"] == STATUS_COMPLETED

    # Verify audit trail contains reviewer approval and completion
    history = audit.get_document_history(doc_id)
    actions = [h["action"] for h in history]
    assert "REVIEWER_APPROVED" in actions
    assert "WORKFLOW_COMPLETED" in actions
    assert any("finance manager" in h["reviewer_note"] for h in history)


def test_reject_document_mandatory_note(workflow_setup):
    db, audit, engine = workflow_setup

    doc_id = db.add_document({
        "original_filename": "test_bad.pdf",
        "stored_filename": "test_bad.pdf",
        "document_type": "Resume",
        "status": STATUS_NEEDS_REVIEW,
        "current_status": STATUS_NEEDS_REVIEW,
        "file_hash": "hash_456"
    })

    # Attempt rejection without note -> MUST BE BLOCKED
    ok_empty, err = engine.reject_document(doc_id, reviewer_note="")
    assert ok_empty is False
    assert "mandatory" in err.lower()

    # Document must still be in Needs Review
    doc = db.get_document_by_id(doc_id)
    assert doc["status"] == STATUS_NEEDS_REVIEW

    # Now reject with mandatory explanation note -> SUCCEEDS
    ok_valid, msg = engine.reject_document(doc_id, reviewer_note="Candidate phone number appears fraudulent.")
    assert ok_valid is True

    doc_after = db.get_document_by_id(doc_id)
    assert doc_after["status"] == STATUS_REJECTED
    assert doc_after["current_status"] == STATUS_REJECTED

    history = audit.get_document_history(doc_id)
    assert any(h["action"] == "REVIEWER_REJECTED" and "fraudulent" in h["reviewer_note"] for h in history)
