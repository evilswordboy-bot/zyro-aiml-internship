"""
Rule-Based Workflow Engine for AI Document Intelligence & Workflow Platform.
Enforces state machine transitions, automated routing decisions, and human review approval/rejection.
"""

from typing import Dict, Any, Tuple, Optional, List, Union
from database.db import DatabaseManager, get_db
from models.document import (
    STATUS_NEW,
    STATUS_PROCESSING,
    STATUS_NEEDS_REVIEW,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_COMPLETED,
    STATUS_PROCESSED,
    STATUS_FAILED,
)
from services.audit import (
    AuditService,
    get_audit_service,
    ACTION_ROUTED_TO_REVIEW,
    ACTION_ROUTED_TO_COMPLETED,
    ACTION_REVIEWER_APPROVED,
    ACTION_REVIEWER_REJECTED,
    ACTION_WORKFLOW_COMPLETED,
    ACTION_WORKFLOW_FAILED,
    ACTION_RETRIED,
)

# Confidence threshold for automated straight-through processing
CONFIDENCE_THRESHOLD = 0.70

# Strict State Transition Matrix
ALLOWED_TRANSITIONS = {
    STATUS_NEW: [STATUS_PROCESSING, STATUS_FAILED],
    STATUS_PROCESSING: [STATUS_NEEDS_REVIEW, STATUS_COMPLETED, STATUS_PROCESSED, STATUS_FAILED],
    STATUS_NEEDS_REVIEW: [STATUS_APPROVED, STATUS_REJECTED, STATUS_PROCESSING],
    STATUS_APPROVED: [STATUS_COMPLETED, STATUS_PROCESSED],
    STATUS_REJECTED: [STATUS_PROCESSING],
    STATUS_COMPLETED: [STATUS_PROCESSING],
    STATUS_PROCESSED: [STATUS_PROCESSING, STATUS_COMPLETED],
    STATUS_FAILED: [STATUS_PROCESSING],
}


class WorkflowEngine:
    """Controls lifecycle state transitions, business rules routing, and review actions."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        audit_service: Optional[AuditService] = None
    ):
        self.db = db_manager or get_db()
        self.audit = audit_service or get_audit_service(self.db)

    @staticmethod
    def can_transition(current_status: str, target_status: str) -> Tuple[bool, str]:
        """
        Validates whether a transition from current_status to target_status is allowed.
        """
        curr = current_status or STATUS_NEW
        allowed_targets = ALLOWED_TRANSITIONS.get(curr, [])
        if target_status in allowed_targets:
            return True, f"Valid transition: {curr} -> {target_status}"
        return False, (
            f"Blocked invalid transition: Cannot move document from state '{curr}' to '{target_status}'. "
            f"Allowed next states from '{curr}': {', '.join(allowed_targets) if allowed_targets else 'None'}."
        )

    def transition_document(
        self,
        doc_id: int,
        target_status: str,
        action: str,
        reason: str = "",
        reviewer_note: str = ""
    ) -> Tuple[bool, str]:
        """
        Executes an audited state transition on a document.
        Guards against unauthorized state jumps.
        """
        doc = self.db.get_document_by_id(doc_id)
        if not doc:
            return False, f"Document #{doc_id} not found."

        current_status = doc.get("current_status") or doc.get("status") or STATUS_NEW
        valid, msg = self.can_transition(current_status, target_status)
        if not valid:
            return False, msg

        # Update document in database
        self.db.update_document_status(doc_id, target_status, reason)

        # Record audit log
        self.audit.record_event(
            document_id=doc_id,
            action=action,
            previous_status=current_status,
            new_status=target_status,
            reason=reason,
            reviewer_note=reviewer_note
        )

        return True, f"Document #{doc_id} transitioned from '{current_status}' to '{target_status}'."

    def evaluate_rules(
        self,
        doc_type: str,
        validation_result: Dict[str, Any],
        confidence: Union[float, str, None],
        text_valid: bool = True
    ) -> Tuple[str, str, str]:
        """
        Applies automated workflow business rules to determine document routing.

        Returns:
            Tuple[target_status, action, reason]
        """
        # Rule 1: Text extraction failure or empty/corrupt document
        if not text_valid:
            return (
                STATUS_FAILED,
                ACTION_WORKFLOW_FAILED,
                "Document text extraction failed or insufficient legible content."
            )

        # Rule 2: Structural or entity validation failure
        if not validation_result.get("is_valid", False):
            reasons = validation_result.get("reasons", ["Document validation checks failed."])
            reasons_str = "; ".join(reasons)
            return (
                STATUS_NEEDS_REVIEW,
                ACTION_ROUTED_TO_REVIEW,
                reasons_str
            )

        # Parse confidence if string like "85%"
        conf_float: Optional[float] = None
        if isinstance(confidence, (int, float)):
            conf_float = float(confidence)
        elif isinstance(confidence, str) and "%" in confidence:
            try:
                conf_float = float(confidence.replace("%", "").strip()) / 100.0
            except ValueError:
                conf_float = None

        # Rule 3: Low model confidence (when calibrated probability is available)
        # Note: Linear SVM and rule baseline do not provide calibrated probabilities (confidence=None).
        # We do not invent fake confidence scores.
        if conf_float is not None and conf_float < CONFIDENCE_THRESHOLD:
            pct = conf_float * 100.0
            return (
                STATUS_NEEDS_REVIEW,
                ACTION_ROUTED_TO_REVIEW,
                f"Classification confidence ({pct:.1f}%) is below approval threshold ({CONFIDENCE_THRESHOLD * 100:.0f}%)."
            )

        # Rule 4: Everything valid and verified
        return (
            STATUS_COMPLETED,
            ACTION_ROUTED_TO_COMPLETED,
            "All structural, field, and confidence validation criteria passed successfully."
        )

    def approve_document(self, doc_id: int, reviewer_note: str = "") -> Tuple[bool, str]:
        """
        Human reviewer approves a document from 'Needs Review'.
        Moves state: Needs Review -> Approved -> Completed.
        """
        doc = self.db.get_document_by_id(doc_id)
        if not doc:
            return False, f"Document #{doc_id} not found."

        curr = doc.get("current_status") or doc.get("status")
        if curr != STATUS_NEEDS_REVIEW:
            return False, f"Only documents in '{STATUS_NEEDS_REVIEW}' can be approved (current state is '{curr}')."

        # 1. Transition to Approved
        ok1, msg1 = self.transition_document(
            doc_id=doc_id,
            target_status=STATUS_APPROVED,
            action=ACTION_REVIEWER_APPROVED,
            reason="Approved by human reviewer.",
            reviewer_note=reviewer_note
        )
        if not ok1:
            return False, msg1

        # 2. Complete workflow
        ok2, msg2 = self.transition_document(
            doc_id=doc_id,
            target_status=STATUS_COMPLETED,
            action=ACTION_WORKFLOW_COMPLETED,
            reason="Workflow completed following human review approval.",
            reviewer_note=reviewer_note
        )
        if not ok2:
            return False, msg2

        return True, f"Document #{doc_id} successfully approved and marked Completed."

    def reject_document(self, doc_id: int, reviewer_note: str) -> Tuple[bool, str]:
        """
        Human reviewer rejects a document from 'Needs Review'.
        MANDATORY REQUIREMENT: reviewer_note must be non-empty explaining the rejection.
        """
        if not reviewer_note or not reviewer_note.strip():
            return False, "Rejection note is strictly mandatory: please provide a clear reason for rejection."

        doc = self.db.get_document_by_id(doc_id)
        if not doc:
            return False, f"Document #{doc_id} not found."

        curr = doc.get("current_status") or doc.get("status")
        if curr != STATUS_NEEDS_REVIEW:
            return False, f"Only documents in '{STATUS_NEEDS_REVIEW}' can be rejected (current state is '{curr}')."

        ok, msg = self.transition_document(
            doc_id=doc_id,
            target_status=STATUS_REJECTED,
            action=ACTION_REVIEWER_REJECTED,
            reason="Rejected by human reviewer.",
            reviewer_note=reviewer_note.strip()
        )
        return ok, msg

    def reprocess_document(self, doc_id: int, reviewer_note: str = "") -> Tuple[bool, str]:
        """
        Moves document back to 'Processing' for re-analysis or retry.
        """
        doc = self.db.get_document_by_id(doc_id)
        if not doc:
            return False, f"Document #{doc_id} not found."

        curr = doc.get("current_status") or doc.get("status")
        valid, msg = self.can_transition(curr, STATUS_PROCESSING)
        if not valid:
            return False, msg

        ok, msg = self.transition_document(
            doc_id=doc_id,
            target_status=STATUS_PROCESSING,
            action=ACTION_RETRIED,
            reason="Queued for pipeline reprocessing.",
            reviewer_note=reviewer_note
        )
        return ok, msg


_workflow_singleton: Optional[WorkflowEngine] = None


def get_workflow_engine(
    db_manager: Optional[DatabaseManager] = None,
    audit_service: Optional[AuditService] = None
) -> WorkflowEngine:
    """Returns singleton or dedicated instance of WorkflowEngine."""
    global _workflow_singleton
    if db_manager is not None or audit_service is not None:
        return WorkflowEngine(db_manager, audit_service)
    if _workflow_singleton is None:
        _workflow_singleton = WorkflowEngine()
    return _workflow_singleton
