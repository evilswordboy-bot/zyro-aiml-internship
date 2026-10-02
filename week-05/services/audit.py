"""
Audit Logging Service for AI Document Intelligence & Workflow Platform.
Provides immutable, chronological tracking of all document transitions and reviewer interactions.
"""

from typing import Dict, Any, List, Optional
from database.db import DatabaseManager, get_db

# Audit Actions
ACTION_UPLOADED = "UPLOADED"
ACTION_PROCESSING_STARTED = "PROCESSING_STARTED"
ACTION_CLASSIFIED = "CLASSIFIED"
ACTION_EXTRACTED = "EXTRACTED"
ACTION_VALIDATION_PASSED = "VALIDATION_PASSED"
ACTION_VALIDATION_FAILED = "VALIDATION_FAILED"
ACTION_ROUTED_TO_REVIEW = "ROUTED_TO_REVIEW"
ACTION_ROUTED_TO_COMPLETED = "ROUTED_TO_COMPLETED"
ACTION_REVIEWER_APPROVED = "REVIEWER_APPROVED"
ACTION_REVIEWER_REJECTED = "REVIEWER_REJECTED"
ACTION_WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
ACTION_WORKFLOW_FAILED = "WORKFLOW_FAILED"
ACTION_RETRIED = "RETRIED"

ALL_ACTIONS = [
    ACTION_UPLOADED,
    ACTION_PROCESSING_STARTED,
    ACTION_CLASSIFIED,
    ACTION_EXTRACTED,
    ACTION_VALIDATION_PASSED,
    ACTION_VALIDATION_FAILED,
    ACTION_ROUTED_TO_REVIEW,
    ACTION_ROUTED_TO_COMPLETED,
    ACTION_REVIEWER_APPROVED,
    ACTION_REVIEWER_REJECTED,
    ACTION_WORKFLOW_COMPLETED,
    ACTION_WORKFLOW_FAILED,
    ACTION_RETRIED,
]


class AuditService:
    """Manages the creation and retrieval of immutable audit events."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db()

    def record_event(
        self,
        document_id: int,
        action: str,
        previous_status: Optional[str],
        new_status: str,
        reason: str = "",
        reviewer_note: str = ""
    ) -> int:
        """
        Records a workflow lifecycle event in SQLite.
        """
        return self.db.log_audit_event(
            document_id=document_id,
            action=action,
            previous_status=previous_status,
            new_status=new_status,
            reason=reason,
            reviewer_note=reviewer_note
        )

    def get_document_history(self, document_id: int) -> List[Dict[str, Any]]:
        """Retrieves chronological timeline for a specific document."""
        return self.db.get_document_audit_history(document_id)

    def get_recent_timeline(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves global audit timeline with joined document info."""
        return self.db.get_recent_audit_events(limit=limit)


_audit_singleton: Optional[AuditService] = None


def get_audit_service(db_manager: Optional[DatabaseManager] = None) -> AuditService:
    """Returns singleton or dedicated instance of AuditService."""
    global _audit_singleton
    if db_manager is not None:
        return AuditService(db_manager)
    if _audit_singleton is None:
        _audit_singleton = AuditService()
    return _audit_singleton
