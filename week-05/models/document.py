"""
Document model definitions, status constants, and workflow state definitions.
Week 5: Controlled Workflow States & Document Intelligence Metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

# Workflow & Processing Statuses
STATUS_NEW = "New"
STATUS_PROCESSING = "Processing"
STATUS_NEEDS_REVIEW = "Needs Review"
STATUS_APPROVED = "Approved"
STATUS_REJECTED = "Rejected"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

# Legacy / Aliased status for Week 4 backward compatibility
STATUS_PROCESSED = "Processed"

# Controlled workflow statuses
WORKFLOW_STATUSES = [
    STATUS_NEW,
    STATUS_PROCESSING,
    STATUS_NEEDS_REVIEW,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
]

# All recognized statuses including legacy 'Processed'
ALL_STATUSES = [
    STATUS_NEW,
    STATUS_PROCESSING,
    STATUS_NEEDS_REVIEW,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_COMPLETED,
    STATUS_PROCESSED,
    STATUS_FAILED,
]

# Document Categories
TYPE_INVOICE = "Invoice"
TYPE_RESUME = "Resume"
TYPE_OTHER = "Other"

ALL_DOCUMENT_TYPES = [TYPE_INVOICE, TYPE_RESUME, TYPE_OTHER]


@dataclass
class DocumentRecord:
    """Represents a persistent document entry in the system with Week 5 workflow attributes."""
    original_filename: str
    stored_filename: str
    document_type: str
    file_path: str
    file_hash: str
    status: str
    text_preview: str
    id: Optional[int] = None
    upload_date: str = field(default_factory=lambda: datetime.now().isoformat())
    company: str = "Not Found"
    invoice_number: str = "Not Found"
    total_amount: str = "Not Found"
    candidate_name: str = "Not Found"
    candidate_email: str = "Not Found"
    candidate_phone: str = "Not Found"
    candidate_skills: str = "Not Found"
    status_reason: str = ""
    metadata_json: str = "{}"
    current_status: Optional[str] = None
    predicted_type: Optional[str] = None
    confidence_score: Optional[float] = None
    validation_result_json: str = "{}"
    missing_fields_json: str = "[]"
    invalid_fields_json: str = "[]"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        if not self.current_status:
            self.current_status = self.status
        if not self.predicted_type:
            self.predicted_type = self.document_type
        if not self.created_at:
            self.created_at = self.upload_date
        if not self.updated_at:
            self.updated_at = self.upload_date

    def to_dict(self) -> Dict[str, Any]:
        """Converts model to dictionary for database insertion."""
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "document_type": self.document_type,
            "upload_date": self.upload_date,
            "company": self.company,
            "invoice_number": self.invoice_number,
            "total_amount": self.total_amount,
            "candidate_name": self.candidate_name,
            "candidate_email": self.candidate_email,
            "candidate_phone": self.candidate_phone,
            "candidate_skills": self.candidate_skills,
            "file_path": self.file_path,
            "text_preview": self.text_preview,
            "file_hash": self.file_hash,
            "status": self.status,
            "status_reason": self.status_reason,
            "metadata_json": self.metadata_json,
            "current_status": self.current_status or self.status,
            "predicted_type": self.predicted_type or self.document_type,
            "confidence_score": self.confidence_score,
            "validation_result_json": self.validation_result_json,
            "missing_fields_json": self.missing_fields_json,
            "invalid_fields_json": self.invalid_fields_json,
            "created_at": self.created_at or self.upload_date,
            "updated_at": self.updated_at or self.upload_date,
        }

    @classmethod
    def from_row(cls, row_dict: Dict[str, Any]) -> "DocumentRecord":
        """Instantiates DocumentRecord from SQLite row dictionary."""
        status = row_dict.get("status") or row_dict.get("current_status") or STATUS_COMPLETED
        upload_date = row_dict.get("upload_date") or datetime.now().isoformat()
        return cls(
            id=row_dict.get("id"),
            original_filename=row_dict.get("original_filename", ""),
            stored_filename=row_dict.get("stored_filename", ""),
            document_type=row_dict.get("document_type", "Other"),
            upload_date=upload_date,
            company=row_dict.get("company", "Not Found"),
            invoice_number=row_dict.get("invoice_number", "Not Found"),
            total_amount=row_dict.get("total_amount", "Not Found"),
            candidate_name=row_dict.get("candidate_name", "Not Found"),
            candidate_email=row_dict.get("candidate_email", "Not Found"),
            candidate_phone=row_dict.get("candidate_phone", "Not Found"),
            candidate_skills=row_dict.get("candidate_skills", "Not Found"),
            file_path=row_dict.get("file_path", ""),
            text_preview=row_dict.get("text_preview", ""),
            file_hash=row_dict.get("file_hash", ""),
            status=status,
            status_reason=row_dict.get("status_reason", ""),
            metadata_json=row_dict.get("metadata_json", "{}"),
            current_status=row_dict.get("current_status", status),
            predicted_type=row_dict.get("predicted_type", row_dict.get("document_type", "Other")),
            confidence_score=row_dict.get("confidence_score"),
            validation_result_json=row_dict.get("validation_result_json", "{}"),
            missing_fields_json=row_dict.get("missing_fields_json", "[]"),
            invalid_fields_json=row_dict.get("invalid_fields_json", "[]"),
            created_at=row_dict.get("created_at", upload_date),
            updated_at=row_dict.get("updated_at", upload_date),
        )
