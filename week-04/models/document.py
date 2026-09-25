"""
Document model definitions and status constants.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

# Processing Statuses
STATUS_PROCESSED = "Processed"
STATUS_NEEDS_REVIEW = "Needs Review"
STATUS_FAILED = "Failed"

ALL_STATUSES = [STATUS_PROCESSED, STATUS_NEEDS_REVIEW, STATUS_FAILED]

# Document Categories
TYPE_INVOICE = "Invoice"
TYPE_RESUME = "Resume"
TYPE_OTHER = "Other"

ALL_DOCUMENT_TYPES = [TYPE_INVOICE, TYPE_RESUME, TYPE_OTHER]


@dataclass
class DocumentRecord:
    """Represents a persistent document entry in the system."""
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
            "metadata_json": self.metadata_json
        }

    @classmethod
    def from_row(cls, row_dict: Dict[str, Any]) -> "DocumentRecord":
        """Instantiates DocumentRecord from SQLite row dictionary."""
        return cls(
            id=row_dict.get("id"),
            original_filename=row_dict.get("original_filename", ""),
            stored_filename=row_dict.get("stored_filename", ""),
            document_type=row_dict.get("document_type", "Other"),
            upload_date=row_dict.get("upload_date", ""),
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
            status=row_dict.get("status", STATUS_PROCESSED),
            status_reason=row_dict.get("status_reason", ""),
            metadata_json=row_dict.get("metadata_json", "{}")
        )
