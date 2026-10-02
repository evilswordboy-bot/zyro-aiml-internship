"""
Document Processor Service for AI Document Intelligence & Workflow Platform.
Orchestrates the complete Week 5 automated lifecycle:
UPLOAD -> PROCESS -> CLASSIFY -> EXTRACT -> VALIDATE -> APPLY RULES -> REVIEW / APPROVE / REJECT -> COMPLETE -> AUDIT HISTORY
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from database.db import DatabaseManager, get_db
from models.document import (
    DocumentRecord,
    STATUS_NEW,
    STATUS_PROCESSING,
    STATUS_NEEDS_REVIEW,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_COMPLETED,
    STATUS_PROCESSED,
    STATUS_FAILED,
    TYPE_INVOICE,
    TYPE_RESUME,
    TYPE_OTHER
)
from services.hashing import calculate_sha256
from services.file_storage import FileStorageManager, get_storage_manager
from services.validation import FileValidator, DocumentValidator
from services.audit import (
    AuditService,
    get_audit_service,
    ACTION_UPLOADED,
    ACTION_PROCESSING_STARTED,
    ACTION_CLASSIFIED,
    ACTION_EXTRACTED,
    ACTION_VALIDATION_PASSED,
    ACTION_VALIDATION_FAILED,
    ACTION_ROUTED_TO_REVIEW,
    ACTION_ROUTED_TO_COMPLETED,
    ACTION_WORKFLOW_FAILED,
)
from services.workflow import WorkflowEngine, get_workflow_engine
from src.document_reader import DocumentReader
from src.ocr_processor import OCRProcessor
from src.text_cleaner import TextCleaner
from src.classifier import DocumentClassifier
from src.extractor import DocumentExtractor, NOT_FOUND


class DocumentProcessor:
    """End-to-end controller for document intake, intelligence pipeline, validation, rules, and audit persistence."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        storage_manager: Optional[FileStorageManager] = None,
        classifier: Optional[DocumentClassifier] = None,
        ocr_engine: Optional[OCRProcessor] = None,
        audit_service: Optional[AuditService] = None,
        workflow_engine: Optional[WorkflowEngine] = None
    ):
        self.db = db_manager or get_db()
        self.storage = storage_manager or get_storage_manager()
        self.classifier = classifier or DocumentClassifier()
        self.ocr_engine = ocr_engine or OCRProcessor()
        self.audit = audit_service or get_audit_service(self.db)
        self.workflow = workflow_engine or get_workflow_engine(self.db, self.audit)

    def process_and_store(
        self,
        file_bytes: bytes,
        original_filename: str,
        active_model_name: str = "Logistic Regression",
        ocr_settings: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full document intelligence workflow:
        Intake -> Hashing -> Duplicate Suppression -> Text/OCR -> Classification
        -> Extraction -> Advanced Validation -> Rule-Based Routing -> Audit Trail -> Storage.
        """
        ocr_opts = ocr_settings or {"preprocess": True}
        stages = []

        # -------------------------------------------------------------
        # 1. File Validation
        # -------------------------------------------------------------
        is_valid, ext_clean, val_error = FileValidator.validate_file(original_filename, file_bytes)
        if not is_valid:
            return {
                "success": False,
                "is_duplicate": False,
                "error": val_error,
                "status": STATUS_FAILED,
                "stages": ["File Validation Failed"]
            }
        stages.append("File Validated")

        # -------------------------------------------------------------
        # 2. SHA-256 Hashing
        # -------------------------------------------------------------
        file_hash = calculate_sha256(file_bytes)
        stages.append("SHA-256 Computed")

        # -------------------------------------------------------------
        # 3. Duplicate Detection Check
        # -------------------------------------------------------------
        existing_doc = self.db.get_document_by_hash(file_hash)
        if existing_doc:
            stages.append("Duplicate Detected")
            return {
                "success": True,
                "is_duplicate": True,
                "file_hash": file_hash,
                "message": (
                    f"Duplicate document detected! '{original_filename}' is identical to previously "
                    f"uploaded '{existing_doc['original_filename']}' (ID #{existing_doc['id']}). "
                    "Skipping duplicate file creation and database write."
                ),
                "document": existing_doc,
                "status": existing_doc.get("status") or existing_doc.get("current_status"),
                "stages": stages
            }

        # -------------------------------------------------------------
        # 4. Text Extraction (PyMuPDF with OCR Fallback)
        # -------------------------------------------------------------
        raw_text = ""
        ocr_used = False
        ocr_steps = []
        extraction_note = ""

        if ext_clean == "pdf":
            read_res = DocumentReader.read_pdf(file_bytes)
            if read_res["error"]:
                return self._store_failed_document(
                    file_bytes, original_filename, file_hash, "Unreadable",
                    read_res["error"], stages
                )

            if read_res["needs_ocr"]:
                ocr_res = self.ocr_engine.ocr_pdf(file_bytes, preprocess=ocr_opts.get("preprocess", True))
                raw_text = ocr_res["text"]
                ocr_used = True
                ocr_steps = ocr_res.get("preprocessing_steps", [])
                extraction_note = "Scanned PDF detected — executed OCR computer vision fallback."
            else:
                raw_text = read_res["text"]
                extraction_note = f"Direct PyMuPDF selectable text extracted ({read_res['page_count']} page(s))."
        else:  # Images: jpg, jpeg, png
            ocr_res = self.ocr_engine.extract_from_image(file_bytes, preprocess=ocr_opts.get("preprocess", True))
            raw_text = ocr_res["text"]
            ocr_used = True
            ocr_steps = ocr_res.get("preprocessing_steps", [])
            extraction_note = "Image processed via OCR computer vision engine."

        stages.append("Text Extracted")

        # -------------------------------------------------------------
        # 5. Text Cleaning & Normalization
        # -------------------------------------------------------------
        clean_res = TextCleaner.clean(raw_text)
        cleaned_text = clean_res["cleaned_text"]
        stages.append("Text Cleaned")

        if not clean_res["is_valid"]:
            return self._store_failed_document(
                file_bytes, original_filename, file_hash, "Unreadable",
                clean_res.get("warning") or "Insufficient text extracted to process document.",
                stages, raw_text=raw_text, ocr_used=ocr_used
            )

        # -------------------------------------------------------------
        # 6. ML Classification
        # -------------------------------------------------------------
        pred_res = self.classifier.predict(cleaned_text, model_name=active_model_name)
        doc_type = pred_res["document_type"]
        confidence_str = pred_res.get("confidence", "Not Available")
        confidence_val = pred_res.get("confidence_val")
        stages.append("Document Classified")

        # -------------------------------------------------------------
        # 7. Entity Extraction
        # -------------------------------------------------------------
        extract_res = DocumentExtractor.extract(cleaned_text, doc_type)
        fields = extract_res.get("fields", {})
        missing_fields = extract_res.get("missing_fields", [])
        stages.append("Fields Extracted")

        # -------------------------------------------------------------
        # 8. Advanced Document Validation (Week 5)
        # -------------------------------------------------------------
        val_result = DocumentValidator.validate_document(doc_type, fields, cleaned_text)
        stages.append("Validation Completed")

        # -------------------------------------------------------------
        # 9. Rule-Based Workflow Engine Routing (Week 5)
        # -------------------------------------------------------------
        target_status, routing_action, routing_reason = self.workflow.evaluate_rules(
            doc_type=doc_type,
            validation_result=val_result,
            confidence=confidence_val if confidence_val is not None else confidence_str,
            text_valid=True
        )
        stages.append(f"Routed to {target_status}")

        # -------------------------------------------------------------
        # 10. Structured Physical Storage
        # -------------------------------------------------------------
        stored_filename, relative_path = self.storage.save_file(
            file_bytes=file_bytes,
            original_filename=original_filename,
            doc_type=doc_type,
            file_hash=file_hash
        )
        stages.append("File Stored")

        # -------------------------------------------------------------
        # 11. Database Persistence
        # -------------------------------------------------------------
        company = fields.get("Company Name", NOT_FOUND)
        invoice_number = fields.get("Invoice Number", NOT_FOUND)
        total_amount = fields.get("Total Amount", NOT_FOUND)

        candidate_name = fields.get("Name", NOT_FOUND)
        candidate_email = fields.get("Email", NOT_FOUND)
        candidate_phone = fields.get("Phone", NOT_FOUND)
        candidate_skills = fields.get("Skills", NOT_FOUND)

        metadata_dict = {
            "classification_model": active_model_name,
            "confidence": confidence_str,
            "confidence_val": confidence_val,
            "probabilities": pred_res.get("probabilities"),
            "ocr_used": ocr_used,
            "ocr_steps": ocr_steps,
            "extraction_note": extraction_note,
            "missing_fields": val_result.get("missing_fields", missing_fields),
            "invalid_fields": val_result.get("invalid_fields", []),
            "passed_fields": val_result.get("passed_fields", []),
            "validation_reasons": val_result.get("reasons", []),
            "word_count": len(cleaned_text.split()),
            "char_count_original": clean_res["char_count_original"],
            "char_count_cleaned": clean_res["char_count_cleaned"],
            "reduction_pct": clean_res["reduction_pct"],
            "fields": fields
        }

        doc_record = {
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "document_type": doc_type,
            "upload_date": datetime.now().isoformat(),
            "company": company,
            "invoice_number": invoice_number,
            "total_amount": total_amount,
            "candidate_name": candidate_name,
            "candidate_email": candidate_email,
            "candidate_phone": candidate_phone,
            "candidate_skills": candidate_skills,
            "file_path": relative_path,
            "text_preview": cleaned_text[:1000],
            "file_hash": file_hash,
            "status": target_status,
            "current_status": target_status,
            "status_reason": routing_reason,
            "predicted_type": doc_type,
            "confidence_score": confidence_val,
            "validation_result_json": val_result,
            "missing_fields_json": val_result.get("missing_fields", []),
            "invalid_fields_json": val_result.get("invalid_fields", []),
            "metadata_json": metadata_dict
        }

        doc_id = self.db.add_document(doc_record)
        stages.append("Metadata Saved to SQLite")

        # -------------------------------------------------------------
        # 12. Audit Logging Trail (Week 5)
        # -------------------------------------------------------------
        # Event 1: Intake Uploaded
        self.audit.record_event(
            document_id=doc_id,
            action=ACTION_UPLOADED,
            previous_status=None,
            new_status=STATUS_NEW,
            reason=f"Uploaded '{original_filename}' ({len(file_bytes)} bytes)."
        )

        # Event 2: Processing Pipeline Triggered
        self.audit.record_event(
            document_id=doc_id,
            action=ACTION_PROCESSING_STARTED,
            previous_status=STATUS_NEW,
            new_status=STATUS_PROCESSING,
            reason=f"Executed OCR/PyMuPDF text extraction ({extraction_note})."
        )

        # Event 3: Classification
        if confidence_val is not None:
            conf_str = f"confidence {confidence_val:.1%}"
        elif confidence_str and confidence_str != "Not Available":
            conf_str = f"confidence {confidence_str}"
        else:
            conf_str = "confidence uncalibrated"

        self.audit.record_event(
            document_id=doc_id,
            action=ACTION_CLASSIFIED,
            previous_status=STATUS_PROCESSING,
            new_status=STATUS_PROCESSING,
            reason=f"Classified as '{doc_type}' with {conf_str} via {active_model_name}."
        )

        # Event 4: Entity Extraction
        self.audit.record_event(
            document_id=doc_id,
            action=ACTION_EXTRACTED,
            previous_status=STATUS_PROCESSING,
            new_status=STATUS_PROCESSING,
            reason=f"Extracted {len(fields)} fields ({len(missing_fields)} missing)."
        )

        # Event 5: Validation Result
        val_action = ACTION_VALIDATION_PASSED if val_result["is_valid"] else ACTION_VALIDATION_FAILED
        val_summary = "All field rules verified" if val_result["is_valid"] else "; ".join(val_result.get("reasons", ["Validation failed"]))
        self.audit.record_event(
            document_id=doc_id,
            action=val_action,
            previous_status=STATUS_PROCESSING,
            new_status=STATUS_PROCESSING,
            reason=val_summary
        )

        # Event 6: Final Routing to Target Status (Completed or Needs Review)
        self.audit.record_event(
            document_id=doc_id,
            action=routing_action,
            previous_status=STATUS_PROCESSING,
            new_status=target_status,
            reason=routing_reason
        )

        saved_doc = self.db.get_document_by_id(doc_id)

        return {
            "success": True,
            "is_duplicate": False,
            "document_id": doc_id,
            "document": saved_doc,
            "file_hash": file_hash,
            "document_type": doc_type,
            "confidence": confidence_str,
            "confidence_val": confidence_val,
            "status": target_status,
            "status_reason": routing_reason,
            "validation": val_result,
            "fields": fields,
            "missing_fields": val_result.get("missing_fields", missing_fields),
            "invalid_fields": val_result.get("invalid_fields", []),
            "passed_fields": val_result.get("passed_fields", []),
            "original_text": raw_text,
            "cleaned_text": cleaned_text,
            "stored_filename": stored_filename,
            "file_path": relative_path,
            "stages": stages,
            "ocr_used": ocr_used,
            "ocr_steps": ocr_steps,
            "error": None
        }

    def _store_failed_document(
        self,
        file_bytes: bytes,
        original_filename: str,
        file_hash: str,
        doc_type: str,
        error_msg: str,
        stages: list,
        raw_text: str = "",
        ocr_used: bool = False
    ) -> Dict[str, Any]:
        """Safely saves unreadable or corrupt documents with 'Failed' status and full audit logging."""
        stored_filename, relative_path = self.storage.save_file(
            file_bytes, original_filename, "Other", file_hash
        )

        doc_record = {
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "document_type": doc_type,
            "upload_date": datetime.now().isoformat(),
            "company": NOT_FOUND,
            "invoice_number": NOT_FOUND,
            "total_amount": NOT_FOUND,
            "candidate_name": NOT_FOUND,
            "candidate_email": NOT_FOUND,
            "candidate_phone": NOT_FOUND,
            "candidate_skills": NOT_FOUND,
            "file_path": relative_path,
            "text_preview": raw_text[:500] if raw_text else "No legible text available.",
            "file_hash": file_hash,
            "status": STATUS_FAILED,
            "current_status": STATUS_FAILED,
            "status_reason": error_msg,
            "predicted_type": doc_type,
            "confidence_score": None,
            "validation_result_json": {"is_valid": False, "reasons": [error_msg]},
            "missing_fields_json": [],
            "invalid_fields_json": [],
            "metadata_json": {
                "error": error_msg,
                "ocr_used": ocr_used,
                "stages": stages + ["Failed"]
            }
        }
        doc_id = self.db.add_document(doc_record)

        # Audit events for failed intake
        self.audit.record_event(
            document_id=doc_id,
            action=ACTION_UPLOADED,
            previous_status=None,
            new_status=STATUS_NEW,
            reason=f"Uploaded '{original_filename}' ({len(file_bytes)} bytes)."
        )
        self.audit.record_event(
            document_id=doc_id,
            action=ACTION_WORKFLOW_FAILED,
            previous_status=STATUS_NEW,
            new_status=STATUS_FAILED,
            reason=error_msg
        )

        saved_doc = self.db.get_document_by_id(doc_id)

        return {
            "success": False,
            "is_duplicate": False,
            "document_id": doc_id,
            "document": saved_doc,
            "file_hash": file_hash,
            "document_type": doc_type,
            "status": STATUS_FAILED,
            "status_reason": error_msg,
            "error": error_msg,
            "stages": stages + ["Failed"]
        }


_processor_singleton: Optional[DocumentProcessor] = None


def get_document_processor(
    db_manager: Optional[DatabaseManager] = None,
    storage_manager: Optional[FileStorageManager] = None,
    classifier: Optional[DocumentClassifier] = None,
    ocr_engine: Optional[OCRProcessor] = None,
    audit_service: Optional[AuditService] = None,
    workflow_engine: Optional[WorkflowEngine] = None
) -> DocumentProcessor:
    """Returns singleton instance of DocumentProcessor."""
    global _processor_singleton
    if _processor_singleton is None:
        _processor_singleton = DocumentProcessor(
            db_manager, storage_manager, classifier, ocr_engine, audit_service, workflow_engine
        )
    return _processor_singleton
