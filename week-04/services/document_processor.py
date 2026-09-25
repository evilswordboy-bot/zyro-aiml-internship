"""
Document Processor Service.
Orchestrates the complete Week 4 Document Intelligence Workflow:
Validation -> SHA-256 Hashing -> Duplicate Detection -> Text Extraction / OCR
-> Text Cleaning -> ML Classification -> Entity Extraction -> Status Assignment
-> Physical Storage -> SQLite Persistence.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from database.db import DatabaseManager, get_db
from models.document import (
    DocumentRecord,
    STATUS_PROCESSED,
    STATUS_NEEDS_REVIEW,
    STATUS_FAILED,
    TYPE_INVOICE,
    TYPE_RESUME,
    TYPE_OTHER
)
from services.hashing import calculate_sha256
from services.file_storage import FileStorageManager, get_storage_manager
from services.validation import FileValidator
from src.document_reader import DocumentReader
from src.ocr_processor import OCRProcessor
from src.text_cleaner import TextCleaner
from src.classifier import DocumentClassifier
from src.extractor import DocumentExtractor, NOT_FOUND


class DocumentProcessor:
    """End-to-end controller for document intake, intelligence pipeline, and persistence."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        storage_manager: Optional[FileStorageManager] = None,
        classifier: Optional[DocumentClassifier] = None,
        ocr_engine: Optional[OCRProcessor] = None
    ):
        self.db = db_manager or get_db()
        self.storage = storage_manager or get_storage_manager()
        self.classifier = classifier or DocumentClassifier()
        self.ocr_engine = ocr_engine or OCRProcessor()

    def process_and_store(
        self,
        file_bytes: bytes,
        original_filename: str,
        active_model_name: str = "Logistic Regression",
        ocr_settings: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline. Detects duplicates before parsing.
        Persists newly parsed documents to storage and database.
        """
        ocr_opts = ocr_settings or {"preprocess": True}
        stages = []

        # -------------------------------------------------------------
        # 1. Validation
        # -------------------------------------------------------------
        is_valid, ext_clean, val_error = FileValidator.validate_file(original_filename, file_bytes)
        if not is_valid:
            return {
                "success": False,
                "is_duplicate": False,
                "error": val_error,
                "stages": ["Validation Failed"]
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
                # Record as failed document
                return self._store_failed_document(
                    file_bytes, original_filename, file_hash, "Unreadable",
                    read_res["error"], stages
                )

            if read_res["needs_ocr"]:
                ocr_res = self.ocr_engine.ocr_pdf(file_bytes, preprocess=ocr_opts.get("preprocess", True))
                raw_text = ocr_res["text"]
                ocr_used = True
                ocr_steps = ocr_res.get("preprocessing_steps", [])
                extraction_note = "Scanned PDF detected — executed OCR fallback."
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
        confidence = pred_res["confidence"]
        stages.append("Document Classified")

        # -------------------------------------------------------------
        # 7. Entity Extraction
        # -------------------------------------------------------------
        extract_res = DocumentExtractor.extract(cleaned_text, doc_type)
        fields = extract_res.get("fields", {})
        missing_fields = extract_res.get("missing_fields", [])
        stages.append("Fields Extracted")

        # -------------------------------------------------------------
        # 8. Status Determination
        # -------------------------------------------------------------
        status, status_reason = self._determine_status(doc_type, fields, missing_fields, pred_res)
        stages.append("Status Evaluated")

        # -------------------------------------------------------------
        # 9. Structured Physical Storage
        # -------------------------------------------------------------
        stored_filename, relative_path = self.storage.save_file(
            file_bytes=file_bytes,
            original_filename=original_filename,
            doc_type=doc_type,
            file_hash=file_hash
        )
        stages.append("File Stored")

        # -------------------------------------------------------------
        # 10. Database Persistence
        # -------------------------------------------------------------
        # Map extracted entities to database columns
        company = fields.get("Company Name", NOT_FOUND)
        invoice_number = fields.get("Invoice Number", NOT_FOUND)
        total_amount = fields.get("Total Amount", NOT_FOUND)

        candidate_name = fields.get("Name", NOT_FOUND)
        candidate_email = fields.get("Email", NOT_FOUND)
        candidate_phone = fields.get("Phone", NOT_FOUND)
        candidate_skills = fields.get("Skills", NOT_FOUND)

        metadata_dict = {
            "classification_model": active_model_name,
            "confidence": confidence,
            "probabilities": pred_res.get("probabilities"),
            "ocr_used": ocr_used,
            "ocr_steps": ocr_steps,
            "extraction_note": extraction_note,
            "missing_fields": missing_fields,
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
            "status": status,
            "status_reason": status_reason,
            "metadata_json": metadata_dict
        }

        doc_id = self.db.add_document(doc_record)
        stages.append("Metadata Saved to SQLite")

        saved_doc = self.db.get_document_by_id(doc_id)

        return {
            "success": True,
            "is_duplicate": False,
            "document_id": doc_id,
            "document": saved_doc,
            "file_hash": file_hash,
            "document_type": doc_type,
            "confidence": confidence,
            "status": status,
            "status_reason": status_reason,
            "fields": fields,
            "missing_fields": missing_fields,
            "original_text": raw_text,
            "cleaned_text": cleaned_text,
            "stored_filename": stored_filename,
            "file_path": relative_path,
            "stages": stages,
            "ocr_used": ocr_used,
            "ocr_steps": ocr_steps,
            "error": None
        }

    def _determine_status(
        self,
        doc_type: str,
        fields: Dict[str, Any],
        missing_fields: list,
        pred_res: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Calculates honest processing status based on field completeness and confidence:
        - Processed: High confidence and all key fields present.
        - Needs Review: Document parsed, but critical fields are missing or require human review.
        - Failed: Unreadable or corrupt.
        """
        if doc_type == TYPE_INVOICE:
            critical_missing = []
            if fields.get("Invoice Number") == NOT_FOUND:
                critical_missing.append("Invoice Number")
            if fields.get("Total Amount") == NOT_FOUND:
                critical_missing.append("Total Amount")

            if critical_missing:
                return STATUS_NEEDS_REVIEW, f"Missing critical invoice field(s): {', '.join(critical_missing)}."
            return STATUS_PROCESSED, "All primary invoice fields identified successfully."

        elif doc_type == TYPE_RESUME:
            critical_missing = []
            if fields.get("Name") == NOT_FOUND:
                critical_missing.append("Candidate Name")
            if fields.get("Email") == NOT_FOUND:
                critical_missing.append("Email Address")
            if fields.get("Phone") == NOT_FOUND:
                critical_missing.append("Phone Number")

            if critical_missing:
                return STATUS_NEEDS_REVIEW, f"Missing resume field(s): {', '.join(critical_missing)}."
            return STATUS_PROCESSED, "Resume profile extracted cleanly."

        else:
            return STATUS_PROCESSED, "Document categorized as 'Other' and indexed for retrieval."

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
        """Safely saves unreadable or corrupt documents with 'Failed' status."""
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
            "status_reason": error_msg,
            "metadata_json": {
                "error": error_msg,
                "ocr_used": ocr_used,
                "stages": stages + ["Failed"]
            }
        }
        doc_id = self.db.add_document(doc_record)
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
    ocr_engine: Optional[OCRProcessor] = None
) -> DocumentProcessor:
    """Returns singleton instance of DocumentProcessor."""
    global _processor_singleton
    if _processor_singleton is None:
        _processor_singleton = DocumentProcessor(db_manager, storage_manager, classifier, ocr_engine)
    return _processor_singleton
