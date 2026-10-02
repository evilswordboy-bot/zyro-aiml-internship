"""
Batch Processing Service for AI Document Intelligence & Workflow Platform.
Provides multi-document parallel intake with robust per-document fault isolation.
A failure or corrupt file in one document never halts or crashes the batch workflow.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
from services.document_processor import DocumentProcessor, get_document_processor


class BatchProcessor:
    """Manages batch ingestion and pipeline execution with isolated error boundaries."""

    def __init__(self, processor: Optional[DocumentProcessor] = None):
        self.processor = processor or get_document_processor()

    def process_batch(
        self,
        files: List[Tuple[str, bytes]],
        active_model_name: str = "Logistic Regression",
        ocr_settings: Optional[Dict[str, bool]] = None,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes workflow processing across multiple documents with strict fault tolerance.

        Args:
            files: List of (filename, file_bytes) tuples.
            active_model_name: ML classifier model to use.
            ocr_settings: OCR options.
            progress_callback: Optional hook called as (index, total, filename, status).

        Returns:
            Dict containing total counts, status breakdowns, and itemized results.
        """
        total = len(files)
        completed_count = 0
        needs_review_count = 0
        failed_count = 0
        duplicate_count = 0
        results: List[Dict[str, Any]] = []

        for idx, (filename, file_bytes) in enumerate(files, start=1):
            try:
                if progress_callback:
                    progress_callback(idx, total, filename, "Processing")

                res = self.processor.process_and_store(
                    file_bytes=file_bytes,
                    original_filename=filename,
                    active_model_name=active_model_name,
                    ocr_settings=ocr_settings
                )

                status = res.get("status") or (res.get("document", {}).get("status") if res.get("document") else "Failed")

                if res.get("is_duplicate"):
                    duplicate_count += 1
                elif status in ("Completed", "Processed"):
                    completed_count += 1
                elif status == "Needs Review":
                    needs_review_count += 1
                else:
                    failed_count += 1

                res["filename"] = filename
                results.append(res)

                if progress_callback:
                    progress_callback(idx, total, filename, status)

            except Exception as exc:
                # ISOLATED ERROR BOUNDARY: Catch any runtime exception for this file
                failed_count += 1
                error_item = {
                    "success": False,
                    "filename": filename,
                    "original_filename": filename,
                    "is_duplicate": False,
                    "status": "Failed",
                    "error": f"Batch item processing error: {str(exc)}",
                    "stages": ["Failed during batch execution"],
                    "document": None
                }
                results.append(error_item)

                if progress_callback:
                    progress_callback(idx, total, filename, "Failed")

        return {
            "total": total,
            "completed": completed_count,
            "needs_review": needs_review_count,
            "failed": failed_count,
            "duplicates": duplicate_count,
            "results": results
        }


_batch_singleton: Optional[BatchProcessor] = None


def get_batch_processor(processor: Optional[DocumentProcessor] = None) -> BatchProcessor:
    """Returns singleton instance of BatchProcessor."""
    global _batch_singleton
    if _batch_singleton is None:
        _batch_singleton = BatchProcessor(processor)
    return _batch_singleton
