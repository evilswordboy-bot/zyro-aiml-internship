"""
Document Reader Module
Uses PyMuPDF to extract native selectable text from PDF documents.
Detects when text is insufficient and triggers OCR fallback.
"""

import re
from typing import Dict, Any
import pymupdf


class DocumentReader:
    """Handles native text extraction from digital PDF documents."""

    @staticmethod
    def read_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts selectable text from a PDF file.

        Returns:
            Dict containing:
                - text: Extracted combined text
                - page_count: Total pages
                - char_count: Total characters
                - needs_ocr: Boolean indicating if OCR fallback is needed
                - is_empty: Boolean indicating if document has no text
                - error: Error message if failed, else None
        """
        if not pdf_bytes or len(pdf_bytes) == 0:
            return {
                "text": "",
                "page_count": 0,
                "char_count": 0,
                "needs_ocr": True,
                "is_empty": True,
                "error": "The uploaded PDF document is empty."
            }

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            return {
                "text": "",
                "page_count": 0,
                "char_count": 0,
                "needs_ocr": False,
                "is_empty": True,
                "error": f"Failed to open PDF document: {str(e)}"
            }

        if doc.page_count == 0:
            doc.close()
            return {
                "text": "",
                "page_count": 0,
                "char_count": 0,
                "needs_ocr": True,
                "is_empty": True,
                "error": "The PDF document contains 0 pages."
            }

        total_pages = doc.page_count
        page_texts = []
        for page_idx in range(total_pages):
            try:
                page = doc.load_page(page_idx)
                text = page.get_text("text")
                if text and text.strip():
                    page_texts.append(text.strip())
            except Exception:
                continue

        doc.close()
        combined_text = "\n\n".join(page_texts).strip()
        alphanumeric_count = len(re.findall(r"\w", combined_text))

        # If selectable text is under 20 alphanumeric characters, OCR is needed
        needs_ocr = alphanumeric_count < 20

        return {
            "text": combined_text,
            "page_count": total_pages,
            "char_count": len(combined_text),
            "needs_ocr": needs_ocr,
            "is_empty": len(combined_text) == 0,
            "error": None
        }
