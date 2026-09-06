"""
Text and metadata extraction engine for AI Document Intelligence Platform.
Supports native digital PDF text extraction via PyMuPDF (fitz) and OCR fallback
via pytesseract / EasyOCR for scanned documents and image files.
"""

import io
import os
import shutil
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Represents the output of the document text extraction pipeline."""
    text: str
    page_count: int
    char_count: int
    used_ocr: bool
    ocr_engine: Optional[str]
    extraction_status: str
    warnings: List[str] = field(default_factory=list)
    page_texts: List[str] = field(default_factory=list)


def _check_tesseract_available() -> Optional[str]:
    """
    Checks if Tesseract OCR binary is installed and reachable.
    Returns path or binary name if available, else None.
    """
    # Check PATH
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd

    # Common Windows installation locations
    common_win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for path in common_win_paths:
        if os.path.exists(path):
            return path
    return None


class DocumentExtractor:
    """Handles text extraction from PDFs and image files with automatic OCR fallback."""

    def __init__(self):
        self.tesseract_cmd = _check_tesseract_available()
        if self.tesseract_cmd:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                self.has_pytesseract = True
            except Exception:
                self.has_pytesseract = False
        else:
            self.has_pytesseract = False

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> ExtractionResult:
        """
        Extract text from raw file bytes based on file type.

        Args:
            file_bytes: Raw bytes of the document.
            filename: Name of the file for extension checking.

        Returns:
            ExtractionResult containing extracted text and metadata.
        """
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return self._extract_pdf(file_bytes)
        elif ext in {".png", ".jpg", ".jpeg"}:
            return self._extract_image(file_bytes)
        else:
            return ExtractionResult(
                text="",
                page_count=0,
                char_count=0,
                used_ocr=False,
                ocr_engine=None,
                extraction_status="Unsupported File Format",
                warnings=[f"Format '{ext}' is not supported for text extraction."]
            )

    def _extract_pdf(self, file_bytes: bytes) -> ExtractionResult:
        """Extracts text from PDF bytes using PyMuPDF, falling back to OCR if scanned."""
        warnings = []
        page_texts = []
        total_text = []

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            page_count = len(doc)

            for page_idx in range(page_count):
                page = doc[page_idx]
                page_str = page.get_text()
                page_texts.append(page_str)
                if page_str.strip():
                    total_text.append(page_str.strip())

            combined_text = "\n\n".join(total_text).strip()
            char_count = len(combined_text)

            # Check if PDF appears to be a scanned document (little to no selectable text)
            if char_count < 25 and page_count > 0:
                logger.info("PDF has very low selectable text. Attempting OCR on rendered pages...")
                ocr_text, ocr_engine, ocr_warn = self._ocr_pdf_pages(doc)
                if ocr_text.strip():
                    return ExtractionResult(
                        text=ocr_text,
                        page_count=page_count,
                        char_count=len(ocr_text),
                        used_ocr=True,
                        ocr_engine=ocr_engine,
                        extraction_status="Scanned PDF — OCR Successful",
                        warnings=ocr_warn,
                        page_texts=[ocr_text]
                    )
                else:
                    warnings.append(
                        "PDF contains minimal selectable text and OCR could not detect readable characters."
                    )
                    if ocr_warn:
                        warnings.extend(ocr_warn)

            return ExtractionResult(
                text=combined_text,
                page_count=page_count,
                char_count=char_count,
                used_ocr=False,
                ocr_engine=None,
                extraction_status="Direct PDF Text Extraction Successful",
                warnings=warnings,
                page_texts=page_texts
            )

        except Exception as e:
            logger.exception("Error extracting text from PDF")
            return ExtractionResult(
                text="",
                page_count=0,
                char_count=0,
                used_ocr=False,
                ocr_engine=None,
                extraction_status=f"Extraction Error: {str(e)}",
                warnings=[f"Failed to parse PDF: {str(e)}"]
            )

    def _ocr_pdf_pages(self, doc: pymupdf.Document) -> tuple[str, Optional[str], List[str]]:
        """Renders PDF pages to pixmaps and applies OCR."""
        warnings = []
        if not self.has_pytesseract:
            return (
                "",
                None,
                ["Tesseract OCR engine is not installed on the host. Please install Tesseract-OCR for scanned document support."]
            )

        import pytesseract

        extracted_pages = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                text = pytesseract.image_to_string(img)
                if text.strip():
                    extracted_pages.append(text.strip())
            except Exception as e:
                warnings.append(f"OCR failed on page {i+1}: {str(e)}")

        combined = "\n\n".join(extracted_pages).strip()
        return combined, "Tesseract OCR", warnings

    def _extract_image(self, file_bytes: bytes) -> ExtractionResult:
        """Extracts text from an image file using OCR."""
        warnings = []

        try:
            img = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            return ExtractionResult(
                text="",
                page_count=0,
                char_count=0,
                used_ocr=False,
                ocr_engine=None,
                extraction_status="Corrupted Image File",
                warnings=[f"Failed to open image: {str(e)}"]
            )

        if not self.has_pytesseract:
            # Informative fallback when Tesseract OCR binary is missing
            return ExtractionResult(
                text="",
                page_count=1,
                char_count=0,
                used_ocr=True,
                ocr_engine="Unavailable",
                extraction_status="OCR Engine Not Installed",
                warnings=[
                    "Image file uploaded successfully, but Tesseract OCR is not installed or configured on this machine.",
                    "Install Tesseract OCR (or set tesseract_cmd) to extract text from images and scanned files."
                ],
                page_texts=[]
            )

        import pytesseract
        try:
            text = pytesseract.image_to_string(img)
            clean_text = text.strip()
            return ExtractionResult(
                text=clean_text,
                page_count=1,
                char_count=len(clean_text),
                used_ocr=True,
                ocr_engine="Tesseract OCR",
                extraction_status="Image OCR Successful",
                warnings=warnings,
                page_texts=[clean_text]
            )
        except Exception as e:
            return ExtractionResult(
                text="",
                page_count=1,
                char_count=0,
                used_ocr=True,
                ocr_engine="Tesseract OCR",
                extraction_status="OCR Execution Failed",
                warnings=[f"Tesseract OCR encountered an error: {str(e)}"],
                page_texts=[]
            )
