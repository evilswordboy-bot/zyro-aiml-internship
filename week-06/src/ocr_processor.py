"""
OCR Processor Module
Provides OCR text extraction with explainable image preprocessing.
Supports both direct image uploads and scanned PDF page rasterization.
"""

import os
import io
import shutil
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import pymupdf


class OCRProcessor:
    """Manages OCR text extraction and image preprocessing pipeline."""

    def __init__(self):
        self.tesseract_available = self._configure_tesseract()

    def _configure_tesseract(self) -> bool:
        """Discovers and configures the Tesseract executable path."""
        if shutil.which("tesseract"):
            return True

        # Common Windows paths
        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for p in windows_paths:
            if os.path.isfile(p):
                pytesseract.pytesseract.tesseract_cmd = p
                return True
        return False

    def preprocess_image(
        self,
        image: Image.Image,
        apply_grayscale: bool = True,
        apply_threshold: bool = True,
        apply_denoise: bool = True
    ) -> Tuple[Image.Image, list]:
        """
        Applies beginner-friendly, explainable computer vision preprocessing:
        1. Grayscale conversion: reduces 3-channel color to luminance.
        2. Basic noise reduction: gentle median/blur filter to remove speckles.
        3. Thresholding: boosts contrast between ink and paper background.
        """
        processed = image.copy()
        steps = []

        # Convert to RGB first if RGBA/Palette
        if processed.mode not in ("RGB", "L"):
            processed = processed.convert("RGB")

        # Step 1: Grayscale
        if apply_grayscale and processed.mode != "L":
            processed = ImageOps.grayscale(processed)
            steps.append("Grayscale conversion")

        # Step 2: Denoise
        if apply_denoise:
            processed = processed.filter(ImageFilter.MedianFilter(size=3))
            steps.append("Median filter noise reduction")

        # Step 3: Thresholding / Contrast Enhancement
        if apply_threshold:
            # Simple automatic thresholding around mid-gray level
            threshold = 150
            processed = processed.point(lambda p: 255 if p > threshold else 0)
            steps.append("Binary contrast thresholding (threshold=150)")

        return processed, steps

    def extract_from_image(
        self,
        image_bytes: bytes,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """Performs OCR on an image byte buffer."""
        if not self.tesseract_available:
            return {
                "text": "",
                "preprocessing_steps": [],
                "ocr_used": True,
                "ocr_available": False,
                "error": "Tesseract OCR engine is not installed or configured on the host."
            }

        try:
            img = Image.open(io.BytesIO(image_bytes))
            steps = []
            if preprocess:
                img, steps = self.preprocess_image(img)

            raw_ocr = pytesseract.image_to_string(img)
            text = raw_ocr.strip()

            return {
                "text": text,
                "preprocessing_steps": steps,
                "ocr_used": True,
                "ocr_available": True,
                "error": None if text else "OCR ran successfully, but no legible text was detected."
            }
        except Exception as e:
            return {
                "text": "",
                "preprocessing_steps": [],
                "ocr_used": True,
                "ocr_available": True,
                "error": f"Image OCR failed: {str(e)}"
            }

    def ocr_pdf(
        self,
        pdf_bytes: bytes,
        max_pages: int = 5,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """Renders scanned PDF pages to images and executes OCR."""
        if not self.tesseract_available:
            return {
                "text": "",
                "preprocessing_steps": [],
                "ocr_used": True,
                "ocr_available": False,
                "error": "Tesseract OCR engine is not installed or configured on the host."
            }

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            extracted_pages = []
            all_steps = []

            for page_num in range(min(doc.page_count, max_pages)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                if preprocess:
                    img, steps = self.preprocess_image(img)
                    all_steps = steps

                page_text = pytesseract.image_to_string(img)
                if page_text.strip():
                    extracted_pages.append(page_text.strip())

            doc.close()
            combined_text = "\n\n".join(extracted_pages).strip()

            return {
                "text": combined_text,
                "preprocessing_steps": all_steps,
                "ocr_used": True,
                "ocr_available": True,
                "error": None if combined_text else "No legible text found via OCR on this PDF."
            }
        except Exception as e:
            return {
                "text": "",
                "preprocessing_steps": [],
                "ocr_used": True,
                "ocr_available": True,
                "error": f"Scanned PDF OCR failed: {str(e)}"
            }
