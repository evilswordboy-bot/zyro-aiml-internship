"""
Input validation module for Zyroo AI Document Intelligence Platform.
Enforces security constraints, allowed file extensions, size limits,
and structural integrity checks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image
import io
from .config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES


@dataclass
class ValidationResult:
    """Represents the outcome of a document validation check."""
    is_valid: bool
    error_message: Optional[str]
    file_type: str
    file_size_kb: float
    filename: str


def validate_document(file_bytes: bytes, filename: str) -> ValidationResult:
    """
    Validates uploaded document against size limits, allowed extensions,
    and byte contents.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original name of the uploaded file.

    Returns:
        ValidationResult with validation state and metadata.
    """
    if not filename or not filename.strip():
        return ValidationResult(
            is_valid=False,
            error_message="Filename cannot be empty.",
            file_type="Unknown",
            file_size_kb=0.0,
            filename="unnamed"
        )

    clean_filename = Path(filename).name
    ext = Path(clean_filename).suffix.lower()
    file_size = len(file_bytes)
    file_size_kb = round(file_size / 1024.0, 2)

    # Check 1: File must not be empty
    if file_size == 0:
        return ValidationResult(
            is_valid=False,
            error_message="The uploaded file is empty (0 bytes). Please upload a valid document.",
            file_type="Empty",
            file_size_kb=0.0,
            filename=clean_filename
        )

    # Check 2: Extension check
    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return ValidationResult(
            is_valid=False,
            error_message=f"Unsupported file format '{ext}'. Allowed formats: {allowed_str}.",
            file_type=ext.replace(".", "").upper() if ext else "None",
            file_size_kb=file_size_kb,
            filename=clean_filename
        )

    # Check 3: Size check
    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return ValidationResult(
            is_valid=False,
            error_message=f"File exceeds maximum allowed size of {max_mb:.0f} MB (Current: {file_size_kb / 1024:.2f} MB).",
            file_type=ext.replace(".", "").upper(),
            file_size_kb=file_size_kb,
            filename=clean_filename
        )

    # Check 4: Magic byte verification
    file_type = "PDF" if ext == ".pdf" else "IMAGE"
    if ext == ".pdf":
        if not file_bytes.startswith(b"%PDF"):
            return ValidationResult(
                is_valid=False,
                error_message="File header indicates this is not a valid PDF document.",
                file_type="Invalid PDF",
                file_size_kb=file_size_kb,
                filename=clean_filename
            )
    elif ext in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img.verify()
        except Exception:
            return ValidationResult(
                is_valid=False,
                error_message="Uploaded image file is corrupted or not a recognized image format.",
                file_type=f"Corrupted {ext.replace('.', '').upper()}",
                file_size_kb=file_size_kb,
                filename=clean_filename
            )

    return ValidationResult(
        is_valid=True,
        error_message=None,
        file_type=file_type,
        file_size_kb=file_size_kb,
        filename=clean_filename
    )
