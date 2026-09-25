"""
File Validation Service.
Enforces file format whitelisting, upload size thresholds, and integrity checks.
"""

import os
from typing import Dict, Any, Tuple

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 Megabytes


class FileValidator:
    """Validates uploaded files before resource-intensive parsing or storage."""

    @staticmethod
    def validate_file(filename: str, file_bytes: bytes) -> Tuple[bool, str, str]:
        """
        Validates file extension, length, and content size.

        Returns:
            Tuple[is_valid, sanitized_extension, error_message]
        """
        if not filename or not filename.strip():
            return False, "", "File upload error: Missing filename."

        _, ext = os.path.splitext(filename)
        ext_clean = ext.lower()

        if ext_clean not in ALLOWED_EXTENSIONS:
            allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
            return False, ext_clean, f"Unsupported file format '{ext}'. The platform accepts: {allowed_list}."

        if not file_bytes or len(file_bytes) == 0:
            return False, ext_clean, "The uploaded file is empty (0 bytes). Please upload a valid document."

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            file_mb = round(len(file_bytes) / (1024 * 1024), 2)
            return False, ext_clean, f"File exceeds maximum upload size limit ({file_mb} MB > {max_mb} MB limit)."

        return True, ext_clean.lstrip("."), ""
