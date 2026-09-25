"""
Structured File Storage Service.
Provides organized, secure file persistence across categorized directories:
- storage/invoices/
- storage/resumes/
- storage/other/
Includes path-traversal guardrails and collision-free unique filename generation.
"""

import os
import re
from datetime import datetime
from typing import Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STORAGE_DIR = os.path.join(BASE_DIR, "storage")


class FileStorageManager:
    """Manages physical file persistence, retrieval, and directory organization."""

    def __init__(self, storage_dir: str = DEFAULT_STORAGE_DIR):
        self.storage_base = os.path.abspath(storage_dir)
        self.categories = {
            "Invoice": "invoices",
            "Resume": "resumes",
            "Other": "other"
        }
        self.init_storage_dirs()

    def init_storage_dirs(self) -> None:
        """Ensures all categorized storage directories exist on disk."""
        os.makedirs(self.storage_base, exist_ok=True)
        for sub in self.categories.values():
            folder_path = os.path.join(self.storage_base, sub)
            os.makedirs(folder_path, exist_ok=True)

    def _get_category_folder(self, doc_type: str) -> str:
        """Resolves target subfolder based on document classification."""
        return self.categories.get(doc_type, "other")

    def _generate_safe_filename(
        self,
        original_filename: str,
        doc_type: str,
        file_hash: str
    ) -> str:
        """
        Generates a collision-resistant, sanitized filename:
        Format: <type>_<YYYYMMDD_HHMMSS>_<short_hash>.<ext>
        Example: invoice_20260925_193000_8f31a2.pdf
        """
        # Extract and sanitize extension
        _, ext = os.path.splitext(original_filename)
        safe_ext = re.sub(r"[^a-zA-Z0-9]", "", ext.lower())
        if not safe_ext:
            safe_ext = "bin"

        type_prefix = self._get_category_folder(doc_type).rstrip("s")  # invoice / resume / other
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = (file_hash[:8] if file_hash else "nohash").lower()

        return f"{type_prefix}_{timestamp}_{short_hash}.{safe_ext}"

    def _validate_safe_path(self, relative_path: str) -> str:
        """
        Guards against path traversal attacks (e.g. '../').
        Ensures the resolved absolute path resides strictly inside storage_base.
        """
        target = os.path.abspath(os.path.join(self.storage_base, relative_path))
        if not target.startswith(self.storage_base + os.sep) and target != self.storage_base:
            raise PermissionError(f"Security Alert: Path traversal attempt blocked for '{relative_path}'.")
        return target

    def save_file(
        self,
        file_bytes: bytes,
        original_filename: str,
        doc_type: str,
        file_hash: str
    ) -> Tuple[str, str]:
        """
        Saves file bytes into the corresponding category folder.

        Returns:
            Tuple[stored_filename, relative_path]
        """
        sub_folder = self._get_category_folder(doc_type)
        stored_filename = self._generate_safe_filename(original_filename, doc_type, file_hash)
        relative_path = os.path.join(sub_folder, stored_filename).replace("\\", "/")
        target_path = self._validate_safe_path(relative_path)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(file_bytes)

        return stored_filename, relative_path

    def get_full_path(self, relative_path: str) -> str:
        """Returns the safe absolute path for a relative storage path."""
        return self._validate_safe_path(relative_path)

    def read_file(self, relative_path: str) -> Optional[bytes]:
        """Safely reads file bytes from the storage repository."""
        try:
            full_path = self._validate_safe_path(relative_path)
            if os.path.isfile(full_path):
                with open(full_path, "rb") as f:
                    return f.read()
            return None
        except Exception:
            return None

    def delete_file(self, relative_path: str) -> bool:
        """Safely removes a stored physical file."""
        try:
            full_path = self._validate_safe_path(relative_path)
            if os.path.isfile(full_path):
                os.remove(full_path)
                return True
            return False
        except Exception:
            return False


_storage_singleton: Optional[FileStorageManager] = None


def get_storage_manager(storage_dir: str = DEFAULT_STORAGE_DIR) -> FileStorageManager:
    """Returns singleton instance of FileStorageManager."""
    global _storage_singleton
    if _storage_singleton is None or _storage_singleton.storage_base != os.path.abspath(storage_dir):
        _storage_singleton = FileStorageManager(storage_dir)
    return _storage_singleton
