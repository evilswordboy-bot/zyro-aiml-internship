"""
Unit tests for File Storage and Hashing services.
"""

import os
import hashlib
import tempfile
import pytest
from services.hashing import calculate_sha256
from services.file_storage import FileStorageManager
from services.validation import FileValidator


def test_sha256_calculation():
    data = b"Hello, AI Document Intelligence!"
    expected_hash = hashlib.sha256(data).hexdigest()
    computed = calculate_sha256(data)
    assert computed == expected_hash

    # Different data produces different hash
    assert calculate_sha256(b"Other content") != expected_hash


def test_file_storage_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorageManager(storage_dir=tmpdir)

        # Check subdirectories
        assert os.path.isdir(os.path.join(tmpdir, "invoices"))
        assert os.path.isdir(os.path.join(tmpdir, "resumes"))
        assert os.path.isdir(os.path.join(tmpdir, "other"))

        content = b"PDF invoice content dummy stream"
        file_hash = calculate_sha256(content)

        stored_name, rel_path = storage.save_file(
            file_bytes=content,
            original_filename="Client Bill #44.pdf",
            doc_type="Invoice",
            file_hash=file_hash
        )

        assert stored_name.startswith("invoice_")
        assert stored_name.endswith(".pdf")
        assert rel_path.startswith("invoices/")

        # Verify reading file bytes back
        read_back = storage.read_file(rel_path)
        assert read_back == content

        # Verify deletion
        deleted = storage.delete_file(rel_path)
        assert deleted is True
        assert storage.read_file(rel_path) is None


def test_path_traversal_prevention():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorageManager(storage_dir=tmpdir)

        # Attempting path traversal with ../ should raise PermissionError
        with pytest.raises(PermissionError):
            storage.get_full_path("../../windows/system32/cmd.exe")

        with pytest.raises(PermissionError):
            storage.get_full_path("../../../secrets.txt")

        # read_file should safely return None when path traversal is attempted
        assert storage.read_file("../../../secrets.txt") is None


def test_file_validation():
    # Valid PDF
    valid, ext, err = FileValidator.validate_file("document.pdf", b"%PDF-1.4 dummy")
    assert valid is True
    assert ext == "pdf"
    assert err == ""

    # Invalid extension
    valid, ext, err = FileValidator.validate_file("exploit.exe", b"binary")
    assert valid is False
    assert "Unsupported file format" in err

    # 0-byte file
    valid, ext, err = FileValidator.validate_file("empty.pdf", b"")
    assert valid is False
    assert "empty" in err

    # Too large file
    oversized = b"0" * (21 * 1024 * 1024)
    valid, ext, err = FileValidator.validate_file("huge.pdf", oversized)
    assert valid is False
    assert "exceeds maximum upload size" in err
