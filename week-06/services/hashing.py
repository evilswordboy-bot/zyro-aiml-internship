"""
Cryptographic Hashing Service.
Computes SHA-256 digests from file byte buffers for tamper-evident duplicate detection.
"""

import hashlib


def calculate_sha256(file_bytes: bytes) -> str:
    """
    Computes a deterministic SHA-256 hexadecimal digest from raw file bytes.
    """
    if not isinstance(file_bytes, (bytes, bytearray)):
        raise TypeError("Expected bytes or bytearray input for SHA-256 hashing.")
    return hashlib.sha256(file_bytes).hexdigest()
