"""
Top-level storage interface module.
Exposes FileStorageManager and get_storage_manager from the services package.
"""

from services.file_storage import FileStorageManager, get_storage_manager

__all__ = ["FileStorageManager", "get_storage_manager"]
