"""
Database package for AI Document Intelligence & Workflow Platform.
"""

from .db import (
    DatabaseManager,
    get_db,
    DEFAULT_DB_PATH
)

__all__ = ["DatabaseManager", "get_db", "DEFAULT_DB_PATH"]
