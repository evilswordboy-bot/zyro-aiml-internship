"""
Top-level database interface module.
Exposes DatabaseManager and get_db from the database package.
"""

from database.db import DatabaseManager, get_db

__all__ = ["DatabaseManager", "get_db"]
