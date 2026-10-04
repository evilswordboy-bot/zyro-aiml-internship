"""
Top-level validation interface module.
Exposes FileValidator and DocumentValidator from the services package.
"""

from services.validation import FileValidator, DocumentValidator

__all__ = ["FileValidator", "DocumentValidator"]
