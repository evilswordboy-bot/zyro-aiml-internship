"""
Classifier Service wrapper bridging to the modular core classification suite.
"""

from src.classifier import (
    DocumentClassifier,
    RuleBasedClassifier,
    CLASS_INVOICE,
    CLASS_RESUME,
    CLASS_OTHER,
    TARGET_NAMES
)

__all__ = [
    "DocumentClassifier",
    "RuleBasedClassifier",
    "CLASS_INVOICE",
    "CLASS_RESUME",
    "CLASS_OTHER",
    "TARGET_NAMES"
]
