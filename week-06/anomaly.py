"""
Top-level Section 13 bridge module for Anomaly Detection & Financial Verification.
Provides direct access to AnomalyDetector and anomaly constants.
"""

from services.anomaly import (
    AnomalyDetector,
    ANOMALY_ARITHMETIC_MISMATCH,
    ANOMALY_SUSPICIOUS_AMOUNT,
    ANOMALY_FUTURE_DATE,
    ANOMALY_LOW_TEXT_QUALITY,
    ANOMALY_DUPLICATE_CONTENT,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    get_anomaly_detector
)

__all__ = [
    "AnomalyDetector",
    "get_anomaly_detector",
    "ANOMALY_ARITHMETIC_MISMATCH",
    "ANOMALY_SUSPICIOUS_AMOUNT",
    "ANOMALY_FUTURE_DATE",
    "ANOMALY_LOW_TEXT_QUALITY",
    "ANOMALY_DUPLICATE_CONTENT",
    "SEVERITY_HIGH",
    "SEVERITY_MEDIUM",
    "SEVERITY_LOW",
]
