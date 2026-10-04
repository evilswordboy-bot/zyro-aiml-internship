"""
Anomaly Detection & Financial Consistency Verification Engine.
Week 6: Advanced Verification, Arithmetic Checks & Pattern Analysis.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from services.validation import DocumentValidator


ANOMALY_ARITHMETIC_MISMATCH = "ARITHMETIC_MISMATCH"
ANOMALY_SUSPICIOUS_AMOUNT = "SUSPICIOUS_AMOUNT"
ANOMALY_FUTURE_DATE = "FUTURE_DATE"
ANOMALY_LOW_TEXT_QUALITY = "LOW_TEXT_QUALITY"
ANOMALY_DUPLICATE_CONTENT = "DUPLICATE_CONTENT"

SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"


class AnomalyDetector:
    """
    Analyzes document fields, financial values, dates, and textual integrity
    to flag arithmetic discrepancies, suspicious values, and formatting outliers.
    """

    @classmethod
    def detect_anomalies(
        cls,
        doc_type: str,
        fields: Dict[str, Any],
        text: str,
        file_hash: Optional[str] = None,
        db_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes comprehensive anomaly analysis across extracted entities and raw text.
        Returns dictionary containing has_anomalies (bool), anomalies list, and summary.
        """
        anomalies: List[Dict[str, Any]] = []

        # 1. Financial Arithmetic Consistency (Invoices)
        if doc_type == "Invoice":
            arithmetic_anomaly = cls._check_invoice_arithmetic(fields, text)
            if arithmetic_anomaly:
                anomalies.append(arithmetic_anomaly)

            amount_anomaly = cls._check_suspicious_amount(fields.get("Total Amount"))
            if amount_anomaly:
                anomalies.append(amount_anomaly)

            date_anomaly = cls._check_future_date(text)
            if date_anomaly:
                anomalies.append(date_anomaly)

        # 2. Text Quality & Garbage OCR Artifacts
        text_anomaly = cls._check_text_quality(text)
        if text_anomaly:
            anomalies.append(text_anomaly)

        # 3. Duplicate Document / Hash Collision Check
        if file_hash and db_manager:
            dup_anomaly = cls._check_duplicate_hash(file_hash, db_manager)
            if dup_anomaly:
                anomalies.append(dup_anomaly)

        has_high_severity = any(a["severity"] == SEVERITY_HIGH for a in anomalies)
        return {
            "has_anomalies": len(anomalies) > 0,
            "has_high_severity": has_high_severity,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "summary": f"Identified {len(anomalies)} anomaly/consistency flag(s)." if anomalies else "Clean — No anomalies detected."
        }

    @classmethod
    def _check_invoice_arithmetic(cls, fields: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
        """
        Checks whether Subtotal + Tax = Total Amount within tolerance.
        """
        total_str = fields.get("Total Amount")
        if not total_str or total_str == "Not Found":
            return None

        total_val = DocumentValidator.parse_currency_amount(str(total_str))
        if total_val is None:
            return None

        # Look for Subtotal in text
        subtotal_match = re.search(
            r"(?:subtotal|sub-total|sub\s+total|net\s+amount)[:\s]*([$€£₹Rs.]*\s*[\d,]+(?:\.\d{2})?)",
            text,
            re.IGNORECASE
        )
        # Look for Tax / VAT / GST in text
        tax_match = re.search(
            r"(?:tax|vat|gst|sales\s+tax)[:\s]*([$€£₹Rs.]*\s*[\d,]+(?:\.\d{2})?)",
            text,
            re.IGNORECASE
        )

        if subtotal_match and tax_match:
            subtotal_val = DocumentValidator.parse_currency_amount(subtotal_match.group(1))
            tax_val = DocumentValidator.parse_currency_amount(tax_match.group(1))

            if subtotal_val is not None and tax_val is not None:
                expected_total = round(subtotal_val + tax_val, 2)
                diff = abs(expected_total - total_val)

                if diff > 0.05:
                    return {
                        "type": ANOMALY_ARITHMETIC_MISMATCH,
                        "field": "Total Amount",
                        "severity": SEVERITY_HIGH,
                        "message": (
                            f"Arithmetic mismatch: Subtotal ({subtotal_val:.2f}) + Tax ({tax_val:.2f}) "
                            f"= {expected_total:.2f}, but Total Amount states {total_val:.2f} (diff: {diff:.2f})."
                        ),
                        "details": {
                            "subtotal": subtotal_val,
                            "tax": tax_val,
                            "expected_total": expected_total,
                            "actual_total": total_val,
                            "difference": round(diff, 2)
                        }
                    }

        return None

    @classmethod
    def _check_suspicious_amount(cls, total_str: Any) -> Optional[Dict[str, Any]]:
        """Flags negative, zero, or extreme outlier amounts."""
        if not total_str or total_str == "Not Found":
            return None

        val = DocumentValidator.parse_currency_amount(str(total_str))
        if val is None:
            return None

        if val <= 0:
            return {
                "type": ANOMALY_SUSPICIOUS_AMOUNT,
                "field": "Total Amount",
                "severity": SEVERITY_HIGH,
                "message": f"Suspicious invoice amount: {val:.2f} (must be strictly positive).",
                "details": {"amount": val}
            }

        if val > 1_000_000.0:
            return {
                "type": ANOMALY_SUSPICIOUS_AMOUNT,
                "field": "Total Amount",
                "severity": SEVERITY_MEDIUM,
                "message": f"High value transaction warning: Amount exceeds $1,000,000.00 ({val:,.2f}).",
                "details": {"amount": val}
            }

        return None

    @classmethod
    def _check_future_date(cls, text: str) -> Optional[Dict[str, Any]]:
        """Flags dates occurring > 365 days into the future."""
        date_matches = re.findall(r"\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text)
        now = datetime.now()
        max_future = now + timedelta(days=365)

        for year_s, month_s, day_s in date_matches:
            try:
                dt = datetime(int(year_s), int(month_s), int(day_s))
                if dt > max_future:
                    return {
                        "type": ANOMALY_FUTURE_DATE,
                        "field": "Date",
                        "severity": SEVERITY_MEDIUM,
                        "message": f"Document contains a date far in the future ({dt.strftime('%Y-%m-%d')}).",
                        "details": {"date": dt.strftime('%Y-%m-%d')}
                    }
            except ValueError:
                continue

        return None

    @classmethod
    def _check_text_quality(cls, text: str) -> Optional[Dict[str, Any]]:
        """Flags garbled, repetitive, or unreadable OCR output."""
        if not text:
            return None

        # Check for extreme character repetition (e.g. "aaaaaaaaaaaa" or "............")
        if re.search(r"(.)\1{14,}", text):
            return {
                "type": ANOMALY_LOW_TEXT_QUALITY,
                "field": "Text Integrity",
                "severity": SEVERITY_HIGH,
                "message": "Detected severe character repetition indicative of corrupted OCR or noise.",
            }

        # Check non-alphanumeric noise ratio
        alpha_count = sum(1 for c in text if c.isalnum() or c.isspace())
        total_count = len(text)
        if total_count > 100:
            noise_ratio = (total_count - alpha_count) / total_count
            if noise_ratio > 0.40:
                return {
                    "type": ANOMALY_LOW_TEXT_QUALITY,
                    "field": "Text Integrity",
                    "severity": SEVERITY_MEDIUM,
                    "message": f"High symbol noise ratio ({noise_ratio:.1%}) in extracted text.",
                    "details": {"noise_ratio": round(noise_ratio, 3)}
                }

        return None

    @classmethod
    def _check_duplicate_hash(cls, file_hash: str, db_manager: Any) -> Optional[Dict[str, Any]]:
        """Flags if another document already exists with this hash."""
        existing = db_manager.get_document_by_hash(file_hash)
        if existing:
            return {
                "type": ANOMALY_DUPLICATE_CONTENT,
                "field": "File Hash",
                "severity": SEVERITY_MEDIUM,
                "message": f"Duplicate file detected: Exactly matches Document #{existing['id']} ('{existing['original_filename']}').",
                "details": {"existing_id": existing["id"], "existing_filename": existing["original_filename"]}
            }
        return None


def get_anomaly_detector() -> AnomalyDetector:
    """Returns AnomalyDetector singleton/instance."""
    return AnomalyDetector()
