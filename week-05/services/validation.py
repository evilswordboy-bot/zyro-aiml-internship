"""
Validation Service for AI Document Intelligence & Workflow Platform.
Provides:
1. FileValidator: Upload format, size, and integrity checks.
2. DocumentValidator: Advanced structural, format, and entity validation for Invoices, Resumes, and Other documents.
"""

import os
import re
from typing import Dict, Any, Tuple, List, Optional

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 Megabytes
NOT_FOUND = "Not Found"

# RFC 5322 compliant simplified email regex
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class FileValidator:
    """Validates uploaded files before resource-intensive parsing or storage."""

    @staticmethod
    def validate_file(filename: str, file_bytes: bytes) -> Tuple[bool, str, str]:
        """
        Validates file extension, length, and content size.

        Returns:
            Tuple[is_valid, sanitized_extension, error_message]
        """
        if not filename or not filename.strip():
            return False, "", "File upload error: Missing filename."

        _, ext = os.path.splitext(filename)
        ext_clean = ext.lower()

        if ext_clean not in ALLOWED_EXTENSIONS:
            allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
            return False, ext_clean, f"Unsupported file format '{ext}'. The platform accepts: {allowed_list}."

        if not file_bytes or len(file_bytes) == 0:
            return False, ext_clean, "The uploaded file is empty (0 bytes). Please upload a valid document."

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            file_mb = round(len(file_bytes) / (1024 * 1024), 2)
            return False, ext_clean, f"File exceeds maximum upload size limit ({file_mb} MB > {max_mb} MB limit)."

        return True, ext_clean.lstrip("."), ""


class DocumentValidator:
    """
    Advanced entity and structural validation engine.
    Applies rule-based integrity checks against extracted metadata fields.
    """

    @classmethod
    def parse_numeric_amount(cls, amount_str: Any) -> Optional[float]:
        """
        Extracts clean floating-point monetary value from currency strings like:
        '$ 1,450.00', 'Rs. 78,500', 'EUR 120.50', '2,500.00'.
        """
        if not amount_str or amount_str == NOT_FOUND:
            return None

        clean_str = str(amount_str).strip()
        # Remove leading currency prefixes (e.g. 'Rs. ', 'USD ', '$', '€', '£')
        clean_str = re.sub(r"^[^\d\-]+", "", clean_str)
        # Remove remaining characters that are not digits, comma, dot, or minus
        clean_str = re.sub(r"[^\d.,\-]", "", clean_str)

        if not clean_str:
            return None

        # Handle comma as thousands separator (e.g. 1,450.00) or decimal (1450,00)
        if "," in clean_str and "." in clean_str:
            # Comma before dot: 1,450.00
            if clean_str.find(",") < clean_str.find("."):
                clean_str = clean_str.replace(",", "")
            else:
                # European dot before comma: 1.450,00
                clean_str = clean_str.replace(".", "").replace(",", ".")
        elif "," in clean_str and "." not in clean_str:
            # e.g. 78,500 or 1500,50
            parts = clean_str.split(",")
            if len(parts[-1]) == 2:
                clean_str = clean_str.replace(",", ".")
            else:
                clean_str = clean_str.replace(",", "")

        try:
            val = float(clean_str)
            return val
        except ValueError:
            return None

    @classmethod
    def validate_invoice(cls, fields: Dict[str, Any], text: str = "") -> Dict[str, Any]:
        """
        Validates invoice entities:
        - Invoice Number: Present, non-empty, min 3 chars
        - Total Amount: Present, valid positive monetary value
        - Company Name: Present, non-empty, min 2 chars
        - Date: Checked if present for basic format
        """
        missing_fields: List[str] = []
        invalid_fields: List[Dict[str, str]] = []
        passed_fields: List[str] = []
        reasons: List[str] = []

        # 1. Invoice Number
        inv_num = fields.get("Invoice Number")
        if not inv_num or inv_num == NOT_FOUND or str(inv_num).strip() == "":
            missing_fields.append("Invoice Number")
            reasons.append("Invoice Number is missing.")
        else:
            clean_num = str(inv_num).strip()
            if len(clean_num) < 3:
                invalid_fields.append({"field": "Invoice Number", "reason": "Invoice number too short (< 3 characters)."})
                reasons.append(f"Invalid Invoice Number format: '{clean_num}'.")
            else:
                passed_fields.append("Invoice Number")

        # 2. Total Amount
        tot_amt = fields.get("Total Amount")
        if not tot_amt or tot_amt == NOT_FOUND or str(tot_amt).strip() == "":
            missing_fields.append("Total Amount")
            reasons.append("Total Amount is missing.")
        else:
            parsed_amt = cls.parse_numeric_amount(tot_amt)
            if parsed_amt is None or parsed_amt <= 0:
                invalid_fields.append({
                    "field": "Total Amount",
                    "reason": f"Cannot parse positive monetary figure from '{tot_amt}'."
                })
                reasons.append(f"Total Amount '{tot_amt}' is invalid or non-positive.")
            else:
                passed_fields.append("Total Amount")

        # 3. Company Name
        company = fields.get("Company Name") or fields.get("Company")
        if not company or company == NOT_FOUND or str(company).strip() == "":
            missing_fields.append("Company Name")
            reasons.append("Vendor / Company Name is missing.")
        else:
            clean_comp = str(company).strip()
            if len(clean_comp) < 2:
                invalid_fields.append({"field": "Company Name", "reason": "Company name is too short (< 2 chars)."})
                reasons.append(f"Invalid Company Name: '{clean_comp}'.")
            else:
                passed_fields.append("Company Name")

        # 4. Optional Date Check
        inv_date = fields.get("Date")
        if inv_date and inv_date != NOT_FOUND:
            date_str = str(inv_date).strip()
            date_match = re.search(r"(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})", date_str)
            if not date_match:
                invalid_fields.append({"field": "Date", "reason": f"Unrecognized date format in '{date_str}'."})
            else:
                passed_fields.append("Date")

        if missing_fields:
            reasons.insert(0, f"Missing critical invoice field(s): {', '.join(missing_fields)}.")

        is_valid = (len(missing_fields) == 0 and len(invalid_fields) == 0)

        return {
            "is_valid": is_valid,
            "doc_type": "Invoice",
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "passed_fields": passed_fields,
            "reasons": reasons
        }

    @classmethod
    def validate_resume(cls, fields: Dict[str, Any], text: str = "") -> Dict[str, Any]:
        """
        Validates resume entities:
        - Candidate Name: Present, length >= 2, no digits
        - Candidate Email: Valid RFC regex format
        - Candidate Phone: 10-15 digits
        - Skills: Present and non-empty
        """
        missing_fields: List[str] = []
        invalid_fields: List[Dict[str, str]] = []
        passed_fields: List[str] = []
        reasons: List[str] = []

        # 1. Candidate Name
        name = fields.get("Name") or fields.get("Candidate Name")
        if not name or name == NOT_FOUND or str(name).strip() == "":
            missing_fields.append("Candidate Name")
            reasons.append("Candidate Name is missing.")
        else:
            clean_name = str(name).strip()
            if len(clean_name) < 2:
                invalid_fields.append({"field": "Candidate Name", "reason": "Name is too short (< 2 characters)."})
                reasons.append("Candidate Name is too short.")
            elif any(c.isdigit() for c in clean_name):
                invalid_fields.append({"field": "Candidate Name", "reason": "Name contains numeric digits."})
                reasons.append(f"Candidate Name '{clean_name}' contains numbers.")
            else:
                passed_fields.append("Candidate Name")

        # 2. Email Address
        email = fields.get("Email") or fields.get("Candidate Email")
        if not email or email == NOT_FOUND or str(email).strip() == "":
            missing_fields.append("Email Address")
            reasons.append("Email Address is missing.")
        else:
            clean_email = str(email).strip().lower()
            if not EMAIL_REGEX.match(clean_email):
                invalid_fields.append({"field": "Email Address", "reason": f"'{clean_email}' is not a valid email format."})
                reasons.append(f"Invalid email format: '{clean_email}'.")
            else:
                passed_fields.append("Email Address")

        # 3. Phone Number
        phone = fields.get("Phone") or fields.get("Candidate Phone")
        if not phone or phone == NOT_FOUND or str(phone).strip() == "":
            missing_fields.append("Phone Number")
            reasons.append("Phone Number is missing.")
        else:
            phone_digits = re.sub(r"\D", "", str(phone))
            if not (10 <= len(phone_digits) <= 15):
                invalid_fields.append({
                    "field": "Phone Number",
                    "reason": f"Phone has {len(phone_digits)} digits; expected between 10 and 15 digits."
                })
                reasons.append(f"Invalid phone number format: '{phone}' ({len(phone_digits)} digits).")
            else:
                passed_fields.append("Phone Number")

        # 4. Skills
        skills = fields.get("Skills") or fields.get("Candidate Skills")
        if not skills or skills == NOT_FOUND or str(skills).strip() == "" or (isinstance(skills, list) and len(skills) == 0):
            missing_fields.append("Skills")
            reasons.append("Skills are missing or empty.")
        else:
            passed_fields.append("Skills")

        if missing_fields:
            reasons.insert(0, f"Missing resume field(s): {', '.join(missing_fields)}.")

        is_valid = (len(missing_fields) == 0 and len(invalid_fields) == 0)

        return {
            "is_valid": is_valid,
            "doc_type": "Resume",
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "passed_fields": passed_fields,
            "reasons": reasons
        }

    @classmethod
    def validate_other(cls, fields: Dict[str, Any], text: str = "") -> Dict[str, Any]:
        """
        Validates documents in 'Other' category:
        Ensures extracted text content is readable and non-trivial.
        """
        missing_fields: List[str] = []
        invalid_fields: List[Dict[str, str]] = []
        passed_fields: List[str] = ["Document Content"]
        reasons: List[str] = []

        if len(text.strip()) < 20:
            invalid_fields.append({"field": "Document Content", "reason": "Text is too short to be meaningful."})
            reasons.append("Document has fewer than 20 readable characters.")
            is_valid = False
        else:
            is_valid = True

        return {
            "is_valid": is_valid,
            "doc_type": "Other",
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "passed_fields": passed_fields,
            "reasons": reasons
        }

    @classmethod
    def validate_document(
        cls,
        doc_type: str,
        fields: Dict[str, Any],
        text: str = ""
    ) -> Dict[str, Any]:
        """
        Unified document validator dispatcher.
        """
        type_clean = (doc_type or "Other").capitalize()
        if type_clean == "Invoice":
            return cls.validate_invoice(fields, text)
        elif type_clean == "Resume":
            return cls.validate_resume(fields, text)
        else:
            return cls.validate_other(fields, text)
