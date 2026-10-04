"""
Information Extraction Module
Extracts structured business and candidate entities from Invoices and Resumes.
Gracefully detects and records missing fields as 'Not Found'.
"""

import re
from typing import Dict, Any, List

NOT_FOUND = "Not Found"


class DocumentExtractor:
    """Regex and heuristic information extraction for Invoices and Resumes."""

    @staticmethod
    def extract_invoice_fields(text: str) -> Dict[str, Any]:
        """
        Extracts:
        - Invoice Number
        - Date
        - Company Name
        - Total Amount
        """
        fields = {
            "Invoice Number": NOT_FOUND,
            "Date": NOT_FOUND,
            "Company Name": NOT_FOUND,
            "Total Amount": NOT_FOUND
        }

        # 1. Invoice Number Patterns
        inv_patterns = [
            r"(?i)\b(?:invoice\s*(?:no\.?|number|#|id)|inv\s*(?:no\.?|number|#)|bill\s*(?:no\.?|number|#))\s*[:#\-]?\s*([A-Za-z0-9\-_/]+)",
            r"(?i)\binvoice\s*[:#]\s*([A-Za-z0-9\-_/]+)",
            r"(?i)\b(INV-[A-Za-z0-9\-_/]+)\b",
            r"(?i)\b(INV\d+)\b"
        ]
        for pattern in inv_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip().strip(":,")
                if len(candidate) >= 3 and candidate.lower() not in ["invoice", "tax", "date", "bill", "due"]:
                    fields["Invoice Number"] = candidate
                    break

        # 2. Date Patterns
        date_patterns = [
            r"(?i)\b(?:date|invoice\s*date|dated|issue\s*date)\s*[:#\-]?\s*([0-9]{1,4}[-/.][0-9]{1,2}[-/.][0-9]{1,4})",
            r"(?i)\b(?:date|invoice\s*date|dated)\s*[:#\-]?\s*([A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4})",
            r"(?i)\b(?:date|invoice\s*date|dated)\s*[:#\-]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9},?\s+[0-9]{4})",
            r"\b([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})\b",
            r"\b([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})\b"
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip()
                if any(c.isdigit() for c in candidate):
                    fields["Date"] = candidate
                    break

        # 3. Company Name Patterns (Header lines heuristics)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        stop_terms = ["invoice", "tax invoice", "commercial invoice", "bill to", "billed to", "date", "due date", "page", "phone", "email", "gstin"]
        corp_indicators = ["inc", "ltd", "pvt", "corp", "corporation", "solutions", "retailers", "technologies", "systems", "services", "company", "enterprises", "agency", "labs"]

        company_candidate = None
        for line in lines[:8]:
            lower_line = line.lower()
            if any(lower_line.startswith(term) for term in stop_terms):
                continue
            if any(re.search(rf"\b{ind}\b", lower_line) for ind in corp_indicators):
                company_candidate = line
                break

        if not company_candidate and lines:
            for line in lines[:5]:
                if line.lower() not in ["invoice", "tax invoice", "commercial invoice"] and len(line) > 3:
                    company_candidate = line
                    break

        if company_candidate:
            fields["Company Name"] = company_candidate.strip()

        # 4. Total Amount Patterns
        amount_patterns = [
            r"(?i)(?:total\s*amount|grand\s*total|amount\s*due|balance\s*due|total\s*balance\s*due|total)\s*[:#\-]?\s*([₹$€£]|USD|INR|PKR|Rs\.?)?\s*([0-9,]+(?:\.[0-9]{2})?)",
            r"(?i)([₹$€£]|USD|INR|PKR|Rs\.?)\s*([0-9,]+(?:\.[0-9]{2})?)",
            r"(?i)(?:total|due)\s*[:#\-]?\s*([0-9,]+(?:\.[0-9]{2})?)"
        ]
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for m in reversed(matches):
                    if isinstance(m, tuple):
                        curr = m[0].strip() if len(m) > 1 and m[0] else ""
                        val = m[1].strip() if len(m) > 1 else m[0].strip()
                    else:
                        curr = ""
                        val = m.strip()

                    if val and any(c.isdigit() for c in val):
                        fields["Total Amount"] = f"{curr} {val}".strip() if curr else val
                        break
                if fields["Total Amount"] != NOT_FOUND:
                    break

        # Compute missing fields list
        missing = [k for k, v in fields.items() if v == NOT_FOUND]
        return {
            "fields": fields,
            "missing_fields": missing,
            "fields_found_count": len(fields) - len(missing),
            "total_fields": len(fields)
        }

    @staticmethod
    def extract_resume_fields(text: str) -> Dict[str, Any]:
        """
        Extracts:
        - Name
        - Email
        - Phone
        - Skills
        """
        fields = {
            "Name": NOT_FOUND,
            "Email": NOT_FOUND,
            "Phone": NOT_FOUND,
            "Skills": NOT_FOUND
        }

        # 1. Email Pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        email_match = re.search(email_pattern, text)
        if email_match:
            fields["Email"] = email_match.group(0).strip()

        # 2. Phone Pattern
        phone_pattern = r"(?:(?:\+|00)\d{1,3}[\s-]?)?(?:\(?\d{2,5}\)?[\s-]?)?\d{3,5}[\s-]?\d{3,5}"
        phone_matches = re.findall(phone_pattern, text)
        for p in phone_matches:
            digits = re.sub(r"\D", "", p)
            if 10 <= len(digits) <= 14:
                fields["Phone"] = p.strip()
                break

        # 3. Candidate Name Heuristics
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        invalid_name_terms = [
            "resume", "curriculum", "vitae", "cv", "email", "phone", "profile",
            "summary", "experience", "education", "skills", "page", "http", "www", "github", "linkedin"
        ]

        for line in lines[:5]:
            lower_line = line.lower()
            if any(term in lower_line for term in invalid_name_terms):
                continue
            if "@" in line or any(c.isdigit() for c in line):
                continue
            words = line.split()
            if 1 <= len(words) <= 4:
                fields["Name"] = line.strip()
                break

        # 4. Skills Extraction
        skills_pattern = r"(?i)(?:skills\s*(?:&|and)?\s*(?:technical\s*expertise|tools)?|technical\s*skills|core\s*(?:competencies|skills)|technical\s*proficiencies)\s*[:\-\n]([\s\S]*?)(?=\n\s*(?:work\s*experience|experience|education|projects|certifications|employment)|$)"
        skills_match = re.search(skills_pattern, text)

        extracted_skills = []
        if skills_match:
            skills_block = skills_match.group(1).strip()
            raw_skills = re.split(r"[,•|\n;]", skills_block)
            for s in raw_skills:
                clean_s = re.sub(
                    r"^(?:languages|frameworks\s*&\s*tools|frameworks|tools|technologies|competencies|core\s*competencies|cloud\s*&\s*infra|programming|libraries)\s*[:\-]",
                    "", s, flags=re.IGNORECASE
                ).strip()
                if clean_s and len(clean_s) <= 40 and not any(clean_s.lower().startswith(b) for b in ["experience", "education", "worked"]):
                    extracted_skills.append(clean_s)

        # Fallback technology vocabulary match
        common_skills = [
            "Python", "SQL", "Java", "C++", "JavaScript", "TypeScript", "HTML", "CSS", "R", "Go", "Julia",
            "Streamlit", "PyMuPDF", "FastAPI", "Flask", "Django", "Docker", "Kubernetes", "Ray",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "PyTorch", "TensorFlow",
            "Scikit-learn", "Pandas", "NumPy", "Git", "PostgreSQL", "Redis", "MongoDB",
            "AWS", "GCP", "Azure", "Linux", "REST APIs", "CI/CD", "Airflow", "Terraform", "Figma"
        ]

        if not extracted_skills or len(extracted_skills) < 2:
            for skill in common_skills:
                if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
                    if skill not in extracted_skills:
                        extracted_skills.append(skill)

        if extracted_skills:
            unique_skills = []
            for s in extracted_skills:
                if s and s not in unique_skills:
                    unique_skills.append(s)
            fields["Skills"] = ", ".join(unique_skills[:12])

        missing = [k for k, v in fields.items() if v == NOT_FOUND]
        return {
            "fields": fields,
            "missing_fields": missing,
            "fields_found_count": len(fields) - len(missing),
            "total_fields": len(fields)
        }

    @classmethod
    def extract(cls, text: str, document_type: str) -> Dict[str, Any]:
        """Routes text to the appropriate entity extraction pipeline."""
        if document_type == "Invoice":
            return cls.extract_invoice_fields(text)
        elif document_type == "Resume":
            return cls.extract_resume_fields(text)
        else:
            return {
                "fields": {
                    "Notice": "Document classified as 'Other'. Structured field extraction is specialized for Invoices and Resumes."
                },
                "missing_fields": [],
                "fields_found_count": 0,
                "total_fields": 0
            }
