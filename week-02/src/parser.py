"""
Information Extraction Engine for AI Document Intelligence Platform.
Extracts key structured entities from Invoices and Resumes:
- Invoice: Invoice Number, Date, Company/Vendor Name, Total Amount
- Resume: Candidate Name, Email, Phone, Categorized Skills
Provides extraction confidence, context snippets, and explainability.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .config import PATTERNS, SKILLS_TAXONOMY


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with confidence and context."""
    name: str
    value: Any
    found: bool
    confidence: float
    context_snippet: str = ""
    validation_status: str = "Extracted"


@dataclass
class ParseResult:
    """Structured container for all extracted fields of a document."""
    document_type: str
    fields: Dict[str, ExtractedEntity] = field(default_factory=dict)
    skills_by_category: Dict[str, List[str]] = field(default_factory=dict)
    all_skills_flat: List[str] = field(default_factory=list)
    fields_found_count: int = 0
    total_expected_fields: int = 0
    completeness_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into clean dictionary for API/JSON export."""
        simple_fields = {}
        for k, v in self.fields.items():
            simple_fields[k] = {
                "value": v.value,
                "found": v.found,
                "confidence": v.confidence,
                "context": v.context_snippet,
                "status": v.validation_status
            }
        return {
            "document_type": self.document_type,
            "fields": simple_fields,
            "skills_by_category": self.skills_by_category,
            "completeness_score": self.completeness_score,
            "fields_found_count": self.fields_found_count,
            "total_expected_fields": self.total_expected_fields
        }


class DocumentParser:
    """Extracts entities from raw document text based on identified document type."""

    def parse(self, text: str, document_type: str) -> ParseResult:
        if document_type == "Invoice":
            return self._parse_invoice(text)
        elif document_type == "Resume":
            return self._parse_resume(text)
        else:
            return self._parse_other(text)

    def _parse_invoice(self, text: str) -> ParseResult:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        fields: Dict[str, ExtractedEntity] = {}

        # 1. Invoice Number
        inv_num, inv_conf, inv_snippet = self._extract_invoice_number(text)
        fields["Invoice Number"] = ExtractedEntity(
            name="Invoice Number",
            value=inv_num if inv_num else "Not found",
            found=bool(inv_num),
            confidence=inv_conf,
            context_snippet=inv_snippet,
            validation_status="Verified Format" if inv_num else "Missing"
        )

        # 2. Date
        date_val, date_conf, date_snippet = self._extract_date(text)
        fields["Date"] = ExtractedEntity(
            name="Date",
            value=date_val if date_val else "Not found",
            found=bool(date_val),
            confidence=date_conf,
            context_snippet=date_snippet,
            validation_status="Standard Date" if date_val else "Missing"
        )

        # 3. Company Name
        comp_val, comp_conf, comp_snippet = self._extract_company_name(text, lines)
        fields["Company Name"] = ExtractedEntity(
            name="Company Name",
            value=comp_val if comp_val else "Not found",
            found=bool(comp_val),
            confidence=comp_conf,
            context_snippet=comp_snippet,
            validation_status="Detected Entity" if comp_val else "Missing"
        )

        # 4. Total Amount
        total_val, total_conf, total_snippet = self._extract_total_amount(text)
        fields["Total Amount"] = ExtractedEntity(
            name="Total Amount",
            value=total_val if total_val else "Not found",
            found=bool(total_val),
            confidence=total_conf,
            context_snippet=total_snippet,
            validation_status="Monetary Value" if total_val else "Missing"
        )

        found_count = sum(1 for f in fields.values() if f.found)
        total_count = len(fields)
        completeness = round((found_count / total_count) * 100, 1)

        return ParseResult(
            document_type="Invoice",
            fields=fields,
            fields_found_count=found_count,
            total_expected_fields=total_count,
            completeness_score=completeness
        )

    def _parse_resume(self, text: str) -> ParseResult:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        fields: Dict[str, ExtractedEntity] = {}

        # 1. Candidate Name
        name_val, name_conf, name_snippet = self._extract_candidate_name(lines)
        fields["Name"] = ExtractedEntity(
            name="Name",
            value=name_val if name_val else "Not found",
            found=bool(name_val),
            confidence=name_conf,
            context_snippet=name_snippet,
            validation_status="Header Detection" if name_val else "Missing"
        )

        # 2. Email Address
        email_val, email_conf, email_snippet = self._extract_email(text)
        fields["Email"] = ExtractedEntity(
            name="Email",
            value=email_val if email_val else "Not found",
            found=bool(email_val),
            confidence=email_conf,
            context_snippet=email_snippet,
            validation_status="RFC 5322 Valid" if email_val else "Missing"
        )

        # 3. Phone Number
        phone_val, phone_conf, phone_snippet = self._extract_phone(text)
        fields["Phone"] = ExtractedEntity(
            name="Phone",
            value=phone_val if phone_val else "Not found",
            found=bool(phone_val),
            confidence=phone_conf,
            context_snippet=phone_snippet,
            validation_status="E.164 / Local Format" if phone_val else "Missing"
        )

        # 4. Skills Taxonomy Match
        skills_cat, all_skills = self._extract_skills(text)
        skills_display = ", ".join(all_skills[:8]) + (f" (+{len(all_skills)-8} more)" if len(all_skills) > 8 else "")
        fields["Skills"] = ExtractedEntity(
            name="Skills",
            value=skills_display if all_skills else "Not found",
            found=bool(all_skills),
            confidence=0.92 if len(all_skills) >= 3 else (0.75 if all_skills else 0.0),
            context_snippet=f"{len(all_skills)} tech skills identified across {len(skills_cat)} categories.",
            validation_status="Taxonomy Match" if all_skills else "Missing"
        )

        found_count = sum(1 for f in fields.values() if f.found)
        total_count = len(fields)
        completeness = round((found_count / total_count) * 100, 1)

        return ParseResult(
            document_type="Resume",
            fields=fields,
            skills_by_category=skills_cat,
            all_skills_flat=all_skills,
            fields_found_count=found_count,
            total_expected_fields=total_count,
            completeness_score=completeness
        )

    def _parse_other(self, text: str) -> ParseResult:
        """Fallback extraction for unstructured or other documents."""
        fields: Dict[str, ExtractedEntity] = {
            "Status": ExtractedEntity(
                name="Status",
                value="General / Unstructured Document",
                found=True,
                confidence=0.85,
                context_snippet="Document not classified as Invoice or Resume. No structured entities extracted.",
                validation_status="Informational"
            )
        }
        return ParseResult(
            document_type="Other",
            fields=fields,
            fields_found_count=1,
            total_expected_fields=1,
            completeness_score=100.0
        )

    # -------------------------------------------------------------
    # Invoice Extraction Helpers
    # -------------------------------------------------------------
    def _extract_invoice_number(self, text: str) -> tuple[Optional[str], float, str]:
        for pat in PATTERNS["invoice_number"]:
            match = re.search(pat, text)
            if match:
                inv_no = match.group(1).strip(" :.,#")
                # Filter out accidental common header words
                if inv_no.lower() not in {"date", "due", "total", "amount", "bill", "pkr", "usd"}:
                    snippet = text[max(0, match.start()-10):min(len(text), match.end()+15)].strip()
                    return inv_no, 0.95, f"...{snippet}..."
        return None, 0.0, ""

    def _extract_date(self, text: str) -> tuple[Optional[str], float, str]:
        for pat in PATTERNS["date"]:
            match = re.search(pat, text)
            if match:
                date_str = match.group(0).strip(" :.,")
                snippet = text[max(0, match.start()-10):min(len(text), match.end()+15)].strip()
                return date_str, 0.90, f"...{snippet}..."
        return None, 0.0, ""

    def _extract_company_name(self, text: str, lines: List[str]) -> tuple[Optional[str], float, str]:
        # Rule A: Look for "Bill To:" or "Billed To:" or "Vendor:"
        bill_to_pat = r"(?i)(?:bill\s+to|billed\s+to|vendor|company|client)[\s:]+([A-Za-z0-9&.,\s'-]{3,40})"
        match = re.search(bill_to_pat, text)
        if match:
            candidate = match.group(1).split("\n")[0].strip(" :,.")
            if len(candidate) > 2 and not candidate.lower().startswith("invoice"):
                snippet = text[max(0, match.start()-5):min(len(text), match.end()+10)].strip()
                return candidate, 0.88, f"...{snippet}..."

        # Rule B: First non-generic line of the document
        ignore_words = {"tax invoice", "invoice", "proforma invoice", "receipt", "statement"}
        for line in lines[:5]:
            clean_line = line.strip(" :-.,#")
            if clean_line.lower() not in ignore_words and len(clean_line) >= 3 and not re.match(r"^\d+$", clean_line):
                return clean_line, 0.75, f"Top Header: {clean_line}"

        return None, 0.0, ""

    def _extract_total_amount(self, text: str) -> tuple[Optional[str], float, str]:
        for pat in PATTERNS["currency_amount"]:
            match = re.search(pat, text)
            if match:
                groups = match.groups()
                # Determine which group is currency and which is number
                val1, val2 = groups[0], groups[1] if len(groups) > 1 else ""
                combined = f"{val1} {val2}".strip() if val2 else val1.strip()
                snippet = text[max(0, match.start()-15):min(len(text), match.end()+15)].strip()
                return combined, 0.92, f"...{snippet}..."
        return None, 0.0, ""

    # -------------------------------------------------------------
    # Resume Extraction Helpers
    # -------------------------------------------------------------
    def _extract_candidate_name(self, lines: List[str]) -> tuple[Optional[str], float, str]:
        # Most resumes feature candidate name in the first 3 lines
        for line in lines[:4]:
            clean = line.strip()
            # Ignore headers like 'Curriculum Vitae', 'Resume'
            if clean.lower() in {"curriculum vitae", "resume", "cv", "profile"}:
                continue
            # Check if looks like a name (2 to 4 words, alphabetic)
            words = clean.split()
            if 2 <= len(words) <= 4 and all(re.match(r"^[A-Za-z.'-]+$", w) for w in words):
                return clean, 0.88, f"Top line: '{clean}'"

        # Look for "Name: John Doe"
        name_pat = r"(?i)\bname[\s:]+([A-Za-z\s.'-]{3,35})"
        match = re.search(name_pat, "\n".join(lines))
        if match:
            cand = match.group(1).strip()
            return cand, 0.85, f"...{match.group(0)}..."

        return None, 0.0, ""

    def _extract_email(self, text: str) -> tuple[Optional[str], float, str]:
        match = re.search(PATTERNS["email"], text)
        if match:
            email_str = match.group(0).strip()
            snippet = text[max(0, match.start()-10):min(len(text), match.end()+10)].strip()
            return email_str, 0.98, f"...{snippet}..."
        return None, 0.0, ""

    def _extract_phone(self, text: str) -> tuple[Optional[str], float, str]:
        match = re.search(PATTERNS["phone"], text)
        if match:
            phone_str = match.group(0).strip()
            if len(re.sub(r"\D", "", phone_str)) >= 7:
                snippet = text[max(0, match.start()-10):min(len(text), match.end()+10)].strip()
                return phone_str, 0.90, f"...{snippet}..."
        return None, 0.0, ""

    def _extract_skills(self, text: str) -> tuple[Dict[str, List[str]], List[str]]:
        lower_text = text.lower()
        skills_by_category: Dict[str, List[str]] = {}
        all_skills: List[str] = []

        for category, skill_list in SKILLS_TAXONOMY.items():
            matched_for_cat = []
            for skill in skill_list:
                # Word boundary match
                pattern = rf"\b{re.escape(skill.lower())}\b"
                if re.search(pattern, lower_text):
                    matched_for_cat.append(skill)
                    if skill not in all_skills:
                        all_skills.append(skill)
            if matched_for_cat:
                skills_by_category[category] = matched_for_cat

        return skills_by_category, all_skills
