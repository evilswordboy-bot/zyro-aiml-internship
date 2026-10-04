"""
Text Cleaner and Normalization Module
Provides systematic preprocessing to remove whitespace anomalies,
repeated blank lines, non-printable extraction noise, and edge cases.
Maintains both original and cleaned text for complete transparency.
"""

import re
import unicodedata
from typing import Dict, Any


class TextCleaner:
    """Cleans and standardizes raw extracted text from PDFs and OCR."""

    @staticmethod
    def clean(text: str) -> Dict[str, Any]:
        """
        Normalizes raw extracted text while preserving semantic layout.

        Returns:
            Dict containing:
                - original_text: Raw input string
                - cleaned_text: Normalized string
                - char_count_original: Length before cleaning
                - char_count_cleaned: Length after cleaning
                - reduction_pct: Percentage reduction in characters
                - is_valid: False if empty or extremely short (< 15 chars)
                - warning: Advisory note if text is abnormally short
        """
        if not text:
            return {
                "original_text": "",
                "cleaned_text": "",
                "char_count_original": 0,
                "char_count_cleaned": 0,
                "reduction_pct": 0.0,
                "is_valid": False,
                "warning": "Text is completely empty."
            }

        original = text

        # 1. Normalize Unicode (e.g. smart quotes, non-breaking spaces, em-dashes)
        normalized = unicodedata.normalize("NFKC", original)

        # 2. Replace non-breaking spaces and irregular tabs with standard space
        normalized = re.sub(r"[\u00A0\u1680\u180e\u2000-\u200b\u202f\u205f\u3000\ufeff]", " ", normalized)

        # 3. Strip non-printable ASCII control characters except \n, \r, \t
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", normalized)

        # 4. Standardize line endings (\r\n -> \n, \r -> \n)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # 5. Clean line-by-line whitespace
        cleaned_lines = []
        for line in normalized.split("\n"):
            # Collapse multiple consecutive horizontal spaces to a single space
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            cleaned_lines.append(cleaned_line)

        # 6. Recombine and collapse 3 or more consecutive blank lines to at most 2
        joined_text = "\n".join(cleaned_lines)
        collapsed_text = re.sub(r"\n{3,}", "\n\n", joined_text).strip()

        # 7. Check validity & length
        char_orig = len(original)
        char_clean = len(collapsed_text)
        reduction = round(((char_orig - char_clean) / char_orig * 100), 2) if char_orig > 0 else 0.0

        is_valid = char_clean >= 15
        warning = None
        if not is_valid:
            warning = f"Extracted text is extremely short ({char_clean} characters). Document might be blank or corrupted."

        return {
            "original_text": original,
            "cleaned_text": collapsed_text,
            "char_count_original": char_orig,
            "char_count_cleaned": char_clean,
            "reduction_pct": reduction,
            "is_valid": is_valid,
            "warning": warning
        }
