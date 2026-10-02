"""
Unit Tests for Text Cleaning and Normalization Module
"""

import pytest
from src.text_cleaner import TextCleaner


def test_clean_normal_text():
    raw = "   Invoice   Number:   INV-2026   \n\n\n\nDate:  15/03/2026   "
    res = TextCleaner.clean(raw)

    assert res["is_valid"] is True
    assert "Invoice Number: INV-2026" in res["cleaned_text"]
    assert "\n\n\n" not in res["cleaned_text"]
    assert res["char_count_cleaned"] < res["char_count_original"]


def test_clean_empty_text():
    res = TextCleaner.clean("")
    assert res["is_valid"] is False
    assert res["cleaned_text"] == ""
    assert res["warning"] == "Text is completely empty."


def test_clean_extremely_short_text():
    res = TextCleaner.clean("Short")
    assert res["is_valid"] is False
    assert "extremely short" in res["warning"]


def test_clean_unicode_and_control_chars():
    raw = "TechCorp\u00A0Solutions\t\tInc.\x00\x07\nTotal: $1,450.00"
    res = TextCleaner.clean(raw)
    assert "\u00A0" not in res["cleaned_text"]
    assert "\x00" not in res["cleaned_text"]
    assert "TechCorp Solutions Inc." in res["cleaned_text"]
