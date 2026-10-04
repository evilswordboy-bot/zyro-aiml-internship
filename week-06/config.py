"""
Central configuration module for the AI Document Intelligence & Workflow Platform.
Loads configuration from environment variables or defaults.
Week 6: Production Hardening, Security & Environment Isolation.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "documents.db")))
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"

# Security & Upload Constraints
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}

# ML & Workflow Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
DEFAULT_CLASSIFICATION_MODEL = os.getenv("DEFAULT_CLASSIFICATION_MODEL", "Logistic Regression")

# RAG & Document AI Assistant Settings
RAG_CHUNK_SIZE_WORDS = int(os.getenv("RAG_CHUNK_SIZE_WORDS", "180"))
RAG_CHUNK_OVERLAP_WORDS = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "30"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.12"))

# Anomaly Detection Settings
ANOMALY_ARITHMETIC_TOLERANCE = float(os.getenv("ANOMALY_ARITHMETIC_TOLERANCE", "0.05"))
ANOMALY_MAX_INVOICE_AMOUNT = float(os.getenv("ANOMALY_MAX_INVOICE_AMOUNT", "1000000.0"))
ANOMALY_MAX_FUTURE_DAYS = int(os.getenv("ANOMALY_MAX_FUTURE_DAYS", "365"))

# Optional External API Keys (Safe Fallback to Local Offline NLP)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
APP_ENV = os.getenv("APP_ENV", "production")


def get_config_summary() -> dict:
    """Returns safe, non-sensitive configuration diagnostics for the UI."""
    return {
        "Environment": APP_ENV,
        "Database Path": str(DATABASE_PATH),
        "Storage Directory": str(STORAGE_DIR),
        "Max Upload Size": f"{MAX_FILE_SIZE_MB} MB",
        "Allowed Formats": ", ".join(sorted(ALLOWED_EXTENSIONS)),
        "Confidence Threshold": f"{CONFIDENCE_THRESHOLD:.0%}",
        "RAG Similarity Threshold": f"{RAG_SIMILARITY_THRESHOLD:.2f}",
        "External LLM Configured": bool(GEMINI_API_KEY or OPENAI_API_KEY),
    }
