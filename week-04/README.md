# AI Document Intelligence & Workflow Platform

**Enterprise-Grade Document Understanding, Persistent Repository, Deduplication & Workflow Management**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://www.sqlite.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.24+-green.svg)](https://pymupdf.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Deployed-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)

---

## 📌 Executive Summary

**AI Document Intelligence & Workflow Platform** is an enterprise-grade document management and understanding system designed to transition ad-hoc document processing into a **persistent, searchable, and audit-ready document platform**.

Moving beyond one-time parsing, this platform validates incoming files, computes tamper-evident **SHA-256 cryptographic digests** to suppress redundant duplicate uploads, normalizes unstructured text, classifies document categories (**Invoice**, **Resume**, **Other**) using evaluated machine learning models, extracts structured domain entities, assigns operational statuses (**Processed**, **Needs Review**, **Failed**), and stores both physical documents and rich metadata in an organized, parameterized **SQLite repository**.

---

## 🏛️ End-to-End Workflow Architecture

```
                       ┌────────────────────────┐
                       │  User Document Upload  │
                       │  (PDF, JPG, JPEG, PNG) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ File Validation Layer  │
                       │ Format & Size (<20MB)  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Cryptographic SHA-256  │
                       │ Digest Computation     │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Duplicate Lookup Check │
                       │ (data/documents.db)    │
                       └───────────┬────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
    [Duplicate Hash Exists]                      [Brand New Document]
            │                                             │
            ▼                                             ▼
  ┌───────────────────┐                         ┌───────────────────┐
  │ Suppress Write &  │                         │ Text Extraction   │
  │ Render Stored Doc │                         │ (PyMuPDF / OCR)   │
  └───────────────────┘                         └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ Text Cleaning &   │
                                                │ Normalization     │
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ ML Classification │
                                                │ (TF-IDF + Models) │
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ Entity Extraction │
                                                │ (Invoices/Resumes)│
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ Status Assignment │
                                                │ (Processed/Review)│
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ Organized Storage │
                                                │ storage/<category>│
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ SQLite Insert &   │
                                                │ Indexed Metadata  │
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ Search, Filter &  │
                                                │ Document Vault UI │
                                                └───────────────────┘
```

---

## 🚀 Key Platform Features

1. **Persistent SQLite Document Repository (`data/documents.db`)**:
   - Production-style schema with indexes on `file_hash`, `document_type`, `status`, `upload_date`, `company`, and `invoice_number`.
   - SQLite Write-Ahead Logging (`WAL` mode) for high concurrency and zero database locking.
   - 100% parameterized SQL queries preventing any SQL injection vulnerabilities.
2. **Cryptographic SHA-256 Duplicate Suppression**:
   - Computes deterministic SHA-256 hashes directly from raw file bytes.
   - Prevents duplicate uploads from cluttering disk storage or producing redundant database records.
   - Provides instant feedback alerting the user with an option to inspect the existing document record.
3. **Structured & Safe File Storage**:
   - Categorized directories: `storage/invoices/`, `storage/resumes/`, `storage/other/`.
   - Generates collision-resistant, sanitized filenames: `<type>_<YYYYMMDD_HHMMSS>_<short_hash>.<ext>`.
   - Strict path traversal guards preventing unauthorized filesystem access outside the repository root.
4. **Multi-Field Parameterized Search**:
   - Lightning-fast search across Original Filename, Company Name, Invoice Number, Candidate Name, Document Category, and Text Preview.
   - Evaluated directly in SQLite via parameterized `LIKE` operations.
5. **Interactive Filters & Sorting**:
   - Filter by Document Type (`Invoice`, `Resume`, `Other`).
   - Filter by Status (`Processed`, `Needs Review`, `Failed`).
   - Dynamic sorting: Newest First or Oldest First.
   - One-click filter reset.
6. **Document Detail Inspection Panel**:
   - Visual inspection of original vs. stored filenames, upload timestamp, cryptographic digest, and storage location.
   - Extracted key business and candidate entities with truthful `"Not Found"` safety defaults.
   - Full 1,000-character text preview and direct one-click file download from the vault.
7. **Document Processing Lifecycle Statuses**:
   - `🟢 Processed`: Document read, classified, and all critical fields identified.
   - `🟡 Needs Review`: Missing critical fields (e.g. invoice total or candidate contact info) requiring human verification.
   - `🔴 Failed`: Unreadable, zero text extracted, or unsupported files.
8. **Multi-Model Machine Learning Engine**:
   - Pre-trained models: Logistic Regression, Linear SVM (`LinearSVC`), Multinomial Naive Bayes, and Rule-Based Baseline.
   - Includes real classification metrics, confusion matrices, and scientific error diagnostics.

---

## 📊 Dataset & Model Benchmarks

### Benchmark Results on Independent Test Corpus

| Classifier Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Inference Latency | Confidence Output |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based Baseline** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | < 1 ms | Deterministic |
| **Logistic Regression** | **0.8889** | **0.9167** | **0.8889** | **0.8857** | ~ 4 ms | Calibrated Probabilities |
| **Linear SVM (`LinearSVC`)** | **0.8889** | **0.9167** | **0.8889** | **0.8857** | ~ 2 ms | Margin (No Probabilities) |
| **Multinomial Naive Bayes** | **0.8889** | **0.9167** | **0.8889** | **0.8857** | ~ 3 ms | Posterior Probabilities |

### 3x3 Test Confusion Matrix

```
                  Predicted
             Invoice  Resume   Other
  Actual
  Invoice       3        0       0
  Resume        0        3       0
  Other         1        0       2
```

---

## 📁 Repository Structure

```
ai-document-intelligence/
├── app.py                      # Multi-tab Streamlit Enterprise SaaS Dashboard
├── requirements.txt            # Production Python package dependencies
├── run_offline.bat             # 1-Click offline local launcher for Windows
├── README.md                   # Complete architectural and technical documentation
│
├── database/
│   ├── __init__.py
│   └── db.py                   # SQLite Connection Manager, Schema, CRUD & Search queries
│
├── services/
│   ├── __init__.py
│   ├── hashing.py              # SHA-256 digest computation service
│   ├── file_storage.py         # Categorized storage, safe naming, path traversal guard
│   ├── validation.py           # Extension whitelisting (.pdf, .jpg, .jpeg, .png) & size guard
│   ├── document_processor.py   # Full pipeline orchestrator (Validate -> Hash -> Process -> Store)
│   ├── ocr.py                  # OCR service wrapper
│   ├── classifier.py           # Classifier service wrapper
│   └── extractor.py            # Information extraction wrapper
│
├── models/
│   ├── __init__.py
│   ├── document.py             # DocumentRecord dataclass and status constants
│   └── classifier.pkl          # Serialized TF-IDF vectorizer + trained ML models
│
├── storage/                    # Persistent organized physical document vault
│   ├── invoices/               # Categorized invoice files
│   ├── resumes/                # Categorized resume files
│   └── other/                  # Categorized miscellaneous files
│
├── data/
│   ├── documents.db            # Persistent SQLite database
│   ├── train/                  # 15 labeled training files across classes
│   └── test/                   # 9 independent test files across classes
│
├── samples/                    # Verified authentic test documents
│   ├── invoice_1.pdf           # TechCorp Solutions invoice ($1,450.00)
│   ├── invoice_2.pdf           # Apex Retailers invoice (Rs. 78,500)
│   ├── invoice_missing_total.pdf # Missing total field test invoice
│   ├── resume_1.pdf            # Alex Smith software engineer resume
│   ├── resume_missing_phone.pdf # Missing phone field test resume
│   └── meeting_minutes.pdf     # Miscellaneous corporate document (Other)
│
├── src/                        # Modular AI core
│   ├── document_reader.py      # PyMuPDF ingestion & selectable text extractor
│   ├── ocr_processor.py        # Adaptive image preprocessing & Tesseract fallback
│   ├── text_cleaner.py         # Whitespace, unicode NFKC & control character normalizer
│   ├── classifier.py           # TF-IDF feature space & multi-model classifier suite
│   ├── extractor.py            # Regex domain entity extractors with missing field logic
│   ├── evaluator.py            # Evaluation metrics & confusion matrix computer
│   └── utils.py                # Dataset loaders & sample generators
│
└── tests/
    ├── conftest.py             # Pytest discovery configuration
    ├── test_database.py        # SQLite CRUD, hashing & search tests
    ├── test_storage.py         # File storage, path defense & hashing tests
    ├── test_processor_e2e.py   # End-to-end processing & duplicate integration tests
    ├── test_week4_acceptance.py# Complete 10+ document acceptance matrix & persistence tests
    ├── test_text_cleaning.py   # Text normalization unit tests
    ├── test_classifier.py      # Model training & inference tests
    ├── test_extraction.py      # Entity extraction & 'Not Found' tests
    └── test_pipeline_e2e.py    # Week 3 backwards-compatible pipeline tests
```

---

## 🛠️ Installation & Quickstart

### Prerequisites
- **Python 3.10** or higher
- Optional: [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for scanned raster images)

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/evilswordboy-bot/ai-document-intelligence.git
cd ai-document-intelligence

# Create virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```
*Expected: **43 passed** in ~5.4s*

### 4. Launch the Platform
```bash
streamlit run app.py
```
Or on Windows, simply double-click **`run_offline.bat`** to start offline without any command typing.

---

## 🧪 Week 4 Acceptance Test Matrix

| Test Case | Document / Input | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Standard Invoice** | `invoice_1.pdf` | Stored in `invoices/`, classified as Invoice, Processed | Stored & indexed cleanly | ✅ PASS |
| **Tax Invoice (INR)** | `invoice_2.pdf` | Stored in `invoices/`, currency Rs. 78,500 extracted | Extracted cleanly | ✅ PASS |
| **Candidate Resume** | `resume_1.pdf` | Stored in `resumes/`, Name, Email & Skills extracted | Extracted cleanly | ✅ PASS |
| **General Document** | `meeting_minutes.pdf` | Stored in `other/`, classified as Other | Indexed cleanly | ✅ PASS |
| **Duplicate Upload** | Identical `invoice_1.pdf` | SHA-256 match, duplicate suppressed, no duplicate file or row | Existing record displayed | ✅ PASS |
| **Missing Field Invoice**| `invoice_missing_total.pdf` | Total "Not Found", status flagged as "Needs Review" | Flagged as Needs Review | ✅ PASS |
| **Missing Field Resume** | `resume_missing_phone.pdf` | Phone "Not Found", status flagged as "Needs Review" | Flagged as Needs Review | ✅ PASS |
| **Scanned Document** | `receipt_scanned.png` | OCR image pipeline triggered, file stored in vault | Stored and indexed | ✅ PASS |
| **Invalid Extension** | `malware.exe` | Upload rejected with friendly error message | Rejected safely | ✅ PASS |
| **Oversized File** | > 20 MB file | Upload rejected exceeding upload size threshold | Rejected safely | ✅ PASS |
| **Multi-Field Search** | Query: "TechCorp" | Parameterized SQL query returns matching invoice | Matching record returned | ✅ PASS |
| **Status Filter** | Filter: "Needs Review"| Returns only flagged documents | Correct subset returned | ✅ PASS |
| **Sort Order** | Newest vs. Oldest | Documents ordered by ISO upload date | Correct temporal ordering | ✅ PASS |
| **Restart Persistence**| Stop & restart DB | SQLite data, files, and queries persist intact | 100% persisted | ✅ PASS |

---

## 🔒 Security & Safe Engineering

- **Path Traversal Defense**: All file paths are strictly resolved and validated to prevent directory traversal (`../`).
- **SQL Injection Prevention**: All queries use parameterized SQL placeholders (`?`).
- **No Raw Exception Leaks**: User interface displays clean, friendly status alerts while logging technical diagnostics internally.
- **Controlled File Naming**: Stored files use programmatic timestamped identifiers, preventing file overwrites or execution.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
