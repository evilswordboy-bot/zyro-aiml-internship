# AI Document Intelligence & Workflow Platform

**Enterprise-Grade Document Understanding, Controlled State Machine, Validation & Anomaly Detection Engine, Grounded RAG Assistant & Human-in-the-Loop Workflow Automation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://www.sqlite.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.24+-green.svg)](https://pymupdf.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-105%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo (Cloudflare)](https://img.shields.io/badge/Live%20Demo-Cloudflare%20Edge-orange.svg?style=for-the-badge&logo=cloudflare)](https://incorporated-attempted-lucy-trusts.trycloudflare.com)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Deployed-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)

**🌐 Public Live URL (Cloudflare Edge):** **[https://incorporated-attempted-lucy-trusts.trycloudflare.com](https://incorporated-attempted-lucy-trusts.trycloudflare.com)**  
**🌐 Public Live URL (Streamlit Cloud):** **[https://mgb56neehcvkyzsiukg2qh.streamlit.app/](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)**  
**📂 Primary Codebase:** [evilswordboy-bot/ai-document-intelligence](https://github.com/evilswordboy-bot/ai-document-intelligence)  
**📂 Internship Portfolio:** [evilswordboy-bot/zyro-aiml-internship](https://github.com/evilswordboy-bot/zyro-aiml-internship) (Week 6 active)

---

## 📌 Executive Summary (Week 6 Final Integration & Optimization)

**AI Document Intelligence & Workflow Platform** is a production-grade document intelligence system finalized for the **ZYROO AI/ML Internship (Week 6)**. Building upon Weeks 2–5, Week 6 integrates all document lifecycle stages into a single unified product:

$$\text{Upload} \longrightarrow \text{Validate} \longrightarrow \text{OCR / Text Extraction} \longrightarrow \text{Classify} \longrightarrow \text{Extract} \longrightarrow \text{Store} \longrightarrow \text{Verify} \longrightarrow \text{Anomaly Check} \longrightarrow \text{AI RAG} \longrightarrow \text{Workflow Decision} \longrightarrow \text{Review / Approve / Reject} \longrightarrow \text{Audit} \longrightarrow \text{Search / Report}$$

### Key Week 6 Capabilities:
1. **Financial Consistency & Anomaly Engine (`anomaly.py` / `services/anomaly.py`)**:
   - Automated invoice arithmetic checking: $|\text{subtotal} + \text{tax} - \text{total}| \le \$0.05$. Catches mathematical inconsistencies and routes them directly to `Needs Review`.
   - Outlier / suspicious amount detector: flags values $\le \$0$ or $> \$1,000,000$.
   - Future date validator: flags documents dated $> 365$ days into the future.
   - Text corruption & noise detection: flags excessive OCR noise or repetitive character sequences.
   - Duplicate hash collision detection.
2. **Grounded RAG & AI Document Assistant (`rag.py` / `services/rag.py` / `src/rag_engine.py`)**:
   - Semantic chunking with configurable overlap (words/tokens).
   - TF-IDF vector space modeling with cosine similarity ranking.
   - Grounded multi-document question answering with precise source citations: `[Filename] (Page X, Chunk Y) | Relevance: Z%`.
   - **Strict Anti-Hallucination Guardrail**: Queries without sufficient grounded context return:
     > *"The available documents do not contain sufficient information to answer this question."*
3. **End-to-End Latency Tracking & Profiling**:
   - Automated profiling of processing time (`processing_time_ms`) across all ingestion, extraction, and anomaly stages.
   - Live KPI cards displaying average system latency and quality metrics.
4. **Controlled State Machine & Human Review**:
   - Six explicit states: `New`, `Processing`, `Needs Review`, `Approved`, `Rejected`, `Completed`.
   - Strictly mandatory reviewer notes on rejection.
   - Fault-tolerant batch ingestion with isolated transaction boundaries.
5. **Security Hardening**:
   - Centralized configuration (`config.py`, `.env.example`).
   - Path traversal prevention using resolved directory boundaries.
   - Whitelisted file formats and sanitized error shields.
6. **Production Acceptance Testing**:
   - **105 out of 105 tests passing (100% pass rate)** across 16 test files.

---

## 🏛️ Full System Architecture

```
                    ┌────────────────────────┐
                    │  User Document Upload  │
                    │  (PDF, JPG, JPEG, PNG) │
                    └───────────┬────────────┘
                                │ State: New (Recorded in SQLite)
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
                    │ Duplicate Detection    │
                    │ (data/documents.db)    │
                    └───────────┬────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
 [Duplicate Hash Exists]                      [Brand New Document]
         │                                             │
         ▼                                             ▼
┌───────────────────┐                        ┌───────────────────┐
│ Suppress Storage  │                        │ State: Processing │
│ Render Vault Doc  │                        │ PyMuPDF / OCR     │
└───────────────────┘                        └─────────┬─────────┘
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
                                             │ (Regex + NLP)     │
                                             └─────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌───────────────────┐
                                             │ Anomaly Detection │
                                             │ Arithmetic & Dates│
                                             └─────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌───────────────────┐
                                             │ Document Validator│
                                             │ (Invoice/Resume)  │
                                             └─────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌───────────────────┐
                                             │  Workflow Engine  │
                                             │ (Rules Evaluation)│
                                             └─────────┬─────────┘
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           │                                                       │
              [Passed Rules, Clean, Conf >= 70%]                      [Anomalies, Failed Checks, Conf < 70%]
                           │                                                       │
                           ▼                                                       ▼
                 ┌───────────────────┐                                   ┌───────────────────┐
                 │ State: Completed  │                                   │State: Needs Review│
                 │ Straight-Through  │                                   │Human Review Queue │
                 └─────────┬─────────┘                                   └─────────┬─────────┘
                           │                                                       │
                           │                              ┌────────────────────────┴────────────────────────┐
                           │                              │                                                 │
                           │                       [Reviewer Approve]                              [Reviewer Reject]
                           │                              │                                                 │
                           │                              ▼                                                 ▼
                           │                    ┌───────────────────┐                             ┌───────────────────┐
                           │                    │  State: Approved  │                             │  State: Rejected  │
                           │                    │         │         │                             │ (Mandatory Note)  │
                           │                    │         ▼         │                             └───────────────────┘
                           │                    │ State: Completed  │
                           │                    └─────────┬─────────┘
                           │                              │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │  RAG Knowledge Base     │
                             │  Semantic Chunks Index  │
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │  SQLite Audit Logging   │
                             │ (audit_logs table)      │
                             └─────────────────────────┘
```

---

## ⚙️ Project Structure & Bridge Modules

```
ai-document-intelligence/
├── app.py                      # Streamlit application UI & navigation router
├── config.py                   # Centralized configuration & environment settings
├── .env.example                # Environment variable documentation
├── database.py                 # Section 13 bridge: SQLite database manager
├── storage.py                  # Section 13 bridge: File storage manager
├── processor.py                # Section 13 bridge: Document intake processor
├── validator.py                # Section 13 bridge: Document validator
├── workflow.py                 # Section 13 bridge: Rule-based workflow engine
├── audit.py                    # Section 13 bridge: Audit logging service
├── batch.py                    # Section 13 bridge: Fault-tolerant batch processor
├── anomaly.py                  # Section 13 bridge: Anomaly detection engine
├── rag.py                      # Section 13 bridge: Grounded RAG assistant
├── requirements.txt            # Production dependencies
├── database/
│   ├── __init__.py
│   └── db.py                   # SQLite DatabaseManager with WAL mode & migrations
├── models/
│   ├── __init__.py
│   └── document.py             # DocumentRecord dataclass with Week 6 fields
├── services/
│   ├── __init__.py
│   ├── anomaly.py              # AnomalyDetector: Arithmetic & pattern integrity
│   ├── audit.py                # AuditService: Event recording & chronological history
│   ├── batch.py                # BatchProcessor: Isolated per-document error boundary
│   ├── classifier.py           # ML classifier service wrapper
│   ├── document_processor.py   # Full pipeline intake, routing & audit orchestrator
│   ├── file_storage.py         # FileStorageManager: SHA-256 partitioned vault
│   ├── hashing.py              # SHA-256 cryptographic digest calculation
│   ├── ocr.py                  # OCR computer vision service
│   ├── rag.py                  # RAGService: High-level vector search & QA coordinator
│   ├── validation.py           # FileValidator & DocumentValidator (Invoices/Resumes)
│   └── workflow.py             # WorkflowEngine: State transitions & rule routing
├── src/
│   ├── classifier.py           # TF-IDF Vectorizer + LR, SVM, NB, Rule Baseline
│   ├── document_reader.py      # PyMuPDF direct PDF text extractor
│   ├── evaluator.py            # Precision, Recall, F1, Accuracy benchmarks
│   ├── extractor.py            # Regex & pattern entity extraction
│   ├── ocr_processor.py        # Tesseract OCR engine with fallback
│   ├── rag_engine.py           # Grounded RAG engine, semantic chunker & guardrails
│   ├── text_cleaner.py         # Whitespace, unicode & artifact normalization
│   ├── train_models.py         # Model training script
│   └── utils.py                # Sample generator (14 authentic PDFs) & dataset loader
├── data/
│   ├── documents.db            # SQLite persistent repository (WAL mode enabled)
│   ├── train/                  # Labeled training dataset (22 domain text samples)
│   └── test/                   # Unseen testing dataset (9 test evaluation samples)
├── samples/                    # 14 authentic multi-category test PDFs
└── tests/
    ├── conftest.py             # Pytest fixtures
    ├── test_anomaly.py         # Arithmetic & financial consistency unit tests (7 tests)
    ├── test_audit.py           # Audit logging & timeline integrity tests (2 tests)
    ├── test_batch.py           # Batch processing & fault isolation tests (2 tests)
    ├── test_classifier.py      # Model training & prediction unit tests (5 tests)
    ├── test_database.py        # SQLite migrations, CRUD & indexing tests (4 tests)
    ├── test_extraction.py      # Field extraction & missing-field tests (6 tests)
    ├── test_pipeline_e2e.py    # Pipeline end-to-end integration tests (5 tests)
    ├── test_processor_e2e.py   # DocumentProcessor workflow tests (2 tests)
    ├── test_rag.py             # Grounded RAG & anti-hallucination tests (6 tests)
    ├── test_storage.py         # SHA-256 & storage lifecycle tests (4 tests)
    ├── test_text_cleaning.py   # Text cleaner normalization tests (4 tests)
    ├── test_validation.py      # DocumentValidator format & currency tests (9 tests)
    ├── test_week4_acceptance.py# Week 4 backward compatibility test matrix (13 tests)
    ├── test_week5_acceptance.py# Week 5 15-document workflow acceptance matrix (18 tests)
    ├── test_week6_acceptance.py# Week 6 mandatory production acceptance test matrix (14 tests)
    └── test_workflow.py        # State transitions & reviewer action tests (4 tests)
```

---

## 🧪 Comprehensive Test Suite (105 / 105 Passed — 100%)

All 105 unit, integration, and acceptance tests pass across 16 test files:

```powershell
pytest -v
```

```
====================== 105 passed, 65 warnings in 10.98s ======================
```

### Production Acceptance Scenarios (Section 6 Verified):
| Scenario | Test Name | Result |
| :--- | :--- | :--- |
| **01. Clean Digital PDF** | `test_scenario_01_clean_digital_pdf` | **PASSED** |
| **02. Scanned Document / OCR** | `test_scenario_02_scanned_document_ocr` | **PASSED** |
| **03. Supported Document Types** | `test_scenario_03_supported_document_types` | **PASSED** |
| **04. Missing Information** | `test_scenario_04_missing_information` | **PASSED** |
| **05. Duplicate Document** | `test_scenario_05_duplicate_document` | **PASSED** |
| **06. Unsupported File Type** | `test_scenario_06_unsupported_file` | **PASSED** |
| **07. Corrupted Document** | `test_scenario_07_corrupted_document` | **PASSED** |
| **08. Low Classifier Confidence** | `test_scenario_08_low_confidence` | **PASSED** |
| **09. Financial Arithmetic Mismatch** | `test_scenario_09_financial_validation_arithmetic_mismatch` | **PASSED** |
| **10. Invalid Workflow Transition** | `test_scenario_10_invalid_workflow_transition` | **PASSED** |
| **11. Mixed Batch Processing** | `test_scenario_11_mixed_batch_processing` | **PASSED** |
| **12. Restart Persistence** | `test_scenario_12_restart_persistence` | **PASSED** |
| **13. Grounded AI Assistant (RAG)** | `test_scenario_13_document_ai_assistant_grounded_qa` | **PASSED** |
| **14. Insufficient Info Guardrail** | `test_scenario_14_insufficient_information_guardrail` | **PASSED** |

---

## 🚀 Quickstart & Execution

```powershell
# 1. Clone repository
git clone https://github.com/evilswordboy-bot/ai-document-intelligence.git
cd ai-document-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full test suite
pytest -v

# 4. Launch Streamlit application
streamlit run app.py
```

---

## 👨‍💻 Author
**Sakthibalan S**  
*AI/ML Intern — ZYROO AI/ML Internship*  
GitHub: [@evilswordboy-bot](https://github.com/evilswordboy-bot)
