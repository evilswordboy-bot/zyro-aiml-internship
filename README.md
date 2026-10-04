# ⚡ AI & Machine Learning Engineering Portfolio

> **Production-Grade AI/ML Projects & Document Intelligence Platform**  
> **Author / Developer:** Sakthibalan S (`evilswordboy-bot`)

[![Live Demo (Cloudflare)](https://img.shields.io/badge/Live%20Demo-Cloudflare%20Edge-orange.svg?style=for-the-badge&logo=cloudflare)](https://incorporated-attempted-lucy-trusts.trycloudflare.com)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Deployed-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue.svg?style=for-the-badge&logo=github)](https://github.com/evilswordboy-bot/zyro-aiml-internship)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-105%20passed-brightgreen.svg)]()

**🌐 Public Live URL (Cloudflare Edge):** **[https://incorporated-attempted-lucy-trusts.trycloudflare.com](https://incorporated-attempted-lucy-trusts.trycloudflare.com)**  
**🌐 Public Live URL (Streamlit Cloud):** **[https://mgb56neehcvkyzsiukg2qh.streamlit.app/](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)**  
**📂 Primary Codebase:** [evilswordboy-bot/ai-document-intelligence](https://github.com/evilswordboy-bot/ai-document-intelligence)  
**📂 Internship Portfolio:** [evilswordboy-bot/zyro-aiml-internship](https://github.com/evilswordboy-bot/zyro-aiml-internship) (Week 6 active)

---

## 📑 Repository Structure

This repository is structured into modular weekly progression modules:

```
zyro-aiml-internship/
│
├── week-01/                     # Week 1: Environment Setup & ML Baseline
│   ├── environment_test.py      # Verified environment & Iris classifier
│   └── screenshots/             # Verification screenshots for Task 1
│
├── week-02/                     # Week 2: AI Document Intelligence Platform MVP
│   ├── app.py                   # Streamlit Dashboard MVP
│   ├── requirements.txt         # Pinned dependencies
│   ├── src/                     # Core extraction and regex engine
│   └── samples/                 # Sample PDFs
│
├── week-03/                     # Week 3: Improved Document Understanding
│   ├── app.py                   # Multi-Model Intelligence Dashboard
│   ├── src/                     # Classifier, OCR, cleaner, evaluator
│   └── data/                    # Labeled training and test datasets
│
├── week-04/                     # Week 4: Persistent AI Document Platform
│   ├── app.py                   # Modern SaaS Multi-Tab Interface
│   ├── database/                # SQLite connection manager, WAL mode, CRUD & search queries
│   ├── services/                # SHA-256 deduplication, file vault, validation, pipeline
│   ├── models/                  # DocumentRecord dataclass & status constants
│   ├── storage/                 # Physical file vault (invoices/, resumes/, other/)
│   └── tests/                   # 43 automated pytest unit & acceptance tests
│
├── week-05/                     # Week 5: Advanced Document Workflow & Automation
│   ├── app.py                   # Enterprise 9-View Workflow Platform Dashboard
│   ├── database.py              # Top-level bridge to DatabaseManager & schema migrations
│   ├── storage.py               # Top-level bridge to FileStorageManager
│   ├── processor.py             # Top-level bridge to DocumentProcessor
│   ├── validator.py             # Advanced Document Validation Engine & currency parser
│   ├── workflow.py              # Finite State Machine & Rule-Based Routing Engine
│   ├── audit.py                 # Immutable Audit Logging Service & chronological timelines
│   ├── batch.py                 # Fault-Tolerant Batch Processor with isolated error boundaries
│   └── tests/                   # 78 comprehensive pytest unit & acceptance tests
│
├── week-06/                     # Week 6: Final Integration, Optimization & Completion (CURRENT ACTIVE)
│   ├── app.py                   # Complete 10-View Enterprise Platform Dashboard
│   ├── anomaly.py               # Financial Consistency & Anomaly Engine Bridge
│   ├── rag.py                   # Grounded RAG & AI Document Assistant Bridge
│   ├── config.py                # Centralized Environment & Security Configuration
│   ├── .env.example             # Documented Configuration Parameters
│   ├── database.py              # SQLite DatabaseManager with Week 6 Schemas & Latency tracking
│   ├── storage.py               # Cryptographic SHA-256 Partitioned File Vault
│   ├── processor.py             # Full-Lifecycle Intake, Profiling & Audit Orchestrator
│   ├── validator.py             # Format & Entity Validation Engine
│   ├── workflow.py              # Finite State Machine & High-Severity Anomaly Router
│   ├── audit.py                 # Immutable Audit Logging System
│   ├── batch.py                 # Fault-Tolerant High-Throughput Batch Processor
│   ├── services/                # Modular Services (anomaly, rag, audit, batch, workflow, storage)
│   ├── src/                     # Machine Learning, OCR, RAG Engine, Cleaners & Evaluators
│   ├── samples/                 # 14 authentic test PDFs (including arithmetic mismatch edge cases)
│   └── tests/                   # 105 automated unit, service, and acceptance tests (100% pass)
│
├── app.py                       # Root Streamlit router (routes to week-06 primary)
├── requirements.txt             # Root deployment requirements
├── run_offline.bat              # Root 1-click offline launcher
├── .gitignore                   # Standard Python/IDE exclusions
└── README.md                    # Root overview (this document)
```

---

## 🏆 Week 06: Final Integration, Optimization & Project Completion (Current Active Version)

The **Week 6** platform represents the final integrated version of the **AI Document Intelligence & Workflow Platform**:

$$\text{Upload} \longrightarrow \text{Validate} \longrightarrow \text{OCR / Text Extraction} \longrightarrow \text{Classify} \longrightarrow \text{Extract} \longrightarrow \text{Store} \longrightarrow \text{Verify} \longrightarrow \text{Anomaly Check} \longrightarrow \text{AI RAG} \longrightarrow \text{Workflow Decision} \longrightarrow \text{Review / Approve / Reject} \longrightarrow \text{Audit} \longrightarrow \text{Search / Report}$$

### Key Capabilities in Week 06:
- **Financial Consistency & Anomaly Engine (`anomaly.py`)**:
  - Automatically verifies invoice arithmetic: $|\text{subtotal} + \text{tax} - \text{total}| \le \$0.05$. Mathematical discrepancies trigger high-severity anomalies routing documents directly to `Needs Review`.
  - Flags suspicious / outlier amounts ($< \$0$ or $> \$1,000,000$).
  - Validates dates against future thresholds ($> 365$ days).
  - Inspects text quality for corrupted OCR artifacts or excessive noise.
- **Grounded RAG & AI Document Assistant (`rag.py`)**:
  - Semantic chunking with configurable overlap.
  - TF-IDF vector space modeling with cosine similarity ranking.
  - Multi-document grounded question answering with exact source citations: `[Filename] (Page X, Chunk Y) | Relevance: Z%`.
  - **Strict Anti-Hallucination Guardrail**: Queries with insufficient grounded context deterministically return:
    > *"The available documents do not contain sufficient information to answer this question."*
- **End-to-End Latency Tracking**:
  - Microsecond-accurate profiling (`processing_time_ms`) logged for all documents and visible on dashboard KPI cards and document detail views.
- **Controlled Finite State Machine & Human Review Queue**:
  - Strict states: `New`, `Processing`, `Needs Review`, `Approved`, `Rejected`, `Completed`.
  - Enforces mandatory reviewer notes on rejection.
- **Security Hardening**:
  - Centralized `config.py` with environment variable loading and sensible defaults.
  - Path traversal defense preventing writes outside the designated storage root.
  - File format whitelisting and friendly sanitized error shields.
- **105 Automated Tests (100% Pass Rate)**:
  - 105 tests across 16 test suites passing in ~11 seconds, including all 14 mandatory Week 6 production acceptance test scenarios.

---

## 🚀 Quickstart & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/evilswordboy-bot/zyro-aiml-internship.git
cd zyro-aiml-internship
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run All Automated Tests
```bash
pytest week-06/tests/ -v
```

### 4. Launch the Platform
```bash
streamlit run app.py
```

---

## 👨‍💻 Author & Intern Information
* **Developer:** Sakthibalan S
* **GitHub Profile:** [@evilswordboy-bot](https://github.com/evilswordboy-bot)
* **Program:** ZYROO AI/ML Internship
* **Final Milestone:** Week 6 — Final Integration, Optimization & Project Completion
