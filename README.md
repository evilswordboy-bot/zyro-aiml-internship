# ⚡ AI & Machine Learning Engineering Portfolio

> **Production-Grade AI/ML Projects & Document Intelligence Platform**  
> **Author / Developer:** Sakthibalan S (`evilswordboy-bot`)

[![Live Demo (Cloudflare)](https://img.shields.io/badge/Live%20Demo-Cloudflare%20Edge-orange.svg?style=for-the-badge&logo=cloudflare)](https://incorporated-attempted-lucy-trusts.trycloudflare.com)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Deployed-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue.svg?style=for-the-badge&logo=github)](https://github.com/evilswordboy-bot/zyro-aiml-internship)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-78%20passed-brightgreen.svg)]()

**🌐 Public Live URL (Cloudflare Edge):** **[https://incorporated-attempted-lucy-trusts.trycloudflare.com](https://incorporated-attempted-lucy-trusts.trycloudflare.com)**  
**🌐 Public Live URL (Streamlit Cloud):** **[https://mgb56neehcvkyzsiukg2qh.streamlit.app/](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)**

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
├── week-05/                     # Week 5: Advanced Document Workflow & Automation (Active Milestone)
│   ├── app.py                   # Enterprise 9-View Workflow Platform Dashboard
│   ├── database.py              # Top-level bridge to DatabaseManager & schema migrations
│   ├── storage.py               # Top-level bridge to FileStorageManager
│   ├── processor.py             # Top-level bridge to DocumentProcessor
│   ├── validator.py             # Advanced Document Validation Engine & currency parser
│   ├── workflow.py              # Finite State Machine & Rule-Based Routing Engine
│   ├── audit.py                 # Immutable Audit Logging Service & chronological timelines
│   ├── batch.py                 # Fault-Tolerant Batch Processor with isolated error boundaries
│   ├── services/                # Modular engine services (validation, workflow, audit, batch)
│   ├── database/                # SQLite migrations, WAL mode, audit_logs table, indices
│   ├── models/                  # DocumentRecord dataclass & Week 5 controlled statuses
│   ├── storage/                 # Segregated physical vault
│   ├── samples/                 # 12 realistic edge-case sample PDFs
│   └── tests/                   # 78 comprehensive pytest unit & acceptance tests
│
├── app.py                       # Root Streamlit router (routes to week-05 primary)
├── requirements.txt             # Root deployment requirements
├── run_offline.bat              # Root 1-click offline launcher
├── .gitignore                   # Standard Python/IDE exclusions
└── README.md                    # Root overview (this document)
```

---

## 🌟 Week 05: Advanced Document Workflow & Automation (Current Active Version)

The **Week 5** platform transforms document management into an automated, auditable, human-in-the-loop document lifecycle pipeline:

### 🔄 End-to-End Document Workflow
```
UPLOAD ──▶ PROCESS ──▶ CLASSIFY ──▶ EXTRACT ──▶ VALIDATE ──▶ APPLY RULES ──▶ REVIEW / APPROVE / REJECT ──▶ COMPLETE ──▶ AUDIT LOG
```

### Key Capabilities in Week 05:
- **Controlled Finite State Machine**: Enforces valid status transitions across `New`, `Processing`, `Needs Review`, `Approved`, `Rejected`, `Completed`, `Failed`, strictly rejecting invalid state jumps.
- **Advanced Document Validation Engine (`validator.py`)**: Multi-currency amount parsing (`$ 1,450.00`, `Rs. 78,500`, `€ 999.00`), invoice number verification, RFC-compliant email regex, 10–15 digit phone check, and skills detection.
- **Rule-Based Workflow Engine (`workflow.py`)**: Automatic straight-through routing to `Approved`/`Completed` for high-confidence, perfectly validated files; flags missing/invalid fields and low confidence into `Needs Review`.
- **Human Review Queue**: Dedicated interactive triage workspace for reviewers to inspect documents, view structured validation failure badges, approve directly, or **reject with a mandatory explanation note**.
- **Fault-Tolerant Batch Processing (`batch.py`)**: Ingests multiple heterogeneous files concurrently with isolated try/except error boundaries so corrupted files never halt or crash batch ingestion.
- **Immutable Audit Logging System (`audit.py`)**: Tracks every automated transition, confidence score, validation failure, reviewer approval/rejection note, and timestamp in a queryable `audit_logs` SQLite table.
- **78 Automated Tests**: 100% test pass rate across unit tests, service tests, integration tests, and the Week 5 acceptance test matrix.

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
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest week-05/tests/ -v
```
*Expected: **78 passed** in ~9.8s*

### 4. Run the Web Application
```bash
streamlit run app.py
```
Or double-click **`run_offline.bat`** on Windows. Open **`http://localhost:8501`** in your browser.

---

## 🌐 Public Deployment on Streamlit Community Cloud

This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:

1. Log in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account (`evilswordboy-bot`).
2. Select repository: `evilswordboy-bot/zyro-aiml-internship`.
3. Branch: `main` | Main file path: `app.py`.
4. Click **"Deploy!"**.

---

## 👨‍💻 Author & Project Details

- **Developer:** Sakthibalan S ([GitHub: evilswordboy-bot](https://github.com/evilswordboy-bot))
- **Track:** Applied AI/ML & Document Intelligence Engineering
