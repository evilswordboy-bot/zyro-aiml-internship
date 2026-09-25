# ⚡ AI & Machine Learning Engineering Portfolio

> **Production-Grade AI/ML Projects & Document Intelligence Platform**  
> **Author / Developer:** Sakthibalan S (`evilswordboy-bot`)

[![Live Demo (Cloudflare)](https://img.shields.io/badge/Live%20Demo-Cloudflare%20Edge-orange.svg?style=for-the-badge&logo=cloudflare)](https://angels-scanned-profit-midnight.trycloudflare.com)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Deployed-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue.svg?style=for-the-badge&logo=github)](https://github.com/evilswordboy-bot/zyro-aiml-internship)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)]()

**🌐 Public Live URL (Cloudflare Edge):** **[https://angels-scanned-profit-midnight.trycloudflare.com](https://angels-scanned-profit-midnight.trycloudflare.com)**  
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
├── week-04/                     # Week 4: Persistent AI Document Platform (Active Milestone)
│   ├── app.py                   # Modern SaaS Multi-Tab Interface
│   ├── database/                # SQLite connection manager, WAL mode, CRUD & search queries
│   │   └── db.py
│   ├── services/                # Hashing, file storage, validation, pipeline orchestrator
│   │   ├── hashing.py           # SHA-256 cryptographic digest computation
│   │   ├── file_storage.py      # Categorized folder vault & path traversal guard
│   │   ├── validation.py        # File type whitelist & upload size limit (<20MB)
│   │   └── document_processor.py# Full pipeline controller
│   ├── models/                  # DocumentRecord dataclass & status constants
│   ├── storage/                 # Physical file vault (invoices/, resumes/, other/)
│   ├── data/                    # Persistent SQLite database (documents.db)
│   ├── samples/                 # Verified test documents
│   └── tests/                   # 43 automated pytest unit & acceptance tests
│
├── app.py                       # Root Streamlit router (direct deployment to Streamlit Cloud)
├── requirements.txt             # Root deployment requirements
├── run_offline.bat              # Root 1-click offline launcher
├── .gitignore                   # Standard Python/IDE exclusions
└── README.md                    # Root overview (this document)
```

---

## 🌟 Week 04: AI Document Intelligence & Workflow Platform (Current Active Version)

The **Week 4** platform elevates document understanding into an industrial, persistent document management and audit platform:

### Key Enhancements in Week 04:
- **Persistent SQLite Document Repository (`data/documents.db`)**: Complete metadata indexing, WAL mode for concurrent access, and parameterized queries.
- **Cryptographic SHA-256 Deduplication**: Computes raw byte digests to identify and suppress redundant file writes and database entries.
- **Categorized Physical Vault (`storage/`)**: Segregated document storage (`invoices/`, `resumes/`, `other/`) with unique sanitized naming and strict path traversal defense.
- **Fast Multi-Field Search**: Parameterized search across Original Filename, Company, Invoice Number, Candidate Name, Category, and Text Preview.
- **Dynamic Filters & Sorting**: Instant filtering by Category, Status (`Processed`, `Needs Review`, `Failed`), and chronological ordering.
- **Detailed Document Inspection & Vault Download**: Deep inspection modal showing cryptographic hash, extracted entities, and one-click file download.
- **Preserved Week 3 AI Core**: Full TF-IDF + Logistic Regression, Linear SVM, Naive Bayes, image-preprocessed OCR fallback, and regex entity extraction.
- **43 Automated Tests**: 100% test pass rate across database CRUD, storage security, duplicate detection, and acceptance test matrices.

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
python -m pytest week-04/tests/ -v
```
*Expected: **43 passed** in ~6.2s*

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
