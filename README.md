# ⚡ AI & Machine Learning Engineering Portfolio

> **Production-Grade AI/ML Projects & Document Intelligence Platform**  
> **Author / Developer:** Sakthibalan S (`evilswordboy-bot`)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Cloudflare%20Edge-orange.svg?style=for-the-badge&logo=cloudflare)](https://spaces-flyer-sticks-surgeon.trycloudflare.com)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue.svg?style=for-the-badge&logo=github)](https://github.com/evilswordboy-bot/zyro-aiml-internship)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=for-the-badge&logo=python)](https://www.python.org/)

**🌐 Direct Live Public URL:** https://mgb56neehcvkyzsiukg2qh.streamlit.app/

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
│   ├── app.py                   # Masterpiece Streamlit Dashboard
│   ├── requirements.txt         # Pinned dependencies
│   ├── README.md                # Dedicated Week 2 architecture guide
│   ├── .env.example             # Environment template
│   ├── src/                     # Core engine (validation, extraction, classification, parsing)
│   ├── samples/                 # Authentic sample PDFs (Invoice 1, Invoice 2, Resume, Other)
│   └── tests/                   # 25 automated pytest unit & integration tests
│
├── app.py                       # Root Streamlit router (direct deployment to Streamlit Cloud)
├── requirements.txt             # Root deployment requirements
├── .gitignore                   # Standard Python/IDE exclusions
└── README.md                    # Root overview (this document)
```

---

## 🌟 Week 02: AI Document Intelligence Platform (Current Task)

The **Week 2** application is a production-grade **AI Document Intelligence & Workflow Platform** that ingests unstructured PDFs and images, validates file integrity, extracts text via PyMuPDF (with OCR fallback), classifies document types into **Invoice**, **Resume**, or **Other** using a hybrid Rule-Based + Scikit-Learn TF-IDF Logistic Regression pipeline, extracts key business entities, and renders explainable results in an interactive SaaS Streamlit dashboard.

### Key Capabilities:
- **Fast Text Ingestion**: Sub-15ms digital PDF extraction via PyMuPDF (`pymupdf`).
- **Hybrid Classifier**: Combines deterministic keyword heuristics with a calibrated Machine Learning model (`TfidfVectorizer` + `LogisticRegression`).
- **Explainable AI (XAI)**: Feature importance attribution charts and class probability distributions via Plotly.
- **Multi-Currency Entity Parsing**: Extracts Invoice Number, Date, Company, and Total Amount (PKR, USD, EUR, INR).
- **Categorized Skills Taxonomy**: Extracts and groups resume skills into domain pills (AI/ML, Languages, Web, Cloud, Databases).
- **1-Click Sample Showcase**: Pre-loaded with authentic sample documents (`ABC Technologies`, `PKR 125,000`, `INV-1024`, `02-09-2026`).
- **Data Export**: Immediate download of structured results in JSON or CSV format.
- **Automated Tests**: 25 automated unit and end-to-end tests passing.

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

### 3. Run the Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 4. Run Automated Tests
```bash
pytest week-02/tests/ -v
```

---

## 🌐 Public Deployment on Streamlit Community Cloud

This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:

1. Log in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account (`evilswordboy-bot`).
2. Click **"Create app"** (or **"New app"**).
3. Fill in the deployment parameters:
   - **Repository:** `evilswordboy-bot/zyro-aiml-internship`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**.
5. Your public shareable link will be generated (e.g. `https://zyro-aiml-internship-evilswordboy-bot.streamlit.app`).

---

## 📋 Week 01: Onboarding & Environment Setup (Completed)

- Configured Python 3.13, Git 2.55, and virtual environment.
- Verified AI/ML libraries: NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn.
- Executed Logistic Regression baseline on the Iris dataset.
- Verified end-to-end environment readiness.

---

## 👨‍💻 Author & Project Details

- **Developer:** Sakthibalan S ([GitHub: evilswordboy-bot](https://github.com/evilswordboy-bot))
- **Track:** Applied AI/ML & Full-Stack Intelligence Engineering
- **Focus:** Document Automation, NLP, and Explainable AI Systems
