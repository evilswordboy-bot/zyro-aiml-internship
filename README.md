# ⚡ AI & Machine Learning Engineering Portfolio

> **Production-Grade AI/ML Projects & Document Intelligence Platform**  
> **Author / Developer:** Sakthibalan S (`evilswordboy-bot`)

[![Live Demo (Cloudflare)](https://img.shields.io/badge/Live%20Demo-Cloudflare%20Edge-orange.svg?style=for-the-badge&logo=cloudflare)](https://van-indie-prepare-assistant.trycloudflare.com)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Deployed-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://mgb56neehcvkyzsiukg2qh.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue.svg?style=for-the-badge&logo=github)](https://github.com/evilswordboy-bot/zyro-aiml-internship)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()

**🌐 Public Live URL (Cloudflare Edge):** **[https://van-indie-prepare-assistant.trycloudflare.com](https://van-indie-prepare-assistant.trycloudflare.com)**  
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
├── week-03/                     # Week 3: Improved Document Understanding & Multi-Model Platform
│   ├── app.py                   # Streamlit Multi-Model Intelligence Dashboard
│   ├── requirements.txt         # Production dependencies (Scikit-Learn, PyMuPDF, Tesseract)
│   ├── README.md                # Dedicated Week 3 technical & architectural guide
│   ├── run_offline.bat          # 1-Click offline local launcher for Windows
│   ├── src/                     # Document reader, OCR, text cleaner, classifier, extractor, evaluator
│   ├── data/                    # Labeled training (15) and test (9) datasets
│   ├── models/                  # Serialized classifier.pkl and metrics.json
│   ├── notebooks/               # Jupyter notebook for model training, metrics, & confusion matrix
│   ├── samples/                 # Test PDFs (sample invoices, resumes, contracts)
│   └── tests/                   # 20 automated pytest unit & end-to-end tests
│
├── app.py                       # Root Streamlit router (direct deployment to Streamlit Cloud)
├── requirements.txt             # Root deployment requirements
├── run_offline.bat              # Root 1-click offline launcher
├── .gitignore                   # Standard Python/IDE exclusions
└── README.md                    # Root overview (this document)
```

---

## 🌟 Week 03: Improved Document Understanding (Current Active Version)

The **Week 3** platform significantly improves the Week 2 MVP by introducing an industrial-strength document-understanding pipeline:

### Key Enhancements in Week 03:
- **6-Stage Visual Workflow Pipeline**: Real-time status indicators tracking Document Upload, Text Extraction, Text Cleaning, Document Classification, Field Extraction, and Missing Field Verification.
- **Image Preprocessing & OCR Fallback**: Grayscale conversion, adaptive thresholding, and noise filtering before invoking Tesseract OCR for scanned or rasterized files.
- **Robust Text Normalization**: Whitespace normalization, unicode quote/dash standardization, and control character sanitization with side-by-side original vs. cleaned text comparison.
- **Multi-Model Machine Learning Suite**:
  - **Rule-Based Baseline**: Transparent deterministic keyword comparator.
  - **Logistic Regression**: Calibrated probabilistic classifier with sublinear TF-IDF.
  - **Linear SVM (`LinearSVC`)**: High-dimensional maximum margin classifier.
  - **Multinomial Naive Bayes**: Fast probabilistic text categorization.
- **Truthful Confidence & Missing Field Handling**:
  - Displays calibrated probabilities where valid, and honest `"Confidence: Not Available"` for SVM and Rule-Based classifiers.
  - Returns explicit `"Not Found"` values for omitted document fields rather than hallucinating.
- **Model Evaluation & Confusion Matrix**: Includes accuracy, precision, recall, macro F1-score, and $3 \times 3$ confusion matrices computed on an independent 9-document test set.
- **20 Automated Tests**: Comprehensive unit tests covering text cleaning, model persistence, entity extraction, and end-to-end pipeline integrity.

### Benchmark Summary on Test Set

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Rule-Based Baseline** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | **0.8889** | **0.9167** | **0.8889** | **0.8857** |
| **Linear SVM** | **0.8889** | **0.9167** | **0.8889** | **0.8857** |
| **Naive Bayes** | **0.8889** | **0.9167** | **0.8889** | **0.8857** |

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
python -m pytest week-03/tests/ -v
```

### 4. Run the Web Application
```bash
streamlit run app.py
```
Or double-click `run_offline.bat` on Windows. Open **`http://localhost:8501`** in your browser.

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

---

## 👨‍💻 Author & Project Details

- **Developer:** Sakthibalan S ([GitHub: evilswordboy-bot](https://github.com/evilswordboy-bot))
- **Track:** Applied AI/ML & Document Intelligence Engineering
