# AI Document Intelligence & Workflow Platform

**Enterprise-Grade Document Understanding, Multi-Model Classification & Entity Extraction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.24+-green.svg)](https://pymupdf.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

**AI Document Intelligence & Workflow Platform** is an end-to-end document processing and understanding system designed to eliminate manual data entry from common enterprise and recruitment documents. 

It accepts **PDFs** and **Images (JPG, JPEG, PNG)**, extracts text via **PyMuPDF** with image-preprocessed **Tesseract OCR fallback**, normalizes unstructured text, classifies document type (**Invoice**, **Resume**, **Other**) using multiple evaluated machine learning models against a deterministic baseline, and extracts structured domain entities with explicit `"Not Found"` missing-field handling and calibrated confidence scoring.

---

## 🏛️ Pipeline Architecture

```
                               ┌────────────────────────┐
                               │  User Document Upload  │
                               │  (PDF, JPG, JPEG, PNG) │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │  Document Ingestion &  │
                               │    Validation Layer    │
                               └───────────┬────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
           [Digital / Vector PDF]                       [Scanned PDF / Image]
                     │                                           │
                     ▼                                           ▼
         ┌───────────────────────┐                  ┌────────────────────────┐
         │ Native PyMuPDF Fitz   │                  │ OpenCV / PIL Adaptive  │
         │ Selectable Text Parse │                  │ Image Preprocessing    │
         └───────────┬───────────┘                  └────────────┬───────────┘
                     │                                           │
                     │                                           ▼
                     │                              ┌────────────────────────┐
                     │                              │  Tesseract OCR Engine  │
                     │                              └────────────┬───────────┘
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Text Normalization &   │
                               │ Cleaning Layer (Regex) │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │  TF-IDF Feature Space  │
                               │ (1-2 N-Grams, Sublinear│
                               └───────────┬────────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     │                     │                     │
                     ▼                     ▼                     ▼
          ┌────────────────────┐ ┌───────────────────┐ ┌───────────────────┐
          │Logistic Regression │ │    Linear SVM     │ │    Multinomial    │
          │    (Calibrated)    │ │   (Max Margin)    │ │    Naive Bayes    │
          └──────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
                     │                     │                     │
                     └─────────────────────┼─────────────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Rule-Based Comparator  │
                               │  (Keyword Baseline)    │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │   Domain Entity &      │
                               │   Field Extraction     │
                               │ (Invoice / Resume RegEx│
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Missing Field Auditing │
                               │  ("Not Found" Default) │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Interactive Dashboard  │
                               │  & JSON / CSV Export   │
                               └────────────────────────┘
```

---

## 🚀 Key Features

1. **Intelligent Text Acquisition & OCR Fallback**:
   - Fast native digital PDF parsing using PyMuPDF (`fitz`).
   - Image contrast enhancement, grayscale conversion, and noise filtering before invoking Tesseract OCR for scanned documents.
2. **Text Normalization Engine**:
   - Strips non-printable ASCII and control characters.
   - Normalizes unicode quotation marks, dashes, and currency symbols.
   - Collapses excessive whitespace and blank lines while preserving document layout cues.
3. **TF-IDF Feature Representation**:
   - Sublinear term-frequency scaling with unigram and bigram tokenization (`ngram_range=(1, 2)`).
   - Frequency filtering (`min_df=1`, `max_df=0.95`) with English stop-word filtering.
4. **Multi-Model Document Classification**:
   - Evaluates **Logistic Regression**, **Linear SVM (`LinearSVC`)**, and **Multinomial Naive Bayes** against a deterministic **Rule-Based Baseline**.
   - Model switcher in the UI allows instant side-by-side comparison on live uploaded files.
5. **Truthful Confidence Reporting**:
   - Probability distributions reported for Logistic Regression and Naive Bayes.
   - Clear and honest `"Confidence: Not Available"` badge for uncalibrated models (Linear SVM, Rule-Based), adhering to zero-hallucination standards.
6. **Structured Entity Extraction with Missing Field Guardrails**:
   - **Invoices**: Invoice Number, Invoice Date, Company / Vendor Name, Total Amount.
   - **Resumes**: Candidate Name, Email Address, Phone Number, Technical Skills.
   - Every field defaults to `"Not Found"` if missing or unconfident, avoiding phantom data.
7. **Comprehensive Model Evaluation Suite**:
   - Integrated testing suite with accuracy, macro precision, macro recall, macro F1-score, and $3 \times 3$ confusion matrices.
8. **Interactive UI & 1-Click Exports**:
   - Built with Streamlit featuring a multi-stage status indicator, side-by-side text cleaning inspector, and instant JSON / CSV downloads.

---

## 📊 Dataset & Model Benchmarks

### Dataset Distribution
The platform was evaluated on a verified corpus of business invoices, candidate resumes, and non-target administrative documents:
- **Training Set (`data/train/`)**: 15 labeled documents (5 Invoices, 5 Resumes, 5 Other)
- **Test Set (`data/test/`)**: 9 independent evaluation documents (3 Invoices, 3 Resumes, 3 Other)

### Experimental Benchmark Results

| Classifier Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Inference Latency | Confidence Output |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based Baseline** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | < 1 ms | Deterministic |
| **Logistic Regression** | **0.8889** | **0.9167** | **0.8889** | **0.8857** | ~ 4 ms | Calibrated Probabilities |
| **Linear SVM (`LinearSVC`)** | **0.8889** | **0.9167** | **0.8889** | **0.8857** | ~ 2 ms | Margin (No Probabilities) |
| **Multinomial Naive Bayes** | **0.8889** | **0.9167** | **0.8889** | **0.8857** | ~ 3 ms | Posterior Probabilities |

> *Evaluation evaluated on 9 unseen test files using Scikit-Learn metrics.*

### Test Set Confusion Matrix ($3 \times 3$)

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
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Production Python package dependencies
├── packages.txt                # Linux dependencies (Tesseract OCR for cloud)
├── run_offline.bat             # 1-Click offline local launcher for Windows
├── README.md                   # Complete architectural and technical documentation
├── data/
│   ├── train/                  # 15 labeled training files across classes
│   │   ├── invoices/
│   │   ├── resumes/
│   │   └── other/
│   └── test/                   # 9 independent test files across classes
│       ├── invoices/
│       ├── resumes/
│       └── other/
├── models/
│   ├── classifier.pkl          # Serialized trained TF-IDF vectorizer + ML models
│   └── metrics.json            # Serialized evaluation metrics & confusion matrices
├── notebooks/
│   └── model_comparison.ipynb  # Interactive Jupyter notebook for training & evaluation
├── samples/                    # Pre-generated PDF test files for instant live demo
│   ├── sample_invoice.pdf
│   ├── sample_resume.pdf
│   └── sample_contract.pdf
├── src/
│   ├── __init__.py
│   ├── document_reader.py      # PyMuPDF ingestion and selectable text extractor
│   ├── ocr_processor.py        # Image preprocessing and Tesseract OCR fallback
│   ├── text_cleaner.py         # Whitespace, unicode, and control text normalizer
│   ├── classifier.py           # TF-IDF vectorizer & multi-model classifier suite
│   ├── extractor.py            # Regex domain entity extractors with missing field logic
│   ├── evaluator.py            # Classification metrics & confusion matrix computer
│   └── utils.py                # Dataset loaders, sample PDF generators, and serializers
└── tests/
    ├── test_text_cleaning.py   # Unit tests for text cleaning and sanitization
    ├── test_classifier.py      # Unit tests for model training, prediction, & confidence
    ├── test_extraction.py      # Unit tests for field extraction and "Not Found" handling
    └── test_pipeline_e2e.py    # End-to-end integration tests on sample PDFs
```

---

## 🛠️ Installation & Quickstart

### Prerequisites
- **Python 3.10** or higher
- Optional: [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) (only required if processing scanned non-searchable PDFs or raw images)

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/evilswordboy-bot/ai-document-intelligence.git
cd ai-document-intelligence

# Create and activate a virtual environment
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

### 3. Run Automated Unit & E2E Tests
```bash
python -m pytest tests/ -v
```
*Expected Output: `20 passed in ~3.5s`*

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Or on Windows, simply double-click **`run_offline.bat`** to start offline without any command typing.

---

## 🧪 Verification & Demonstration Walkthrough

When evaluating or demonstrating the platform:

1. **Load Pre-built Samples**: Use the sidebar **"Quick Load Sample Document"** selector to test `sample_invoice.pdf`, `sample_resume.pdf`, or `sample_contract.pdf`.
2. **Review the 6 Pipeline Stages**: Verify that each stage dynamically executes and checks off:
   - `✓ Document Uploaded`
   - `✓ Text Extracted`
   - `✓ Text Cleaned`
   - `✓ Document Classified`
   - `✓ Fields Extracted`
   - `✓ Missing Fields Checked`
3. **Compare Models Live**: Switch the model selector between **Logistic Regression**, **Linear SVM**, **Naive Bayes**, and **Rule-Based Baseline** to view classification predictions in real time.
4. **Inspect Text Cleaning**: Open the **"Text Cleaning & Preprocessing"** tab to compare `Original Extracted Text` against `Normalized Clean Text`.
5. **Inspect Missing Field Truthfulness**: Review the extracted entities table. Any field not present in the document cleanly displays `"Not Found"` in muted styling rather than hallucinated estimates.
6. **Export Output**: Download the extracted structured records in structured JSON or CSV format with one click.

---

## 🌐 Cloud Deployment (Streamlit Community Cloud)

1. Fork or push this repository to your GitHub account.
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select repository: `ai-document-intelligence`.
4. Set Main file path to `app.py`.
5. Click **Deploy**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
