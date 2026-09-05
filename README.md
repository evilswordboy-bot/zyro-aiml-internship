Zyroo AI/ML Internship

Week 01 — Onboarding & Environment Setup

This repository contains my Week 1 work for the Zyroo AI/ML Internship.

Objective

The goal of Week 1 is to:

Prepare the AI/ML development environment

Configure Python and Git

Create and use a Python virtual environment

Install the required AI/ML libraries

Run a basic machine learning program

Configure GitHub

Maintain Week 1 work in a GitHub repository

Development Environment

Python: 3.13.15

Git: 2.55.0.windows.3

Operating System: Windows

Virtual Environment: Python venv

Libraries

The following AI/ML libraries were installed:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Jupyter

See requirements.txt for the project dependencies.

Machine Learning Test

The Week 1 ML test uses the Iris dataset with a Logistic Regression model.

The program verifies:

Python environment

AI/ML library imports

Iris dataset loading

Dataset conversion using Pandas

Train/test data splitting

Machine learning model training

Prediction

Model accuracy

Result

AI/ML Environment Test: PASSED

Model Accuracy: 96.67%

Project Structure

zyro-aiml-internship/
│
├── week-01/
│   └── environment_test.py
│
├── .gitignore
├── README.md
└── requirements.txt

How to Run

1. Activate the virtual environment

Windows CMD:

.venv\Scripts\activate

2. Run the Week 1 ML test

python week-01\environment_test.py

Week 1 Status

Python installed and verified

Git installed and verified

Virtual environment created

AI/ML libraries installed

Basic ML program executed successfully

requirements.txt created

.gitignore created

GitHub evidence and submission completed

Internship

Organization: Zyroo
Program: AI/ML Internship
Task: Week 01 — Onboarding & Environment Setup

This repository tracks my weekly assignments and projects for the **Zyroo AI/ML Internship**.

* **Week 01**: Onboarding & AI/ML Development Environment Setup (Completed)
* **Week 02**: AI Document Intelligence MVP (Completed)

---

# Week 02 — AI Document Intelligence MVP

> An intelligent, end-to-end Document Processing MVP designed to ingest documents, extract raw text, classify document categories (Invoice, Resume, Other), extract high-value metadata, and provide instant structured insights.

---

## Overview

**AI Document Intelligence** is an automated document processing system built with Python and Streamlit. It enables users to upload documents in PDF and image formats (PNG, JPG, JPEG), inspect extracted text, automatically determine the document category, and extract key fields without relying on complex, costly third-party SaaS APIs.

---

## Problem

In enterprise and operational workflows, manual document processing is tedious, error-prone, and repetitive. Teams spend countless hours manually extracting invoice numbers, billing totals, candidate contact info, and skill sets from diverse document formats.

---

## Solution

This MVP provides an automated, lightweight, and deterministic pipeline:
1. Accepts PDFs and images.
2. Validates file integrity and format.
3. Extracts selectable text with high-speed engine **PyMuPDF**.
4. Automatically falls back to **Tesseract OCR** for scanned documents or image inputs.
5. Classifies the document as an **Invoice**, **Resume**, or **Other** using rule-based keyword frequency analysis (with an optional TF-IDF + Logistic Regression machine learning model).
6. Extracts structured entity fields using targeted regular expressions and layout heuristics.
7. Displays immediate results on a modern dashboard with one-click JSON export.

---

## Features

- **Multi-Format Upload**: Native support for `.pdf`, `.jpg`, `.jpeg`, and `.png`.
- **Selectable PDF Extraction**: Direct, lightning-fast text extraction using PyMuPDF (`fitz`).
- **Scanned Document OCR Fallback**: Integrated OCR pipeline using Pillow and `pytesseract` with graceful non-crashing fallbacks.
- **Dual Classification Engine**:
  - *Primary (MVP)*: Rule-based keyword frequency scoring.
  - *Optional Enhancement*: Scikit-learn TF-IDF + Logistic Regression classifier.
- **Invoice Field Extraction**: Extracts `Invoice Number`, `Date`, `Company Name`, and `Total Amount`.
- **Resume Field Extraction**: Extracts `Candidate Name`, `Email Address`, `Phone Number`, and `Skills`.
- **Strict Hallucination Prevention**: Explicitly reports `"Not found"` if a field is absent rather than inventing values.
- **Interactive Streamlit Interface**: Clean, professional layout with live system status badges and responsive metrics.
- **Raw Text Viewer**: Collapsible inspection area for verifying full extracted document text.
- **Exportable Results**: Instant one-click structured JSON download.
- **Robust Error Handling**: Never crashes on unsupported file types, empty files, or missing OCR binaries.

---

## Technology Stack

- **Core Language**: Python 3.10+
- **Web Application**: Streamlit
- **PDF Extraction**: PyMuPDF (`pymupdf`)
- **Image Processing & OCR**: Pillow, `pytesseract` (Tesseract OCR)
- **Pattern Matching**: Regular Expressions (`re`)
- **Machine Learning (Optional)**: Scikit-learn (`TfidfVectorizer`, `LogisticRegression`)
- **Data Handling**: Pandas
- **Version Control**: Git & GitHub

---

## Architecture

```
                       USER
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Upload PDF / JPG / PNG       │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │         File Validation        │
        │   (Format check & Size check)  │
        └────────────────┬───────────────┘
                         │
                         ▼
               Is document PDF?
             ┌───────────┴───────────┐
            YES                      NO (Image)
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │ PyMuPDF Extract │     │   Pillow + OCR  │
    └────────┬────────┘     └────────┬────────┘
             │                       │
     Has text?                       │
     ┌───────┴───────┐               │
    YES              NO              │
     │                └──────────────┤
     ▼                               ▼
┌──────────────┐            ┌─────────────────┐
│ Extracted    │            │  OCR Fallback   │
│ Selectable   │            │  (Render Page → │
│ Text         │            │   pytesseract)  │
└──────┬───────┘            └────────┬────────┘
       │                             │
       └──────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │   Document Classification  │
        │  (Rule-Based Keyword / ML) │
        └─────────────┬──────────────┘
                      │
             ┌────────┴────────┐
             ▼                 ▼
        ┌─────────┐       ┌─────────┐
        │ Invoice │       │ Resume  │   (or Other)
        └────┬────┘       └────┬────┘
             │                 │
             ▼                 ▼
   ┌─────────────────┐ ┌─────────────────┐
   │ Invoice Fields: │ │ Resume Fields:  │
   │ • Invoice No    │ │ • Name          │
   │ • Date          │ │ • Email         │
   │ • Company Name  │ │ • Phone         │
   │ • Total Amount  │ │ • Skills        │
   └────────┬────────┘ └───────┬─────────┘
            │                  │
            └─────────┬────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │  Streamlit Results Display │
        │  & Downloadable JSON Data  │
        └────────────────────────────┘
```

---

## Installation

Follow these exact steps to set up the project locally on Windows:

### 1. Clone the repository
```cmd
git clone https://github.com/evilswordboy-bot/zyro-aiml-internship.git
cd zyro-aiml-internship
```

### 2. Create and activate a Python virtual environment
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install required dependencies
```cmd
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit web server:
```cmd
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## OCR Setup (Optional for Scanned Documents)

The application handles native PDFs directly using PyMuPDF without any external installations. However, to extract text from **scanned documents or image files (`.jpg`, `.png`)**, Tesseract OCR must be installed on your machine:

1. Download the Windows installer from the official UB Mannheim repository:  
   [Tesseract OCR Windows 64-bit Installer](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and install to the default location (`C:\Program Files\Tesseract-OCR`).
3. Add `C:\Program Files\Tesseract-OCR` to your system `PATH` environment variable, or set `TESSERACT_PATH` in your environment.
4. Restart your terminal or command prompt.

> [!NOTE]
> If Tesseract OCR is not installed, the application **will not crash**. It displays a clear system status indicator in the sidebar and guides the user that OCR is required only for scanned documents.

---

## Testing & Sample Results

The repository includes pre-built realistic test documents in the `samples/` directory:

| Filename | Expected Type | Detected Type | Fields Found | Missing / "Not found" Fields |
| :--- | :--- | :--- | :--- | :--- |
| `samples/invoice_1.pdf` | **Invoice** | **Invoice** | Invoice Number (`INV-2026-1024`), Date (`02/09/2026`), Company Name (`Acme Cloud Solutions Inc.`), Total Amount (`$9,075.00`) | *None* (All 4 found) |
| `samples/invoice_2.pdf` | **Invoice** | **Invoice** | Invoice Number (`APX-99821`), Date (`2026-08-15`), Company Name (`Apex Digital Retailers`), Total Amount (`125,000`) | *None* (All 4 found) |
| `samples/resume_1.pdf` | **Resume** | **Resume** | Name (`Aravind Sharma`), Email (`aravind.sharma@example.com`), Phone (`+91 98765 43210`), Skills (`Python, Streamlit, PyMuPDF, Scikit-Learn, ...`) | *None* (All 4 found) |

---

## Screenshots

### 1. Upload & Landing Screen
![Upload Screen](screenshots/upload_screen.png)
*Displays clean landing interface, system status badges in sidebar, and multi-format file uploader.*

### 2. Invoice Analysis Result
![Invoice Result](screenshots/invoice_result.png)
*Displays classified invoice, detection method, and extracted invoice fields (Number, Date, Company, Total).*

### 3. Resume Analysis Result
![Resume Result](screenshots/resume_result.png)
*Displays classified candidate resume, contact details, and technical skill tags.*

### 4. Extracted Raw Text Inspection
![Extracted Text View](screenshots/extracted_text.png)
*Displays full raw extracted text inside the expandable inspection section with JSON download option.*

---

## Limitations

- **Rule-Based Classification**: Relies on frequency counts of domain keywords. Documents with sparse or unconventional vocabulary may be classified as "Other".
- **Regex Field Extraction**: Designed for common date, phone, invoice, and currency patterns. Highly idiosyncratic formats or multi-page table summaries may require custom heuristics.
- **Scanned Document Quality**: OCR accuracy is dependent on image resolution, skew, and contrast.
- **Complex Multi-Column Layouts**: Single-stream text extraction may interweave adjacent columns if layout bounding boxes are not segmented.

---

## Future Improvements

- **Layout-Aware Parsing**: Implement bounding box analysis (e.g. LayoutLM or Microsoft Document Intelligence) to parse complex table structures.
- **Advanced NLP / LLM Extraction**: Integrate zero-shot extraction using local LLMs (e.g., Ollama / Llama 3) for arbitrary document schemas.
- **Expanded Document Types**: Support Purchase Orders, Bank Statements, ID Cards, and Medical Reports.
- **RESTful API**: Provide FastAPI microservice endpoints alongside the Streamlit UI.
- **Database Persistence**: Store parsed document records and analytics in PostgreSQL / SQLite.

---

## Deployment to Streamlit Community Cloud

This project is prepared for 1-click deployment on Streamlit Community Cloud:

1. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Fill in:
   * **Repository**: `evilswordboy-bot/zyro-aiml-internship`
   * **Branch**: `main`
   * **Main file path**: `app.py`
4. Click **"Deploy"**.

---

## Week 01 Archive — Onboarding & Environment Setup

* **Objective**: Prepare the AI/ML development environment, configure Python & Git, install essential AI/ML libraries, and run a baseline Iris dataset classifier.
* **Test script**: `environment_test.py`
* **Test result**: Passed with 96.67% accuracy.

---

## License

Developed for the **Zyroo AI/ML Internship Week 2 Evaluation**. Free to use for educational and demonstrational purposes.
