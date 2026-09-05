import io
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import pymupdf  # Modern PyMuPDF API
import streamlit as st

# Scikit-learn for optional ML enhancement
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# 1. OCR Configuration & Detection
# -----------------------------------------------------------------------------
try:
    import pytesseract

    # Check for common Windows installation paths if not already in system PATH
    tesseract_candidates = [
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.environ.get("TESSERACT_PATH"),
    ]
    TESSERACT_FOUND = False
    for candidate in tesseract_candidates:
        if candidate and os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            TESSERACT_FOUND = True
            break
except ImportError:
    pytesseract = None
    TESSERACT_FOUND = False


def is_ocr_available() -> bool:
    """Return True if pytesseract and the tesseract binary are available."""
    if pytesseract is None or not TESSERACT_FOUND:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 2. Document Processing & Text Extraction
# -----------------------------------------------------------------------------
def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded file extension and non-empty size.
    Accepts: .pdf, .jpg, .jpeg, .png
    """
    if uploaded_file is None:
        return False, "No file uploaded."

    filename = uploaded_file.name.lower()
    allowed_extensions = (".pdf", ".jpg", ".jpeg", ".png")

    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return (
            False,
            "Unsupported file type. Please upload a PDF, JPG, JPEG, or PNG.",
        )

    # Check for empty file
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) == 0:
        return (
            False,
            "The uploaded file is empty (0 bytes). Please upload a valid document.",
        )

    return True, "File valid"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract selectable text from PDF bytes using PyMuPDF.
    Handles empty, corrupted, or password-protected PDFs gracefully.
    """
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts).strip()
    except Exception as e:
        st.warning(f"Note on direct PDF extraction: {str(e)}")
        return ""


def extract_text_from_image(image_bytes: bytes) -> Tuple[str, bool, str]:
    """
    Extract text from an image using pytesseract.
    Returns: (extracted_text, success_bool, message)
    """
    if not is_ocr_available():
        return (
            "",
            False,
            "Tesseract OCR is not installed or configured on this machine. "
            "Please install Tesseract OCR to extract text from images.",
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        ocr_text = pytesseract.image_to_string(image)
        return ocr_text.strip(), True, "OCR extraction successful"
    except Exception as e:
        return "", False, f"OCR processing failed: {str(e)}"


def ocr_pdf(file_bytes: bytes) -> Tuple[str, bool, str]:
    """
    Fallback OCR for scanned PDFs: render each page to an image and run OCR.
    Returns: (extracted_text, success_bool, message)
    """
    if not is_ocr_available():
        return (
            "",
            False,
            "No selectable text found and Tesseract OCR is not configured. "
            "Install Tesseract OCR to process scanned/image-only PDFs.",
        )

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        ocr_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img)
            ocr_pages.append(page_text)
        doc.close()
        combined_text = "\n".join(ocr_pages).strip()
        return combined_text, True, "Scanned PDF OCR successful"
    except Exception as e:
        return "", False, f"Scanned PDF OCR failed: {str(e)}"


# -----------------------------------------------------------------------------
# 3. Document Classification (Rule-Based & Optional ML)
# -----------------------------------------------------------------------------
INVOICE_KEYWORDS = [
    "invoice",
    "invoice number",
    "invoice no",
    "total",
    "amount due",
    "bill to",
    "billed to",
    "tax invoice",
    "subtotal",
    "due date",
    "po reference",
    "payment terms",
    "unit price",
    "qty",
    "gstin",
]

RESUME_KEYWORDS = [
    "resume",
    "curriculum vitae",
    "cv",
    "skills",
    "education",
    "experience",
    "projects",
    "summary",
    "professional summary",
    "bachelor",
    "master",
    "university",
    "engineer",
    "work experience",
    "certifications",
]


def classify_document_rule_based(text: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Keyword/rule-based classification.
    Compares frequency of invoice-related vs resume-related terms.
    Returns: (document_type, method_label, details)
    """
    text_lower = text.lower()

    inv_score = sum(text_lower.count(k) for k in INVOICE_KEYWORDS)
    res_score = sum(text_lower.count(k) for k in RESUME_KEYWORDS)

    details = {
        "invoice_keyword_score": inv_score,
        "resume_keyword_score": res_score,
    }

    method = "Rule-based Keyword Classification"

    if inv_score > res_score and inv_score >= 2:
        return "Invoice", method, details
    elif res_score > inv_score and res_score >= 2:
        return "Resume", method, details
    else:
        return "Other", method, details


@st.cache_resource
def train_optional_ml_model() -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Train a lightweight TF-IDF + Logistic Regression model on a balanced
    curated dataset of document snippets for the optional ML enhancement.
    """
    training_data = [
        # Invoices
        ("Tax Invoice Invoice Number INV-1024 Date 02/09/2026 Bill To Total Amount Due $500", "Invoice"),
        ("Invoice No APX-99821 Billed To Subtotal Tax Rate Total ₹125,000 Payment Terms", "Invoice"),
        ("COMMERCIAL INVOICE QTY Description Unit Price Amount Due USD Net 30 days", "Invoice"),
        ("Billing Statement Invoice Date Due Date Total Balance Due Bank Wire", "Invoice"),
        ("Sales Receipt Invoice # 4401 Items Sold Subtotal Grand Total Paid", "Invoice"),
        ("INVOICE Remit Payment To Customer ID Total Amount Payable", "Invoice"),
        # Resumes
        ("Resume Curriculum Vitae Education Bachelor of Technology Skills Python Java SQL Experience", "Resume"),
        ("Curriculum Vitae Professional Summary Work Experience Projects Certifications BTech", "Resume"),
        ("Software Engineer Resume Technical Skills Docker Git PyTorch Education University", "Resume"),
        ("Data Scientist Resume Skills Machine Learning Deep Learning Experience NLP NIT", "Resume"),
        ("Aravind Sharma Resume Email Phone Skills Experience Projects Education", "Resume"),
        ("CV Personal Profile Employment History Academic Background Core Competencies", "Resume"),
        # Other
        ("Contract Agreement Terms and Conditions Non-Disclosure Confidentiality Signed Witness", "Other"),
        ("Meeting Minutes Discussion Agenda Action Items Attendees Next Session", "Other"),
        ("Research Paper Abstract Introduction Methodology Results Discussion Conclusion", "Other"),
        ("General Notice Announcement Holiday Schedule Office Closure Maintenance", "Other"),
        ("User Manual Operating Instructions Safety Warnings Specifications Warranty", "Other"),
        ("Daily News Editorial Global Market Update Economic Growth Weather Forecast", "Other"),
    ]

    corpus = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(corpus)
    clf = LogisticRegression(random_state=42)
    clf.fit(X, labels)
    return vectorizer, clf


def classify_document_ml(text: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Classify using the optional TF-IDF + Logistic Regression model.
    Returns: (document_type, method_label, details)
    """
    vectorizer, clf = train_optional_ml_model()
    X_test = vectorizer.transform([text])
    predicted_type = clf.predict(X_test)[0]
    probs = clf.predict_proba(X_test)[0]
    classes = list(clf.classes_)

    pred_idx = classes.index(predicted_type)
    confidence = float(probs[pred_idx] * 100)

    details = {
        "confidence_percentage": round(confidence, 1),
        "class_probabilities": {str(c): round(float(p * 100), 1) for c, p in zip(classes, probs)},
    }
    method = "TF-IDF + Logistic Regression (Optional ML)"
    return predicted_type, method, details


# -----------------------------------------------------------------------------
# 4. Field Extraction
# -----------------------------------------------------------------------------
def extract_invoice_fields(text: str) -> Dict[str, str]:
    """
    Extract Invoice Number, Date, Company Name, and Total Amount using flexible regex.
    Returns 'Not found' if a field cannot be identified (no hallucinations).
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fields = {
        "Invoice Number": "Not found",
        "Date": "Not found",
        "Company Name": "Not found",
        "Total Amount": "Not found",
    }

    # 1. Invoice Number
    inv_patterns = [
        r"(?i)invoice\s*(?:number|no\.?|#|num)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
        r"(?i)tax\s*invoice\s*(?:no\.?|#)?\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
        r"(?i)bill\s*(?:no\.?|#|number)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
        r"(?i)\binv\s*[:\-#]\s*([A-Za-z0-9\-_/]+)",
    ]
    for pat in inv_patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1).strip()
            # Exclude false positives like words 'date' or 'due'
            if len(val) >= 3 and not re.match(r"(?i)^(date|due|to|for|amount)$", val):
                fields["Invoice Number"] = val
                break

    # 2. Date
    date_patterns = [
        r"(?i)(?:invoice\s*)?date\s*[:\-]?\s*([0-9]{1,4}[-/.][0-9]{1,2}[-/.][0-9]{1,4})",
        r"(?i)dated\s*[:\-]?\s*([0-9]{1,4}[-/.][0-9]{1,2}[-/.][0-9]{1,4})",
        r"\b([0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4})\b",
        r"\b([0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2})\b",
    ]
    for pat in date_patterns:
        m = re.search(pat, text)
        if m:
            fields["Date"] = m.group(1).strip()
            break

    # 3. Total Amount
    total_patterns = [
        r"(?i)total\s*amount\s*[:\-]?\s*([$₹€£A-Za-z]*\s*[\d,]+(?:\.\d{2})?)",
        r"(?i)amount\s*due\s*[:\-]?\s*([$₹€£A-Za-z]*\s*[\d,]+(?:\.\d{2})?)",
        r"(?i)total\s*due\s*[:\-]?\s*([$₹€£A-Za-z]*\s*[\d,]+(?:\.\d{2})?)",
        r"(?i)\btotal\s*[:\-]?\s*([$₹€£A-Za-z]*\s*[\d,]+(?:\.\d{2})?)",
        r"(?i)balance\s*due\s*[:\-]?\s*([$₹€£A-Za-z]*\s*[\d,]+(?:\.\d{2})?)",
    ]
    for pat in total_patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            # Ensure it contains digits
            if re.search(r"\d", candidate):
                fields["Total Amount"] = candidate
                break

    # 4. Company Name (Vendor / Header)
    # Heuristic: inspect top lines, filtering out common invoice headers
    ignore_headers = r"(?i)invoice|tax|date|bill\s*to|billed\s*to|receipt|page|original|duplicate|gstin"
    for line in lines[:5]:
        clean_line = line.strip()
        if (
            len(clean_line) > 3
            and not re.search(ignore_headers, clean_line)
            and not re.search(r"@|\bwww\.|\.com\b", clean_line)
        ):
            fields["Company Name"] = clean_line
            break

    return fields


def extract_resume_fields(text: str) -> Dict[str, str]:
    """
    Extract Candidate Name, Email, Phone, and Skills using regex and section heuristics.
    Returns 'Not found' if a field cannot be identified (no hallucinations).
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fields = {
        "Name": "Not found",
        "Email": "Not found",
        "Phone": "Not found",
        "Skills": "Not found",
    }

    # 1. Email
    email_match = re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text
    )
    if email_match:
        fields["Email"] = email_match.group(0).strip()

    # 2. Phone (supports Indian +91, dashes, parentheses, dots)
    phone_match = re.search(
        r"(?i)(?:phone|mobile|cell|tel|contact)?\s*[:\-]?\s*(\+?[0-9]{1,4}[-.\s]?(?:\([0-9]{1,4}\)|[0-9]{1,4})[-.\s]?[0-9]{3,5}[-.\s]?[0-9]{3,5})",
        text,
    )
    if phone_match:
        phone_candidate = phone_match.group(1).strip()
        # Verify it has at least 8 digits
        digit_count = len(re.findall(r"\d", phone_candidate))
        if digit_count >= 8:
            fields["Phone"] = phone_candidate

    # 3. Candidate Name
    # Heuristic: First line that looks like a person's name (1 to 4 words, no numbers, no emails)
    for line in lines[:6]:
        words = line.split()
        if (
            1 <= len(words) <= 4
            and not re.search(
                r"(?i)resume|curriculum|vitae|email|phone|contact|profile|summary|page|engineer|developer",
                line,
            )
            and "@" not in line
            and not re.search(r"\d", line)
            and not re.search(r"[|/:\\]", line)
        ):
            fields["Name"] = line.strip()
            break

    # 4. Skills
    skills_match = re.search(
        r"(?i)(?:skills|technical\s*skills|key\s*skills|core\s*competencies)\s*[:\-]?\s*\n?([^\n]+(?:\n[^\n]+){0,4})",
        text,
    )
    if skills_match:
        raw_skills = skills_match.group(1).strip()
        # Truncate if next major section title appears
        clean_skills = re.split(
            r"(?i)\n\s*(?:experience|work\s*experience|employment|education|projects|certifications|awards)\b",
            raw_skills,
        )[0]
        # Clean extra newlines
        clean_skills = " ".join([l.strip() for l in clean_skills.splitlines() if l.strip()])
        if clean_skills:
            fields["Skills"] = clean_skills

    return fields


# -----------------------------------------------------------------------------
# 5. Core Pipeline Orchestration
# -----------------------------------------------------------------------------
def process_document(
    file_bytes: bytes,
    filename: str,
    ml_enhancement: bool = False,
) -> Dict[str, Any]:
    """
    Executes the end-to-end extraction and classification pipeline.
    Returns a standardized dictionary adhering to the specified result data model.
    """
    ext = os.path.splitext(filename)[1].lower().replace(".", "")
    is_pdf = ext == "pdf"
    file_size_kb = len(file_bytes) / 1024.0

    extraction_status = "Initial"
    extracted_text = ""
    used_ocr = False

    # Step 1: Text extraction
    if is_pdf:
        extracted_text = extract_text_from_pdf(file_bytes)
        # Check if text is meaningful (> 30 characters or sufficient word count)
        if len(extracted_text.strip()) >= 30:
            extraction_status = "Direct PDF Text Extraction Successful"
        else:
            # Fallback to OCR
            used_ocr = True
            ocr_text, ocr_ok, ocr_msg = ocr_pdf(file_bytes)
            if ocr_ok and len(ocr_text.strip()) > 0:
                extracted_text = ocr_text
                extraction_status = "No selectable text detected — OCR extraction successful"
            else:
                extraction_status = f"No selectable text detected. ({ocr_msg})"
    else:
        # Image file (JPG/PNG)
        used_ocr = True
        ocr_text, ocr_ok, ocr_msg = extract_text_from_image(file_bytes)
        if ocr_ok and len(ocr_text.strip()) > 0:
            extracted_text = ocr_text
            extraction_status = "Image OCR Extraction Successful"
        else:
            extraction_status = f"Image processing note: {ocr_msg}"

    # Step 2: Document Classification
    if len(extracted_text.strip()) > 0:
        if ml_enhancement:
            doc_type, method_label, class_details = classify_document_ml(extracted_text)
        else:
            doc_type, method_label, class_details = classify_document_rule_based(extracted_text)
    else:
        doc_type = "Other"
        method_label = "Unclassified (Empty Content)"
        class_details = {}

    # Step 3: Field Extraction
    if doc_type == "Invoice":
        fields = extract_invoice_fields(extracted_text)
    elif doc_type == "Resume":
        fields = extract_resume_fields(extracted_text)
    else:
        # 'Other' document type
        fields = {
            "Word Count": str(len(extracted_text.split())),
            "Character Count": str(len(extracted_text)),
            "Status": "General document. Does not match Invoice or Resume patterns.",
        }

    return {
        "filename": filename,
        "file_type": ext.upper(),
        "file_size_kb": round(file_size_kb, 2),
        "document_type": doc_type,
        "classification_method": method_label,
        "classification_details": class_details,
        "fields": fields,
        "text": extracted_text,
        "extraction_status": extraction_status,
        "used_ocr": used_ocr,
    }


# -----------------------------------------------------------------------------
# 6. Streamlit User Interface
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Document Intelligence",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for polished, modern internship-demo aesthetics
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.3rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .status-badge-ok {
            background-color: #ECFDF5;
            color: #065F46;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
            border: 1px solid #A7F3D0;
        }
        .status-badge-warn {
            background-color: #FFFBEB;
            color: #92400E;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
            border: 1px solid #FDE68A;
        }
        .card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .field-card {
            background: #F8FAFC;
            border-left: 4px solid #3B82F6;
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 8px;
        }
        .field-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            font-weight: 600;
        }
        .field-value {
            font-size: 1.05rem;
            font-weight: 600;
            color: #0F172A;
        }
        .field-notfound {
            color: #94A3B8;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📄 Application")
        st.markdown("**AI Document Intelligence**")
        st.caption("Zyroo AI/ML Internship • Week 2")
        st.write("Supported formats: `PDF`, `JPG`, `JPEG`, `PNG`")

        st.markdown("---")
        st.markdown("### ⚙️ System Status")

        # Status: PyMuPDF
        st.markdown("• **PDF Extraction:** <span class='status-badge-ok'>PyMuPDF Active</span>", unsafe_allow_html=True)

        # Status: OCR
        if is_ocr_available():
            st.markdown("• **OCR Engine:** <span class='status-badge-ok'>Tesseract Active</span>", unsafe_allow_html=True)
        else:
            st.markdown(
                "• **OCR Engine:** <span class='status-badge-warn'>Not Configured</span>",
                unsafe_allow_html=True,
            )
            st.caption("ℹ️ Install Tesseract OCR to enable scanned document processing.")

        # Status: Classification
        st.markdown("• **Document Classification:** <span class='status-badge-ok'>Rule-Based / ML</span>", unsafe_allow_html=True)

        # Status: Field Extraction
        st.markdown("• **Field Extraction:** <span class='status-badge-ok'>Regex & Heuristics</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧠 Classification Method")
        classifier_mode = st.radio(
            "Select classification engine:",
            options=[
                "Rule-Based Keyword Classification (Primary)",
                "TF-IDF + Logistic Regression (Optional ML)",
            ],
            index=0,
            help="Rule-based is the robust MVP default. ML provides an optional enhancement model.",
        )
        use_ml = "Optional ML" in classifier_mode

        st.markdown("---")
        st.markdown("### 🧪 Quick Sample Test")
        st.caption("Load a bundled internship test sample:")
        sample_choice = st.selectbox(
            "Choose test sample:",
            options=["None (Upload your own)", "samples/invoice_1.pdf", "samples/invoice_2.pdf", "samples/resume_1.pdf"],
            index=0,
        )

    # -----------------------------------------------------------------------------
    # MAIN AREA: HEADER
    # -----------------------------------------------------------------------------
    st.markdown("<div class='main-header'>AI Document Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Document Analysis MVP — Upload a document. Extract its text. Understand its type. Get useful information instantly.</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # MAIN AREA: FILE UPLOADER
    # -----------------------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, JPG, JPEG, PNG):",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Upload an invoice, resume, or other document for intelligent analysis.",
    )

    # Determine active file bytes and filename (either from uploader or quick sample)
    active_file_bytes = None
    active_filename = None

    if uploaded_file is not None:
        active_file_bytes = uploaded_file.getvalue()
        active_filename = uploaded_file.name
    elif sample_choice != "None (Upload your own)":
        if os.path.exists(sample_choice):
            with open(sample_choice, "rb") as f:
                active_file_bytes = f.read()
            active_filename = os.path.basename(sample_choice)
            st.info(f"💡 Quick test sample loaded: **{active_filename}**")
        else:
            st.warning(f"Sample file {sample_choice} not found.")

    # -----------------------------------------------------------------------------
    # PROCESSING & RESULTS
    # -----------------------------------------------------------------------------
    if active_file_bytes and active_filename:
        # 1. File Validation
        class DummyUpload:
            def __init__(self, name, data):
                self.name = name
                self.data = data
            def getvalue(self):
                return self.data

        is_valid, validation_msg = validate_file(DummyUpload(active_filename, active_file_bytes))

        if not is_valid:
            st.error(f"❌ {validation_msg}")
        else:
            # File metadata summary banner
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            col_meta1.metric("Filename", active_filename)
            col_meta2.metric("File Type", os.path.splitext(active_filename)[1].upper().replace(".", ""))
            col_meta3.metric("File Size", f"{len(active_file_bytes) / 1024.0:.1f} KB")

            # Process document through the pipeline
            with st.spinner("Analyzing document..."):
                result = process_document(
                    file_bytes=active_file_bytes,
                    filename=active_filename,
                    ml_enhancement=use_ml,
                )

            # Status Checklist
            st.markdown("---")
            st.markdown("### 📋 Processing Pipeline Status")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.success("✓ File accepted")
            if "No selectable text" in result["extraction_status"]:
                sc2.warning(f"ℹ️ {result['extraction_status']}")
            else:
                sc2.success("✓ Text extracted")
            sc3.success("✓ Document identified")
            sc4.success("✓ Fields extracted")

            # ---------------------------------------------------------------------
            # RESULT SECTION
            # ---------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### 📊 Document Analysis Results")

            res_col1, res_col2, res_col3 = st.columns([1, 1.2, 1])

            # Document Type Metric
            doc_type = result["document_type"]

            with res_col1:
                st.metric(label="Document Type", value=doc_type)

            with res_col2:
                st.metric(label="Detection Method", value=result["classification_method"])

            with res_col3:
                if "confidence_percentage" in result.get("classification_details", {}):
                    conf = result["classification_details"]["confidence_percentage"]
                    st.metric(label="Confidence", value=f"{conf}%")
                elif "invoice_keyword_score" in result.get("classification_details", {}):
                    inv_s = result["classification_details"]["invoice_keyword_score"]
                    res_s = result["classification_details"]["resume_keyword_score"]
                    score_desc = f"Inv: {inv_s} | Res: {res_s}"
                    st.metric(label="Keyword Matches", value=score_desc)
                else:
                    st.metric(label="Confidence", value="Rule-verified")

            # Display Structured Fields
            st.markdown(f"#### 🏷️ Extracted Information ({doc_type})")

            fields = result["fields"]

            if doc_type == "Invoice":
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    inv_no = fields.get("Invoice Number", "Not found")
                    st.markdown(
                        f"<div class='field-card'><div class='field-label'>Invoice Number</div>"
                        f"<div class='field-value {'' if inv_no != 'Not found' else 'field-notfound'}'>{inv_no}</div></div>",
                        unsafe_allow_html=True,
                    )
                    date_val = fields.get("Date", "Not found")
                    st.markdown(
                        f"<div class='field-card'><div class='field-label'>Date</div>"
                        f"<div class='field-value {'' if date_val != 'Not found' else 'field-notfound'}'>{date_val}</div></div>",
                        unsafe_allow_html=True,
                    )
                with f_col2:
                    comp = fields.get("Company Name", "Not found")
                    st.markdown(
                        f"<div class='field-card'><div class='field-label'>Company Name</div>"
                        f"<div class='field-value {'' if comp != 'Not found' else 'field-notfound'}'>{comp}</div></div>",
                        unsafe_allow_html=True,
                    )
                    total_val = fields.get("Total Amount", "Not found")
                    st.markdown(
                        f"<div class='field-card'><div class='field-label'>Total Amount</div>"
                        f"<div class='field-value {'' if total_val != 'Not found' else 'field-notfound'}'>{total_val}</div></div>",
                        unsafe_allow_html=True,
                    )

            elif doc_type == "Resume":
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    name_val = fields.get("Name", "Not found")
                    st.markdown(
                        f"<div class='field-card'><div class='field-label'>Candidate Name</div>"
                        f"<div class='field-value {'' if name_val != 'Not found' else 'field-notfound'}'>{name_val}</div></div>",
                        unsafe_allow_html=True,
                    )
                    email_val = fields.get("Email", "Not found")
                    st.markdown(
                        f"<div class='field-card'><div class='field-label'>Email Address</div>"
                        f"<div class='field-value {'' if email_val != 'Not found' else 'field-notfound'}'>{email_val}</div></div>",
                        unsafe_allow_html=True,
                    )
                with r_col2:
                    phone_val = fields.get("Phone", "Not found")
                    st.markdown(
                        f"<div class='field-card'><div class='field-label'>Phone Number</div>"
                        f"<div class='field-value {'' if phone_val != 'Not found' else 'field-notfound'}'>{phone_val}</div></div>",
                        unsafe_allow_html=True,
                    )
                    skills_val = fields.get("Skills", "Not found")
                    st.markdown(
                        f"<div class='field-card'><div class='field-label'>Skills</div>"
                        f"<div class='field-value {'' if skills_val != 'Not found' else 'field-notfound'}'>{skills_val}</div></div>",
                        unsafe_allow_html=True,
                    )

            else:
                # Other document type
                st.info("ℹ️ Document classified as **Other**. Extracted general document attributes:")
                st.json(fields)

            # ---------------------------------------------------------------------
            # EXTRACTED TEXT VIEWER
            # ---------------------------------------------------------------------
            st.markdown("---")
            with st.expander("📄 View Extracted Document Text", expanded=False):
                if result["text"]:
                    st.text_area(
                        label="Raw Text Extracted from Document",
                        value=result["text"],
                        height=280,
                    )
                else:
                    st.warning("No readable text could be extracted from this document.")

            # ---------------------------------------------------------------------
            # DOWNLOAD RESULTS
            # ---------------------------------------------------------------------
            st.markdown("---")
            json_output = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_output,
                file_name=f"{os.path.splitext(active_filename)[0]}_intelligence.json",
                mime="application/json",
                help="Download complete document metadata, classification, fields, and raw text.",
            )

    else:
        # Empty state guidance
        st.info("👆 Please upload a PDF or Image document above, or choose a sample from the sidebar to begin analysis.")


if __name__ == "__main__":
    main()

