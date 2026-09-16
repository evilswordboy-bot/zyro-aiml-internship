"""
AI Document Intelligence & Workflow Platform
Week 3: Improved Document Understanding
Understand. Classify. Extract.
"""

import os
import io
import time
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Import Modular Core
from src.document_reader import DocumentReader
from src.ocr_processor import OCRProcessor
from src.text_cleaner import TextCleaner
from src.classifier import DocumentClassifier
from src.extractor import DocumentExtractor, NOT_FOUND
from src.evaluator import ModelEvaluator
from src.utils import (
    load_dataset_from_dir,
    export_result_to_json,
    export_result_to_csv,
    generate_rich_sample_pdfs
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")


# =============================================================================
# INITIALIZATION & MODEL CACHING
# =============================================================================

@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initializes sample files, loads training dataset, and prepares classifiers."""
    generate_rich_sample_pdfs(SAMPLES_DIR)

    clf = DocumentClassifier()
    model_loaded = clf.load_model(MODEL_PATH)

    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    train_texts, train_labels = load_dataset_from_dir(train_dir)
    test_texts, test_labels = load_dataset_from_dir(test_dir)

    if not model_loaded and train_texts:
        clf.train(train_texts, train_labels)
        try:
            clf.save_model(MODEL_PATH)
        except Exception:
            pass

    # Compute evaluation metrics across test set
    eval_results = None
    if test_texts and test_labels and clf.is_trained:
        eval_results = ModelEvaluator.evaluate_all_models(clf, test_texts, test_labels)

    ocr_engine = OCRProcessor()
    return clf, ocr_engine, eval_results


# =============================================================================
# PIPELINE CONTROLLER
# =============================================================================

def process_document(
    file_bytes: bytes,
    filename: str,
    file_extension: str,
    classifier: DocumentClassifier,
    ocr_engine: OCRProcessor,
    active_model_name: str,
    ocr_settings: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Executes the 6-stage end-to-end processing pipeline:
    1. Validate
    2. Extract Text / OCR
    3. Clean Text
    4. Classify Type
    5. Extract Fields
    6. Check Missing Fields
    """
    ext = file_extension.lower()
    stages = []

    # Stage 1: Document Uploaded & Validated
    stages.append("Document Uploaded")

    # Stage 2: Text Extraction / OCR
    raw_text = ""
    ocr_used = False
    ocr_steps = []
    extraction_note = ""

    if ext == "pdf":
        read_res = DocumentReader.read_pdf(file_bytes)
        if read_res["error"]:
            return {"error": read_res["error"], "stages": stages}

        if read_res["needs_ocr"]:
            ocr_res = ocr_engine.ocr_pdf(
                file_bytes,
                preprocess=ocr_settings.get("preprocess", True)
            )
            raw_text = ocr_res["text"]
            ocr_used = True
            ocr_steps = ocr_res.get("preprocessing_steps", [])
            extraction_note = "Scanned PDF detected — executed OCR fallback."
        else:
            raw_text = read_res["text"]
            extraction_note = f"Direct PyMuPDF selectable text extracted ({read_res['page_count']} page(s))."
    elif ext in ("jpg", "jpeg", "png"):
        ocr_res = ocr_engine.extract_from_image(
            file_bytes,
            preprocess=ocr_settings.get("preprocess", True)
        )
        raw_text = ocr_res["text"]
        ocr_used = True
        ocr_steps = ocr_res.get("preprocessing_steps", [])
        extraction_note = "Image upload processed via OCR engine."
    else:
        return {
            "error": "Unsupported file format. Please upload a PDF, JPG, JPEG, or PNG.",
            "stages": stages
        }

    stages.append("Text Extracted")

    # Stage 3: Text Cleaning & Normalization
    clean_res = TextCleaner.clean(raw_text)
    cleaned_text = clean_res["cleaned_text"]
    stages.append("Text Cleaned")

    if not clean_res["is_valid"]:
        return {
            "filename": filename,
            "file_type": ext.upper(),
            "document_type": "Unreadable",
            "confidence": "Not Available",
            "classification_model": active_model_name,
            "fields": {},
            "missing_fields": [],
            "original_text": raw_text,
            "cleaned_text": cleaned_text,
            "ocr_used": ocr_used,
            "ocr_steps": ocr_steps,
            "extraction_note": extraction_note,
            "stages": stages,
            "error": clean_res.get("warning") or "Unable to extract readable text from this document. Please upload a clearer document."
        }

    # Stage 4: Document Classification
    pred_res = classifier.predict(cleaned_text, model_name=active_model_name)
    doc_type = pred_res["document_type"]
    confidence = pred_res["confidence"]
    stages.append("Document Classified")

    # Stage 5 & 6: Information Extraction & Missing Fields Detection
    extract_res = DocumentExtractor.extract(cleaned_text, doc_type)
    fields = extract_res["fields"]
    missing_fields = extract_res["missing_fields"]
    stages.append("Fields Extracted")
    stages.append("Missing Fields Checked")

    return {
        "filename": filename,
        "file_type": ext.upper(),
        "document_type": doc_type,
        "confidence": confidence,
        "classification_model": active_model_name,
        "probabilities": pred_res.get("probabilities"),
        "fields": fields,
        "missing_fields": missing_fields,
        "fields_found_count": extract_res.get("fields_found_count", 0),
        "total_fields": extract_res.get("total_fields", 0),
        "original_text": raw_text,
        "cleaned_text": cleaned_text,
        "char_count_original": clean_res["char_count_original"],
        "char_count_cleaned": clean_res["char_count_cleaned"],
        "reduction_pct": clean_res["reduction_pct"],
        "word_count": len(cleaned_text.split()),
        "ocr_used": ocr_used,
        "ocr_steps": ocr_steps,
        "extraction_note": extraction_note,
        "stages": stages,
        "error": None
    }


# =============================================================================
# STREAMLIT UI
# =============================================================================

def render_ui():
    st.set_page_config(
        page_title="AI Document Intelligence & Workflow Platform",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.1rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .badge-tag {
            background-color: #EEF2FF;
            color: #4338CA;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        .stage-pill {
            display: inline-flex;
            align-items: center;
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            padding: 0.28rem 0.65rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }
        .entity-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        .entity-label {
            font-size: 0.75rem;
            color: #64748B;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .entity-value {
            font-size: 1.15rem;
            color: #0F172A;
            font-weight: 600;
            margin-top: 0.2rem;
            word-break: break-word;
        }
        .entity-missing {
            color: #DC2626 !important;
            font-style: italic;
        }
        .conf-badge {
            background-color: #F1F5F9;
            border: 1px solid #CBD5E1;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize Core Systems
    clf, ocr_engine, eval_results = initialize_system()

    # -------------------------------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.title("📄 Doc Intelligence")
        st.caption("AI Document Understanding & Classification")

        st.markdown("---")
        st.subheader("⚙️ Classification Model")

        model_options = [
            "Logistic Regression",
            "Linear SVM",
            "Naive Bayes",
            "Rule-Based Baseline"
        ]
        selected_model = st.selectbox(
            "Active Classifier:",
            model_options,
            index=0,
            help="Choose the model used to classify documents. Logistic Regression & Naive Bayes provide calibrated probabilistic confidence."
        )

        st.markdown("---")
        st.subheader("🔍 OCR & Preprocessing")
        if ocr_engine.tesseract_available:
            st.success("Tesseract OCR: Active")
        else:
            st.info("PyMuPDF Direct Extraction: Active (OCR available when Tesseract is installed)")

        with st.expander("Image Preprocessing Options", expanded=False):
            apply_preprocess = st.checkbox("Enable Image Preprocessing", value=True)
            apply_grayscale = st.checkbox("Grayscale Conversion", value=True, disabled=not apply_preprocess)
            apply_threshold = st.checkbox("Contrast Thresholding", value=True, disabled=not apply_preprocess)
            apply_denoise = st.checkbox("Noise Reduction Filter", value=True, disabled=not apply_preprocess)

        ocr_settings = {
            "preprocess": apply_preprocess,
            "grayscale": apply_grayscale,
            "threshold": apply_threshold,
            "denoise": apply_denoise
        }

        st.markdown("---")
        st.subheader("📁 Test Samples Showcase")
        st.caption("Load a verified test sample directly:")

        sample_choices = [
            "None (Upload my own file)",
            "invoice_1.pdf (Consulting Invoice - $ 1,450.00)",
            "invoice_2.pdf (Tax Invoice - Rs. 78,500)",
            "invoice_missing_total.pdf (Invoice with Missing Total)",
            "resume_1.pdf (Software Engineer Resume)",
            "resume_missing_phone.pdf (Resume with Missing Phone)",
            "meeting_minutes.pdf (Other - Strategy Memo)"
        ]
        selected_sample_label = st.selectbox("Choose sample:", sample_choices)

    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------
    st.markdown('<span class="badge-tag">AI DOCUMENT INTELLIGENCE & WORKFLOW</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">AI Document Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header"><strong>Understand. Classify. Extract.</strong> Ingest documents, extract normalized text, identify document types, and extract key fields with honest confidence metrics.</p>', unsafe_allow_html=True)

    # Tabs for Workflow vs Model Evaluation
    tab_workflow, tab_evaluation = st.tabs(["🚀 Document Analysis Pipeline", "📊 Model Evaluation & Benchmarks"])

    # =========================================================================
    # TAB 1: WORKFLOW PIPELINE
    # =========================================================================
    with tab_workflow:
        # File Upload Area
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, JPG, JPEG, PNG):",
            type=["pdf", "jpg", "jpeg", "png"],
            help="Drop any invoice, resume, or business document to extract and analyze."
        )

        file_bytes: Optional[bytes] = None
        filename: str = ""
        file_type: str = ""
        file_size_bytes: int = 0

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            file_type = filename.split(".")[-1].lower()
            file_size_bytes = len(file_bytes)
        elif selected_sample_label != "None (Upload my own file)":
            sample_filename = selected_sample_label.split(" ")[0]
            sample_path = os.path.join(SAMPLES_DIR, sample_filename)
            if os.path.isfile(sample_path):
                with open(sample_path, "rb") as f:
                    file_bytes = f.read()
                filename = sample_filename
                file_type = filename.split(".")[-1].lower()
                file_size_bytes = len(file_bytes)
                st.info(f"Loaded sample file: **`{filename}`**")

        # Process Document if provided
        if file_bytes is not None:
            # Metadata summary
            size_kb = file_size_bytes / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Filename", filename)
            with col_m2:
                st.metric("File Type", file_type.upper())
            with col_m3:
                st.metric("File Size", size_str)
            with col_m4:
                st.metric("Active Model", selected_model)

            with st.spinner("Processing document through intelligent pipeline..."):
                result = process_document(
                    file_bytes=file_bytes,
                    filename=filename,
                    file_extension=file_type,
                    classifier=clf,
                    ocr_engine=ocr_engine,
                    active_model_name=selected_model,
                    ocr_settings=ocr_settings
                )

            # Error handling
            if result.get("error"):
                st.error(f"Processing Note: {result['error']}")
                return

            # Display Pipeline Progress Stages
            st.markdown("---")
            st.markdown("### 🔄 Processing Stages")
            stage_html = "".join([f'<span class="stage-pill">✓ {stage}</span>' for stage in result.get("stages", [])])
            st.markdown(stage_html, unsafe_allow_html=True)
            st.caption(f"Extraction Route: {result['extraction_note']}")

            # Classification & Result Section
            st.markdown("---")
            st.subheader("📑 Document Classification & Confidence")

            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.markdown(f"### Detected Type: **{result['document_type']}**")
                st.markdown(f"**Classifier:** `{result['classification_model']}`")

            with res_col2:
                conf = result.get("confidence", "Not Available")
                st.markdown(f"### Confidence: **{conf}**")
                if result.get("probabilities"):
                    prob_str = " | ".join([f"{k}: {round(v*100)}%" for k, v in result["probabilities"].items()])
                    st.caption(f"Class distribution: {prob_str}")
                else:
                    st.caption("Confidence: Probabilistic calibration is not available for this model.")

            # Structured Information Extraction
            st.markdown("---")
            st.subheader("🔍 Extracted Structured Entities")

            fields = result.get("fields", {})
            doc_type = result.get("document_type")

            if doc_type == "Invoice":
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)

                inv_num = fields.get("Invoice Number", NOT_FOUND)
                inv_class = "entity-missing" if inv_num == NOT_FOUND else ""
                with c1:
                    st.markdown(f"""
                    <div class="entity-card">
                        <div class="entity-label">Invoice Number</div>
                        <div class="entity-value {inv_class}">{inv_num}</div>
                    </div>
                    """, unsafe_allow_html=True)

                inv_date = fields.get("Date", NOT_FOUND)
                date_class = "entity-missing" if inv_date == NOT_FOUND else ""
                with c2:
                    st.markdown(f"""
                    <div class="entity-card">
                        <div class="entity-label">Date</div>
                        <div class="entity-value {date_class}">{inv_date}</div>
                    </div>
                    """, unsafe_allow_html=True)

                comp = fields.get("Company Name", NOT_FOUND)
                comp_class = "entity-missing" if comp == NOT_FOUND else ""
                with c3:
                    st.markdown(f"""
                    <div class="entity-card">
                        <div class="entity-label">Company Name</div>
                        <div class="entity-value {comp_class}">{comp}</div>
                    </div>
                    """, unsafe_allow_html=True)

                total = fields.get("Total Amount", NOT_FOUND)
                total_class = "entity-missing" if total == NOT_FOUND else ""
                with c4:
                    st.markdown(f"""
                    <div class="entity-card">
                        <div class="entity-label">Total Amount</div>
                        <div class="entity-value {total_class}">{total}</div>
                    </div>
                    """, unsafe_allow_html=True)

            elif doc_type == "Resume":
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)

                name = fields.get("Name", NOT_FOUND)
                name_class = "entity-missing" if name == NOT_FOUND else ""
                with c1:
                    st.markdown(f"""
                    <div class="entity-card">
                        <div class="entity-label">Candidate Name</div>
                        <div class="entity-value {name_class}">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)

                email = fields.get("Email", NOT_FOUND)
                email_class = "entity-missing" if email == NOT_FOUND else ""
                with c2:
                    st.markdown(f"""
                    <div class="entity-card">
                        <div class="entity-label">Email Address</div>
                        <div class="entity-value {email_class}">{email}</div>
                    </div>
                    """, unsafe_allow_html=True)

                phone = fields.get("Phone", NOT_FOUND)
                phone_class = "entity-missing" if phone == NOT_FOUND else ""
                with c3:
                    st.markdown(f"""
                    <div class="entity-card">
                        <div class="entity-label">Phone Number</div>
                        <div class="entity-value {phone_class}">{phone}</div>
                    </div>
                    """, unsafe_allow_html=True)

                skills = fields.get("Skills", NOT_FOUND)
                skills_class = "entity-missing" if skills == NOT_FOUND else ""
                with c4:
                    st.markdown(f"""
                    <div class="entity-card">
                        <div class="entity-label">Skills & Competencies</div>
                        <div class="entity-value {skills_class}">{skills}</div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info("The document is identified as **Other**. General text has been cleaned and classified. Field extraction schemas are currently tailored for Invoices and Resumes.")

            # Missing Fields Summary Alert
            missing_fields = result.get("missing_fields", [])
            if missing_fields:
                st.warning(f"⚠️ **Missing Fields Detected**: {', '.join(missing_fields)} — explicitly marked as **`Not Found`**.")

            # Text Transparency Section
            st.markdown("---")
            st.subheader("📄 Text Transparency & Preprocessing")

            exp1, exp2, exp3 = st.tabs(["Cleaned Text (Active Input)", "Original Extracted Text", "OCR & CV Processing"])

            with exp1:
                st.caption(f"Cleaned Characters: {result['char_count_cleaned']} | Words: {result['word_count']} | Whitespace Reduction: {result['reduction_pct']}%")
                st.text_area("Cleaned Text Content", value=result["cleaned_text"], height=220, disabled=True)

            with exp2:
                st.caption(f"Original Characters: {result['char_count_original']}")
                st.text_area("Raw Extracted Content", value=result["original_text"], height=220, disabled=True)

            with exp3:
                st.markdown(f"**OCR Used:** `{result['ocr_used']}`")
                if result.get("ocr_steps"):
                    st.markdown("**Image Preprocessing Steps Applied:**")
                    for step in result["ocr_steps"]:
                        st.markdown(f"- {step}")
                else:
                    st.markdown("*No OCR image preprocessing needed (Direct selectable text was used).*")

            # Export Section
            st.markdown("---")
            st.subheader("💾 Export Structured Data")
            d1, d2, d3 = st.columns(3)

            with d1:
                json_data = export_result_to_json(result)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_data,
                    file_name=f"{filename}_analysis.json",
                    mime="application/json",
                    use_container_width=True
                )

            with d2:
                csv_data = export_result_to_csv(result)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"{filename}_metadata.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with d3:
                st.download_button(
                    label="⬇️ Download Cleaned Text (.txt)",
                    data=result["cleaned_text"],
                    file_name=f"{filename}_cleaned.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # =========================================================================
    # TAB 2: MODEL EVALUATION & BENCHMARKS
    # =========================================================================
    with tab_evaluation:
        st.subheader("📊 Classifier Benchmark & Performance Comparison")
        st.markdown(
            "Evaluation performed across an independent, unseen test dataset (`data/test/`) "
            "using consistent TF-IDF feature representations for fair comparison."
        )

        if eval_results:
            # Model Comparison Table
            df_comp = pd.DataFrame(eval_results["comparison_table"])
            st.table(df_comp.set_index("Model"))

            st.success(f"🏆 **Selected Best Performing Model:** `{eval_results['best_model_name']}` (Identified via highest Macro F1-Score)")

            st.markdown("---")
            st.subheader("🔲 Confusion Matrix Diagnostics")

            model_for_cm = st.selectbox(
                "Select Model to Inspect Confusion Matrix:",
                options=list(eval_results["detailed_metrics"].keys()),
                index=0
            )

            cm_data = eval_results["detailed_metrics"][model_for_cm]["confusion_matrix"]
            classes = eval_results["classes"]
            df_cm = pd.DataFrame(cm_data, index=[f"Actual {c}" for c in classes], columns=[f"Predicted {c}" for c in classes])

            st.table(df_cm)

            st.markdown("---")
            st.subheader("📝 Explainable Diagnostic Report")
            diag = eval_results.get("diagnostics", {})

            st.markdown(f"**Strengths:** {diag.get('strengths')}")
            st.markdown(f"**Common Confusion Patterns:** {diag.get('confusion_patterns')}")
            st.markdown(f"**Potential Causes of Errors:** {diag.get('causes_of_errors')}")
            st.markdown(f"**Dataset Limitations:** {diag.get('dataset_limitations')}")
        else:
            st.warning("Evaluation metrics currently loading or test dataset is unavailable.")


if __name__ == "__main__":
    render_ui()
