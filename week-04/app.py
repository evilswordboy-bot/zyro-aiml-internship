"""
AI Document Intelligence & Workflow Platform
Enterprise Document Management & Understanding Platform
Persistent SQLite Repository • SHA-256 Deduplication • Multi-Field Search • SaaS UX
"""

import os
import io
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd
from PIL import Image

# Core Services & Database
from database.db import DatabaseManager, get_db
from models.document import (
    STATUS_PROCESSED,
    STATUS_NEEDS_REVIEW,
    STATUS_FAILED,
    TYPE_INVOICE,
    TYPE_RESUME,
    TYPE_OTHER,
    ALL_DOCUMENT_TYPES,
    ALL_STATUSES
)
from services.file_storage import FileStorageManager, get_storage_manager
from services.document_processor import DocumentProcessor, get_document_processor
from src.classifier import DocumentClassifier
from src.ocr_processor import OCRProcessor
from src.evaluator import ModelEvaluator
from src.utils import (
    load_dataset_from_dir,
    generate_rich_sample_pdfs,
    export_result_to_json,
    export_result_to_csv
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
DB_PATH = os.path.join(DATA_DIR, "documents.db")


# =============================================================================
# SYSTEM INITIALIZATION & CACHING
# =============================================================================

@st.cache_resource(show_spinner=False)
def initialize_system(full: bool = False):
    """Initializes SQLite repository, storage folders, models, and sample PDFs."""
    # Ensure sample PDFs exist
    generate_rich_sample_pdfs(SAMPLES_DIR)

    db = get_db(DB_PATH)
    storage = get_storage_manager(STORAGE_DIR)

    # Initialize Classifier
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

    eval_results = None
    if test_texts and test_labels and clf.is_trained:
        try:
            eval_results = ModelEvaluator.evaluate_all_models(clf, test_texts, test_labels)
        except Exception:
            eval_results = None

    ocr_engine = OCRProcessor()
    processor = DocumentProcessor(db_manager=db, storage_manager=storage, classifier=clf, ocr_engine=ocr_engine)

    if full:
        return db, storage, clf, ocr_engine, processor, eval_results
    return clf, ocr_engine, eval_results


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
    Direct document processing pipeline helper (Week 3 compatibility).
    """
    from src.document_reader import DocumentReader
    from src.text_cleaner import TextCleaner
    from src.extractor import DocumentExtractor

    ext = file_extension.lower().lstrip(".")
    stages = ["Document Uploaded"]

    raw_text = ""
    ocr_used = False
    ocr_steps = []
    extraction_note = ""

    if ext == "pdf":
        read_res = DocumentReader.read_pdf(file_bytes)
        if read_res["error"]:
            return {"error": read_res["error"], "stages": stages}
        if read_res["needs_ocr"]:
            ocr_res = ocr_engine.ocr_pdf(file_bytes, preprocess=ocr_settings.get("preprocess", True))
            raw_text = ocr_res["text"]
            ocr_used = True
            ocr_steps = ocr_res.get("preprocessing_steps", [])
            extraction_note = "Scanned PDF detected — executed OCR fallback."
        else:
            raw_text = read_res["text"]
            extraction_note = f"Direct PyMuPDF selectable text extracted ({read_res['page_count']} page(s))."
    elif ext in ("jpg", "jpeg", "png"):
        ocr_res = ocr_engine.extract_from_image(file_bytes, preprocess=ocr_settings.get("preprocess", True))
        raw_text = ocr_res["text"]
        ocr_used = True
        ocr_steps = ocr_res.get("preprocessing_steps", [])
        extraction_note = "Image processed via OCR engine."
    else:
        return {"error": "Unsupported file format. Please upload a PDF, JPG, JPEG, or PNG.", "stages": stages}

    stages.append("Text Extracted")
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
            "error": clean_res.get("warning") or "Unable to extract readable text from this document."
        }

    pred_res = classifier.predict(cleaned_text, model_name=active_model_name)
    doc_type = pred_res["document_type"]
    confidence = pred_res["confidence"]
    stages.append("Document Classified")

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
# UI COMPONENTS & STYLING
# =============================================================================

def apply_custom_css():
    st.markdown("""
        <style>
        /* Modern Typography & Container */
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        .main-subtitle {
            font-size: 1.0rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        /* Metric Card styling */
        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 0.75rem;
        }
        .kpi-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .kpi-value {
            font-size: 2.0rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
        }
        /* Status Badges */
        .badge-processed {
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-block;
        }
        .badge-review {
            background-color: #FFFBEB;
            color: #92400E;
            border: 1px solid #FDE68A;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-block;
        }
        .badge-failed {
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-block;
        }
        .badge-type {
            background-color: #EEF2FF;
            color: #3730A3;
            border: 1px solid #C7D2FE;
            padding: 0.22rem 0.6rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-block;
        }
        /* Pipeline Stage Pill */
        .stage-pill {
            display: inline-flex;
            align-items: center;
            background-color: #F8FAFC;
            color: #1E293B;
            border: 1px solid #E2E8F0;
            padding: 0.3rem 0.7rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .stage-pill.done {
            background-color: #ECFDF5;
            color: #065F46;
            border-color: #A7F3D0;
        }
        /* Callout Box */
        .dup-box {
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            padding: 1.0rem 1.2rem;
            border-radius: 6px;
            margin-bottom: 1.2rem;
        }
        </style>
    """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    """Formats HTML badge for document status."""
    if status == STATUS_PROCESSED:
        return f'<span class="badge-processed">✓ {status}</span>'
    elif status == STATUS_NEEDS_REVIEW:
        return f'<span class="badge-review">⚠ {status}</span>'
    else:
        return f'<span class="badge-failed">✕ {status}</span>'


def render_type_badge(doc_type: str) -> str:
    """Formats HTML badge for document category."""
    return f'<span class="badge-type">{doc_type}</span>'


# =============================================================================
# VIEW: 🏠 DASHBOARD
# =============================================================================

def render_dashboard(db: DatabaseManager, storage: FileStorageManager):
    st.markdown('<div class="main-title">Document Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Real-time repository health, volume metrics, and recent document streams</div>', unsafe_allow_html=True)

    stats = db.get_statistics()

    # Top Row: Volume KPI Counters
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Documents</div>
                <div class="kpi-value">{stats['total']}</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Invoices Processed</div>
                <div class="kpi-value">{stats['invoices']}</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Resumes Parsed</div>
                <div class="kpi-value">{stats['resumes']}</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Other Documents</div>
                <div class="kpi-value">{stats['other']}</div>
            </div>
        """, unsafe_allow_html=True)

    # Second Row: Processing Status Counters
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #10B981;">
                <div class="kpi-title">🟢 Processed Cleanly</div>
                <div class="kpi-value" style="color: #065F46;">{stats['processed']}</div>
            </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #F59E0B;">
                <div class="kpi-title">🟡 Needs Human Review</div>
                <div class="kpi-value" style="color: #92400E;">{stats['needs_review']}</div>
            </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #EF4444;">
                <div class="kpi-title">🔴 Failed / Unreadable</div>
                <div class="kpi-value" style="color: #991B1B;">{stats['failed']}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🕒 Recently Ingested Documents")

    recent = stats.get("recent_documents", [])
    if not recent:
        st.info("The repository is currently empty. Head over to the **📤 Upload Document** section to ingest your first file!")
    else:
        # Build clean interactive dataframe
        table_rows = []
        for d in recent:
            # Determine summary entity
            entity = d.get("company") if d["document_type"] == "Invoice" else d.get("candidate_name")
            if not entity or entity == "Not Found":
                entity = d.get("invoice_number", "-")

            table_rows.append({
                "ID": d["id"],
                "Original File": d["original_filename"],
                "Type": d["document_type"],
                "Status": d["status"],
                "Key Entity": entity,
                "Uploaded At": d["upload_date"][:19].replace("T", " ")
            })
        df_recent = pd.DataFrame(table_rows)
        st.dataframe(df_recent, use_container_width=True, hide_index=True)

        st.caption("Tip: Select **🔎 Search Documents** or **📁 Document Repository** to inspect complete metadata, text previews, and download stored files.")


# =============================================================================
# VIEW: 📤 UPLOAD DOCUMENT
# =============================================================================

def render_upload_page(
    db: DatabaseManager,
    storage: FileStorageManager,
    processor: DocumentProcessor,
    classifier: DocumentClassifier
):
    st.markdown('<div class="main-title">Upload & Ingest Document</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Automated validation, cryptographic deduplication, text extraction, ML classification & persistence</div>', unsafe_allow_html=True)

    # Sidebar or top option to load pre-built sample files
    col_upload, col_sample = st.columns([3, 2])

    uploaded_file = None
    sample_file_bytes = None
    sample_filename = None

    with col_upload:
        st.markdown("##### 📁 Upload Document from Computer")
        uploaded_file = st.file_uploader(
            "Choose a PDF or Image file",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Supported formats: PDF, JPG, JPEG, PNG. Maximum size: 20MB."
        )

    with col_sample:
        st.markdown("##### 🧪 Or Test Pre-loaded Sample Files")
        sample_choice = st.selectbox(
            "Select sample file:",
            [
                "-- Select a sample document --",
                "invoice_1.pdf (TechCorp Solutions, $1450)",
                "invoice_2.pdf (Apex Retailers, Rs. 78500)",
                "invoice_missing_total.pdf (Missing Total Field)",
                "resume_1.pdf (Alex Smith, AI Engineer)",
                "resume_missing_phone.pdf (Missing Phone Field)",
                "meeting_minutes.pdf (Other Category)"
            ]
        )
        if sample_choice != "-- Select a sample document --":
            sample_filename = sample_choice.split(" ")[0]
            sample_path = os.path.join(SAMPLES_DIR, sample_filename)
            if os.path.isfile(sample_path):
                with open(sample_path, "rb") as f:
                    sample_file_bytes = f.read()
                st.success(f"Loaded `{sample_filename}` ({len(sample_file_bytes)} bytes)")

    # Advanced Processing Configurations
    with st.expander("⚙️ Advanced Pipeline Configurations", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            active_model = st.selectbox(
                "Active Classification Model:",
                ["Logistic Regression", "Linear SVM", "Naive Bayes", "Rule-Based Baseline"],
                index=0
            )
        with c2:
            apply_preprocess = st.checkbox(
                "Apply OCR Image Preprocessing (Grayscale + Thresholding + Denoise)",
                value=True
            )

    # Process Document Button
    file_to_process = None
    filename_to_process = None

    if uploaded_file is not None:
        file_to_process = uploaded_file.getvalue()
        filename_to_process = uploaded_file.name
    elif sample_file_bytes is not None:
        file_to_process = sample_file_bytes
        filename_to_process = sample_filename

    if file_to_process:
        st.markdown("---")
        process_btn = st.button("🚀 Ingest & Process Document", type="primary", use_container_width=True)

        if process_btn:
            with st.spinner("Processing document through AI intelligence pipeline..."):
                result = processor.process_and_store(
                    file_bytes=file_to_process,
                    original_filename=filename_to_process,
                    active_model_name=active_model,
                    ocr_settings={"preprocess": apply_preprocess}
                )

            # Check for Duplicate
            if result.get("is_duplicate"):
                existing = result["document"]
                st.markdown(f"""
                    <div class="dup-box">
                        <h4 style="color: #B45309; margin-top: 0;">⚠️ Duplicate Document Suppressed</h4>
                        <p>{result['message']}</p>
                        <p><strong>SHA-256 Digest:</strong> <code>{result['file_hash']}</code></p>
                    </div>
                """, unsafe_allow_html=True)

                st.subheader("Existing Document Record in Repository")
                render_single_document_details(existing, storage, db)
                return

            if not result.get("success"):
                st.error(f"Processing Failed: {result.get('error')}")
                if "document" in result and result["document"]:
                    st.warning("Document was recorded in SQLite as 'Failed' for audit logging.")
                return

            # Display Pipeline Success Stages
            st.success("🎉 Document successfully processed and committed to repository!")
            st.markdown("##### 📌 Workflow Execution Trail:")
            stage_html = " ".join([f'<span class="stage-pill done">✓ {s}</span>' for s in result.get("stages", [])])
            st.markdown(stage_html, unsafe_allow_html=True)

            st.markdown("---")
            doc = result["document"]
            render_single_document_details(doc, storage, db, live_analysis=result)


# =============================================================================
# VIEW: 🔎 SEARCH & FILTER DOCUMENTS
# =============================================================================

def render_search_page(db: DatabaseManager, storage: FileStorageManager):
    st.markdown('<div class="main-title">Search & Filter Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Fast, multi-field parameterized discovery across metadata, text content, and extracted entities</div>', unsafe_allow_html=True)

    # Search Box & Filters
    with st.container():
        query = st.text_input(
            "🔍 Global Document Search:",
            placeholder="Search by filename, company, invoice number, candidate, or full text...",
            help="Queries SQLite using parameterized LIKE indexing."
        )

        f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
        with f1:
            doc_type_filter = st.selectbox("Document Category:", ["All"] + ALL_DOCUMENT_TYPES, index=0)
        with f2:
            status_filter = st.selectbox("Processing Status:", ["All"] + ALL_STATUSES, index=0)
        with f3:
            sort_order = st.selectbox("Sort Order:", ["Newest First", "Oldest First"], index=0)
        with f4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset Filters", use_container_width=True):
                st.rerun()

    sql_sort = "DESC" if "Newest" in sort_order else "ASC"

    # Execute search query
    results = db.search_documents(
        search_query=query,
        doc_type=doc_type_filter,
        status=status_filter,
        sort_order=sql_sort,
        limit=100
    )

    st.markdown(f"**Found {len(results)} matching document(s)**")

    if not results:
        st.warning("No documents matched your criteria. Try adjusting your query or resetting the filters.")
        return

    # Display results as structured cards
    for doc in results:
        with st.container():
            st.markdown(f"""
                <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; background: #FFFFFF;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">
                            #{doc['id']} — {doc['original_filename']}
                        </span>
                        <div>
                            {render_type_badge(doc['document_type'])}
                            {render_status_badge(doc['status'])}
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.5rem;">
                        <strong>Uploaded:</strong> {doc['upload_date'][:19].replace('T', ' ')} | 
                        <strong>Stored Filename:</strong> <code>{doc['stored_filename']}</code>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            col_details, col_actions = st.columns([4, 1])
            with col_details:
                # Key extracted highlights
                if doc["document_type"] == TYPE_INVOICE:
                    st.markdown(f"**Company:** `{doc['company']}` | **Invoice #:** `{doc['invoice_number']}` | **Total:** `{doc['total_amount']}`")
                elif doc["document_type"] == TYPE_RESUME:
                    st.markdown(f"**Candidate:** `{doc['candidate_name']}` | **Email:** `{doc['candidate_email']}` | **Phone:** `{doc['candidate_phone']}`")
                else:
                    st.markdown(f"**Text Snippet:** *{doc['text_preview'][:160]}...*")

            with col_actions:
                if st.button("Inspect Details", key=f"btn_inspect_{doc['id']}", use_container_width=True):
                    st.session_state["selected_doc_id"] = doc["id"]

    # If a document is selected, render its full details panel
    if st.session_state.get("selected_doc_id"):
        st.markdown("---")
        st.subheader(f"📑 Document Inspection Panel (ID #{st.session_state['selected_doc_id']})")
        selected_record = db.get_document_by_id(st.session_state["selected_doc_id"])
        if selected_record:
            render_single_document_details(selected_record, storage, db)
            if st.button("Close Inspection", use_container_width=False):
                st.session_state["selected_doc_id"] = None
                st.rerun()


# =============================================================================
# VIEW: 📁 DOCUMENT REPOSITORY
# =============================================================================

def render_repository_page(db: DatabaseManager, storage: FileStorageManager):
    st.markdown('<div class="main-title">Document Repository</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Centralized document inventory, metadata inspection, and download vault</div>', unsafe_allow_html=True)

    all_docs = db.get_all_documents(sort_order="DESC")

    if not all_docs:
        st.info("No documents are currently stored in the repository.")
        return

    # Tabs for fast category filtering
    tab_all, tab_inv, tab_res, tab_oth, tab_review = st.tabs([
        f"All Documents ({len(all_docs)})",
        f"Invoices ({sum(1 for d in all_docs if d['document_type'] == 'Invoice')})",
        f"Resumes ({sum(1 for d in all_docs if d['document_type'] == 'Resume')})",
        f"Other ({sum(1 for d in all_docs if d['document_type'] not in ('Invoice', 'Resume'))})",
        f"Needs Review ({sum(1 for d in all_docs if d['status'] == STATUS_NEEDS_REVIEW)})"
    ])

    def render_repo_table(docs_subset: List[Dict[str, Any]], tab_key: str):
        if not docs_subset:
            st.info("No documents match this category.")
            return

        rows = []
        for d in docs_subset:
            rows.append({
                "ID": d["id"],
                "Filename": d["original_filename"],
                "Category": d["document_type"],
                "Status": d["status"],
                "Company / Candidate": d["company"] if d["document_type"] == "Invoice" else d["candidate_name"],
                "Invoice # / Email": d["invoice_number"] if d["document_type"] == "Invoice" else d["candidate_email"],
                "Amount / Phone": d["total_amount"] if d["document_type"] == "Invoice" else d["candidate_phone"],
                "Uploaded Date": d["upload_date"][:19].replace("T", " ")
            })
        df_sub = pd.DataFrame(rows)
        st.dataframe(df_sub, use_container_width=True, hide_index=True)

        # Quick Inspection Selector
        doc_ids = [d["id"] for d in docs_subset]
        selected_id = st.selectbox(
            "Select Document ID to View / Download:",
            options=doc_ids,
            key=f"select_doc_{tab_key}",
            format_func=lambda x: f"ID #{x} — {next((d['original_filename'] for d in docs_subset if d['id'] == x), '')}"
        )
        if selected_id:
            doc_record = db.get_document_by_id(selected_id)
            if doc_record:
                st.markdown("---")
                render_single_document_details(doc_record, storage, db)

    with tab_all:
        render_repo_table(all_docs, "all")
    with tab_inv:
        render_repo_table([d for d in all_docs if d["document_type"] == "Invoice"], "inv")
    with tab_res:
        render_repo_table([d for d in all_docs if d["document_type"] == "Resume"], "res")
    with tab_oth:
        render_repo_table([d for d in all_docs if d["document_type"] not in ("Invoice", "Resume")], "oth")
    with tab_review:
        render_repo_table([d for d in all_docs if d["status"] == STATUS_NEEDS_REVIEW], "rev")


# =============================================================================
# VIEW: 📊 ANALYTICS & BENCHMARK
# =============================================================================

def render_analytics_page(db: DatabaseManager, eval_results: Optional[Dict[str, Any]]):
    st.markdown('<div class="main-title">Analytics & Machine Learning Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Repository distribution breakdown, model evaluation metrics, and confusion matrix</div>', unsafe_allow_html=True)

    stats = db.get_statistics()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Category Distribution")
        cat_data = pd.DataFrame({
            "Category": ["Invoices", "Resumes", "Other"],
            "Count": [stats["invoices"], stats["resumes"], stats["other"]]
        })
        st.bar_chart(cat_data.set_index("Category"))

    with c2:
        st.subheader("🚦 Processing Health Status")
        status_data = pd.DataFrame({
            "Status": ["Processed", "Needs Review", "Failed"],
            "Count": [stats["processed"], stats["needs_review"], stats["failed"]]
        })
        st.bar_chart(status_data.set_index("Status"))

    st.markdown("---")
    st.subheader("🧪 Machine Learning Classifier Evaluation (Test Corpus)")

    if eval_results and "comparison_table" in eval_results:
        df_comp = pd.DataFrame(eval_results["comparison_table"])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown("##### 3x3 Multi-Class Confusion Matrix (Test Partition)")
        best_name = eval_results.get("best_model_name", "Logistic Regression")
        if best_name in eval_results.get("detailed_metrics", {}):
            cm = eval_results["detailed_metrics"][best_name]["confusion_matrix"]
            classes = eval_results["classes"]
            df_cm = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
            st.table(df_cm)

        if "diagnostics" in eval_results:
            diag = eval_results["diagnostics"]
            with st.expander("🔍 Scientific Diagnostics & Error Analysis", expanded=True):
                st.markdown(f"**Top Model Strengths:** {diag.get('strengths')}")
                st.markdown(f"**Confusion Patterns:** {diag.get('confusion_patterns')}")
                st.markdown(f"**Error Root Causes:** {diag.get('causes_of_errors')}")
                st.markdown(f"**Dataset Boundaries:** {diag.get('dataset_limitations')}")
    else:
        st.info("Evaluation benchmark metrics not available. Ensure test dataset is present in `data/test/`.")


# =============================================================================
# VIEW: ⚙️ SETTINGS & HEALTH
# =============================================================================

def render_settings_page(db: DatabaseManager, storage: FileStorageManager, ocr_engine: OCRProcessor):
    st.markdown('<div class="main-title">System Settings & Health</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Storage directory inspection, database health, and OCR discovery status</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🗄️ SQLite Database Health")
        st.write(f"**Database Path:** `{db.db_path}`")
        if os.path.isfile(db.db_path):
            size_kb = round(os.path.getsize(db.db_path) / 1024, 2)
            st.write(f"**Database File Size:** `{size_kb} KB`")
            st.write(f"**WAL Journal:** Enabled (`journal_mode=WAL`)")
        else:
            st.warning("Database file not yet created.")

        st.markdown("##### 📁 Storage Vault Health")
        st.write(f"**Storage Directory:** `{storage.storage_base}`")
        for sub in ("invoices", "resumes", "other"):
            sub_path = os.path.join(storage.storage_base, sub)
            count = len(os.listdir(sub_path)) if os.path.isdir(sub_path) else 0
            st.write(f"- `storage/{sub}/`: **{count}** files stored")

    with c2:
        st.markdown("##### 🔍 OCR Engine Discovery")
        if ocr_engine.tesseract_available:
            st.success("✓ Tesseract OCR engine discovered and operational on host system.")
        else:
            st.warning("⚠ Tesseract OCR engine was not auto-detected on PATH. Native PDF digital extraction is fully functional; install Tesseract to enable scanned image OCR.")

        st.markdown("##### 🧹 Maintenance Actions")
        if st.button("Delete All Database Records & Storage (Reset Repository)"):
            all_docs = db.get_all_documents()
            for d in all_docs:
                storage.delete_file(d["file_path"])
                db.delete_document(d["id"])
            st.success("Repository and storage vault cleared successfully.")
            st.rerun()


# =============================================================================
# REUSABLE COMPONENT: DOCUMENT DETAIL PANEL
# =============================================================================

def render_single_document_details(
    doc: Dict[str, Any],
    storage: FileStorageManager,
    db: DatabaseManager,
    live_analysis: Optional[Dict[str, Any]] = None
):
    """Renders a comprehensive, SaaS-grade document detail view."""
    st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 10px; padding: 1.2rem; margin-bottom: 1.0rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #0F172A;">{doc['original_filename']}</h3>
                <div>
                    {render_type_badge(doc['document_type'])}
                    {render_status_badge(doc['status'])}
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.35rem;">
                <strong>Document ID:</strong> #{doc['id']} | 
                <strong>Stored Filename:</strong> <code>{doc['stored_filename']}</code> | 
                <strong>Uploaded:</strong> {doc['upload_date'][:19].replace('T', ' ')}
            </div>
        </div>
    """, unsafe_allow_html=True)

    if doc.get("status_reason"):
        if doc["status"] == STATUS_NEEDS_REVIEW:
            st.warning(f"⚠️ **Review Advisory:** {doc['status_reason']}")
        elif doc["status"] == STATUS_FAILED:
            st.error(f"✕ **Failure Reason:** {doc['status_reason']}")

    col_meta, col_fields = st.columns([1, 1])

    with col_meta:
        st.markdown("##### 📌 Cryptographic & Storage Metadata")
        st.write(f"**SHA-256 Digest:**")
        st.code(doc["file_hash"], language="text")
        st.write(f"**Storage Path:** `{doc['file_path']}`")

        metadata = doc.get("metadata", {})
        if metadata:
            st.write(f"**Model Used:** `{metadata.get('classification_model', 'N/A')}`")
            st.write(f"**Classifier Confidence:** `{metadata.get('confidence', 'N/A')}`")
            st.write(f"**OCR Fallback Used:** `{'Yes' if metadata.get('ocr_used') else 'No'}`")
            st.write(f"**Word Count:** `{metadata.get('word_count', 0)} words`")

        # Download Stored File
        file_bytes = storage.read_file(doc["file_path"])
        if file_bytes:
            st.download_button(
                label=f"📥 Download Stored File ({doc['original_filename']})",
                data=file_bytes,
                file_name=doc["original_filename"],
                mime="application/pdf" if doc["original_filename"].lower().endswith(".pdf") else "image/png",
                use_container_width=True
            )

    with col_fields:
        st.markdown("##### 📋 Extracted Structured Fields")
        if doc["document_type"] == TYPE_INVOICE:
            fields_data = [
                ("Invoice Number", doc.get("invoice_number", "Not Found")),
                ("Company Name", doc.get("company", "Not Found")),
                ("Total Amount", doc.get("total_amount", "Not Found")),
                ("Date", metadata.get("fields", {}).get("Date", "Not Found"))
            ]
        elif doc["document_type"] == TYPE_RESUME:
            fields_data = [
                ("Candidate Name", doc.get("candidate_name", "Not Found")),
                ("Email Address", doc.get("candidate_email", "Not Found")),
                ("Phone Number", doc.get("candidate_phone", "Not Found")),
                ("Skills", doc.get("candidate_skills", "Not Found"))
            ]
        else:
            fields_data = [
                ("Category Notice", "Document classified as 'Other'. Field extraction is targeted for Invoices and Resumes.")
            ]

        for k, v in fields_data:
            val_style = "color: #0F172A; font-weight: 600;"
            if v == "Not Found":
                val_style = "color: #94A3B8; font-style: italic;"
            st.markdown(f"**{k}:** <span style='{val_style}'>{v}</span>", unsafe_allow_html=True)

    # Text Preview Expander
    with st.expander("📄 Document Text Preview (First 1,000 Characters)", expanded=False):
        st.text_area("Extracted Clean Text", doc.get("text_preview", ""), height=200, disabled=True)

    # Actions Row: Delete or Change Status
    c_act1, c_act2 = st.columns([1, 1])
    with c_act1:
        new_status = st.selectbox(
            "Change Status:",
            ALL_STATUSES,
            index=ALL_STATUSES.index(doc["status"]),
            key=f"status_select_{doc['id']}"
        )
        if new_status != doc["status"]:
            db.update_document_status(doc["id"], new_status, "Manually updated by reviewer.")
            st.success(f"Status updated to '{new_status}'")
            st.rerun()

    with c_act2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Delete Document Record", key=f"del_doc_{doc['id']}", type="secondary"):
            deleted_record = db.delete_document(doc["id"])
            if deleted_record:
                storage.delete_file(deleted_record["file_path"])
                st.success("Document and stored file removed cleanly from repository.")
                st.rerun()


# =============================================================================
# MAIN CONTROLLER & NAVIGATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI Document Intelligence & Workflow Platform",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_custom_css()

    # Initialize backend
    db, storage, clf, ocr_engine, processor, eval_results = initialize_system(full=True)

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 📄 AI Document Intelligence")
        st.caption("Enterprise Document Workflow Platform • Week 4")
        st.markdown("---")

        page = st.radio(
            "Navigation:",
            [
                "🏠 Dashboard",
                "📤 Upload Document",
                "🔎 Search Documents",
                "📁 Document Repository",
                "📊 Analytics & Benchmark",
                "⚙️ Settings & Health"
            ],
            index=0
        )

        st.markdown("---")
        stats = db.get_statistics()
        st.markdown(f"**Repository Vault:** `{stats['total']}` files")
        st.markdown(f"**Cleanly Processed:** `{stats['processed']}`")
        st.markdown(f"**Needs Review:** `{stats['needs_review']}`")
        st.markdown("---")
        st.caption("AI/ML Document Intelligence Platform\nDeveloped by Sakthibalan S")

    # Route Page
    if page == "🏠 Dashboard":
        render_dashboard(db, storage)
    elif page == "📤 Upload Document":
        render_upload_page(db, storage, processor, clf)
    elif page == "🔎 Search Documents":
        render_search_page(db, storage)
    elif page == "📁 Document Repository":
        render_repository_page(db, storage)
    elif page == "📊 Analytics & Benchmark":
        render_analytics_page(db, eval_results)
    elif page == "⚙️ Settings & Health":
        render_settings_page(db, storage, ocr_engine)


if __name__ == "__main__":
    main()
