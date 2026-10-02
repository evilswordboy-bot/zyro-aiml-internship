"""
AI Document Intelligence & Workflow Platform — Week 5
Advanced Document Workflow & Automation Platform
Controlled State Machine • Validation Engine • Rule-Based Routing
Human Review Queue • Audit History • Fault-Tolerant Batch Processing
"""

import os
import io
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import pandas as pd
from PIL import Image

# Core Services & Database
from database.db import DatabaseManager, get_db
from models.document import (
    STATUS_NEW,
    STATUS_PROCESSING,
    STATUS_NEEDS_REVIEW,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_COMPLETED,
    STATUS_PROCESSED,
    STATUS_FAILED,
    WORKFLOW_STATUSES,
    ALL_STATUSES,
    TYPE_INVOICE,
    TYPE_RESUME,
    TYPE_OTHER,
    ALL_DOCUMENT_TYPES
)
from services.file_storage import FileStorageManager, get_storage_manager
from services.validation import FileValidator, DocumentValidator
from services.audit import (
    AuditService,
    get_audit_service,
    ACTION_UPLOADED,
    ACTION_PROCESSING_STARTED,
    ACTION_CLASSIFIED,
    ACTION_EXTRACTED,
    ACTION_VALIDATION_PASSED,
    ACTION_VALIDATION_FAILED,
    ACTION_ROUTED_TO_REVIEW,
    ACTION_ROUTED_TO_COMPLETED,
    ACTION_REVIEWER_APPROVED,
    ACTION_REVIEWER_REJECTED,
    ACTION_WORKFLOW_COMPLETED,
    ACTION_WORKFLOW_FAILED,
    ACTION_RETRIED
)
from services.workflow import WorkflowEngine, get_workflow_engine, CONFIDENCE_THRESHOLD
from services.document_processor import DocumentProcessor, get_document_processor
from services.batch import BatchProcessor, get_batch_processor
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
    """Initializes SQLite repository, storage, models, sample PDFs, and services."""
    generate_rich_sample_pdfs(SAMPLES_DIR)

    db = get_db(DB_PATH)
    storage = get_storage_manager(STORAGE_DIR)
    audit = get_audit_service(db)
    workflow = get_workflow_engine(db, audit)

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
    processor = DocumentProcessor(
        db_manager=db,
        storage_manager=storage,
        classifier=clf,
        ocr_engine=ocr_engine,
        audit_service=audit,
        workflow_engine=workflow
    )
    batch_proc = BatchProcessor(processor=processor)

    if full:
        return db, storage, clf, ocr_engine, processor, batch_proc, workflow, audit, eval_results
    return clf, ocr_engine, eval_results


def process_document(
    file_bytes: bytes,
    filename: str,
    file_extension: str,
    classifier: DocumentClassifier,
    ocr_engine: OCRProcessor,
    active_model_name: str = "Logistic Regression",
    ocr_settings: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """
    Direct document processing pipeline helper for in-memory analysis without DB persistence.
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
            return {
                "success": False,
                "error": read_res["error"],
                "stages": stages + ["Failed"]
            }
        if read_res["needs_ocr"]:
            ocr_res = ocr_engine.ocr_pdf(file_bytes, preprocess=True)
            raw_text = ocr_res["text"]
            ocr_used = True
            ocr_steps = ocr_res.get("preprocessing_steps", [])
            extraction_note = "Scanned PDF processed via OCR fallback."
        else:
            raw_text = read_res["text"]
            extraction_note = f"Direct PyMuPDF selectable text ({read_res['page_count']} page(s))."
    else:
        ocr_res = ocr_engine.extract_from_image(file_bytes, preprocess=True)
        raw_text = ocr_res["text"]
        ocr_used = True
        ocr_steps = ocr_res.get("preprocessing_steps", [])
        extraction_note = "Image processed via OCR computer vision engine."

    stages.append("Text Extracted")

    clean_res = TextCleaner.clean(raw_text)
    cleaned_text = clean_res["cleaned_text"]
    stages.append("Text Cleaned")

    pred_res = classifier.predict(cleaned_text, model_name=active_model_name)
    doc_type = pred_res["document_type"]
    confidence = pred_res["confidence"]
    stages.append("Document Classified")

    extract_res = DocumentExtractor.extract(cleaned_text, doc_type)
    fields = extract_res.get("fields", {})
    missing_fields = extract_res.get("missing_fields", [])
    stages.append("Fields Extracted")

    stages.append("Pipeline Complete")

    return {
        "success": True,
        "filename": filename,
        "file_type": ext.upper(),
        "document_type": doc_type,
        "confidence": confidence,
        "classification_model": active_model_name,
        "ocr_used": ocr_used,
        "ocr_steps": ocr_steps,
        "extraction_note": extraction_note,
        "fields": fields,
        "missing_fields": missing_fields,
        "original_text": raw_text,
        "cleaned_text": cleaned_text,
        "stages": stages
    }


# =============================================================================
# UI STYLING & CUSTOM CSS
# =============================================================================

def apply_custom_css():
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.2rem;
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
        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 0.75rem;
        }
        .kpi-title {
            font-size: 0.75rem;
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
        .badge-completed {
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
        .badge-approved {
            background-color: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-block;
        }
        .badge-rejected {
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            display: inline-block;
        }
        .badge-processing {
            background-color: #FAF5FF;
            color: #6B21A8;
            border: 1px solid #E9D5FF;
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
        .audit-item {
            border-left: 3px solid #3B82F6;
            padding: 0.5rem 0.8rem;
            margin-bottom: 0.5rem;
            background-color: #F8FAFC;
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
        }
        .review-box {
            border: 1px solid #FDE68A;
            background-color: #FFFDF5;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1.2rem;
        }
        .warning-pill {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.3rem;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    s = (status or "").strip()
    if s in (STATUS_COMPLETED, STATUS_PROCESSED):
        return f'<span class="badge-completed">✓ {s}</span>'
    elif s == STATUS_NEEDS_REVIEW:
        return f'<span class="badge-review">⚠️ Needs Review</span>'
    elif s == STATUS_APPROVED:
        return f'<span class="badge-approved">✓ Approved</span>'
    elif s == STATUS_REJECTED:
        return f'<span class="badge-rejected">✕ Rejected</span>'
    elif s == STATUS_PROCESSING:
        return f'<span class="badge-processing">⟳ Processing</span>'
    elif s == STATUS_NEW:
        return f'<span class="stage-pill">● New</span>'
    else:
        return f'<span class="badge-failed">✕ Failed</span>'


def render_confidence_badge(conf: Any) -> str:
    if conf is None or conf == "Not Available" or conf == "":
        return '<span style="color: #64748B; font-weight: 600; font-size: 0.8rem;">Confidence: Not Available</span>'
    if isinstance(conf, float):
        pct = round(conf * 100)
    elif isinstance(conf, str) and "%" in conf:
        pct = int(conf.replace("%", "").strip())
    else:
        try:
            pct = round(float(conf) * 100)
        except Exception:
            return f'<span style="color: #64748B; font-weight: 600;">Confidence: {conf}</span>'

    color = "#059669" if pct >= 70 else "#D97706"
    return f'<span style="color: {color}; font-weight: 700; font-size: 0.85rem;">Confidence: {pct}%</span>'


# =============================================================================
# VIEW 1: 🏠 DASHBOARD & KPI OVERVIEW
# =============================================================================

def render_dashboard(db: DatabaseManager, storage: FileStorageManager, workflow: WorkflowEngine):
    st.markdown('<div class="main-title">AI Document Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Automated Intake • Controlled State Transitions • Human-in-the-Loop Review • Complete Audit History</div>', unsafe_allow_html=True)

    stats = db.get_statistics()

    # KPI Metric Cards Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Ingested</div>
                <div class="kpi-value">{stats['total']}</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title" style="color: #059669;">Completed</div>
                <div class="kpi-value" style="color: #059669;">{stats['completed']}</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        rev_color = "#D97706" if stats['needs_review'] > 0 else "#64748B"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title" style="color: {rev_color};">Needs Review</div>
                <div class="kpi-value" style="color: {rev_color};">{stats['needs_review']}</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title" style="color: #2563EB;">Approved</div>
                <div class="kpi-value" style="color: #2563EB;">{stats['approved']}</div>
            </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title" style="color: #DC2626;">Rejected</div>
                <div class="kpi-value" style="color: #DC2626;">{stats['rejected']}</div>
            </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title" style="color: #475569;">Audit Events</div>
                <div class="kpi-value" style="color: #475569;">{stats['total_audit_events']}</div>
            </div>
        """, unsafe_allow_html=True)

    # Human Review Callout Banner if items are pending review
    if stats['needs_review'] > 0:
        st.markdown(f"""
            <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 1rem 1.2rem; margin: 1rem 0 1.5rem 0;">
                <h4 style="color: #92400E; margin: 0 0 0.4rem 0;">⚠️ Human Review Required ({stats['needs_review']} document(s) pending)</h4>
                <p style="color: #78350F; margin: 0;">Documents with missing mandatory fields or low classification confidence require manual verification.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("👉 Open Human Review Queue", type="primary"):
            st.session_state["nav_selection"] = "⚖️ Human Review Queue"
            st.rerun()

    # Visual Workflow Status & Category Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### 📊 Document Workflow Status Breakdown")
        status_data = {
            "Status": ["Completed", "Needs Review", "Approved", "Rejected", "Failed"],
            "Count": [stats["completed"], stats["needs_review"], stats["approved"], stats["rejected"], stats["failed"]]
        }
        df_status = pd.DataFrame(status_data)
        st.bar_chart(df_status.set_index("Status"), height=260)

    with col_chart2:
        st.markdown("##### 📁 Document Types Distribution")
        cat_data = {
            "Category": ["Invoice", "Resume", "Other"],
            "Count": [stats["invoices"], stats["resumes"], stats["other"]]
        }
        df_cat = pd.DataFrame(cat_data)
        st.bar_chart(df_cat.set_index("Category"), height=260)

    # Live Audit Event Feed
    st.markdown("---")
    st.markdown("##### ⏱️ Live System Audit Activity Feed (Recent Workflow Transitions)")
    recent_events = db.get_recent_audit_events(limit=8)
    if recent_events:
        event_rows = []
        for e in recent_events:
            event_rows.append({
                "Timestamp": e["timestamp"][:19].replace("T", " "),
                "Doc ID": f"#{e['document_id']}",
                "Filename": e.get("original_filename") or "—",
                "Action": e["action"],
                "From": e.get("previous_status") or "—",
                "To": e["new_status"],
                "Reason / Note": e.get("reviewer_note") or e.get("reason") or "—"
            })
        st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No audit events recorded yet. Ingest documents to view the live audit activity feed.")


# =============================================================================
# VIEW 2: 📤 INGEST & WORKFLOW EXECUTION (SINGLE DOCUMENT)
# =============================================================================

def render_upload_page(
    db: DatabaseManager,
    storage: FileStorageManager,
    processor: DocumentProcessor,
    clf: DocumentClassifier
):
    st.markdown('<div class="main-title">Ingest & Document Workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Automated Intake • Deduplication • OCR & Text Extraction • ML Classification • Advanced Validation • Rule Routing</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    file_to_process = None
    filename_to_process = ""

    with c1:
        st.markdown("##### 📁 Option A: Upload Custom Document")
        uploaded_file = st.file_uploader(
            "Select PDF or Image (PNG, JPG, JPEG):",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Files are verified against whitelisted formats and SHA-256 deduplicated."
        )
        if uploaded_file is not None:
            file_to_process = uploaded_file.getvalue()
            filename_to_process = uploaded_file.name

    with c2:
        st.markdown("##### 🧪 Option B: Load Rich Test Sample")
        sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".pdf")])
        selected_sample = st.selectbox("Choose pre-built sample document:", ["-- Select Sample --"] + sample_files)
        if selected_sample != "-- Select Sample --":
            sample_path = os.path.join(SAMPLES_DIR, selected_sample)
            with open(sample_path, "rb") as f:
                file_to_process = f.read()
            filename_to_process = selected_sample
            st.info(f"Loaded sample: `{selected_sample}` ({len(file_to_process)} bytes)")

    # Model & OCR Settings
    st.markdown("---")
    st.markdown("##### ⚙️ Pipeline Configuration")
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        active_model = st.selectbox(
            "Classification Engine:",
            ["Logistic Regression", "Linear SVM", "Naive Bayes", "Rule-Based Baseline"],
            index=0,
            help="Logistic Regression & Naive Bayes calculate calibrated confidence probabilities."
        )
    with col_cfg2:
        apply_preprocess = st.checkbox("Enable Adaptive OCR Preprocessing (Binarization & Denoising)", value=True)

    if file_to_process:
        st.markdown("---")
        if st.button("🚀 Ingest & Execute Document Workflow", type="primary", use_container_width=True):
            with st.spinner("Executing end-to-end document intelligence pipeline..."):
                result = processor.process_and_store(
                    file_bytes=file_to_process,
                    original_filename=filename_to_process,
                    active_model_name=active_model,
                    ocr_settings={"preprocess": apply_preprocess}
                )

            # Check for Duplicate
            if result.get("is_duplicate"):
                existing = result["document"]
                st.warning(f"⚠️ Duplicate Suppressed: '{filename_to_process}' is identical to existing Document #{existing['id']}.")
                render_single_document_details(existing, storage, db)
                return

            if not result.get("success"):
                st.error(f"Workflow Processing Failed: {result.get('error')}")
                if "document" in result and result["document"]:
                    st.warning("Document was recorded in SQLite with 'Failed' status and audit log.")
                return

            # Display Pipeline Success Stages
            st.success("🎉 Document workflow successfully executed and recorded!")
            st.markdown("##### 📌 Workflow Execution Trail:")
            stage_html = " ".join([f'<span class="stage-pill done">✓ {s}</span>' for s in result.get("stages", [])])
            st.markdown(stage_html, unsafe_allow_html=True)

            st.markdown("---")
            doc = result["document"]
            render_single_document_details(doc, storage, db, live_analysis=result)


# =============================================================================
# VIEW 3: ⚡ BATCH WORKFLOW PROCESSING
# =============================================================================

def render_batch_page(
    db: DatabaseManager,
    storage: FileStorageManager,
    batch_proc: BatchProcessor,
    clf: DocumentClassifier
):
    st.markdown('<div class="main-title">Batch Document Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">High-Throughput Parallel Ingestion • Isolated Error Boundaries • Non-Blocking Fault Tolerance</div>', unsafe_allow_html=True)

    st.markdown("""
        > **Fault-Tolerant Execution Guarantee:** Batch items execute inside isolated transaction boundaries.
        > An unreadable, corrupt, or invalid file in one item **never halts or halts the batch**.
    """)

    uploaded_files = st.file_uploader(
        "Upload multiple document files (PDF, PNG, JPG, JPEG):",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Select multiple documents for batch ingestion."
    )

    c_opt1, c_opt2 = st.columns([2, 1])
    with c_opt1:
        batch_model = st.selectbox(
            "Batch Classifier:",
            ["Logistic Regression", "Linear SVM", "Naive Bayes", "Rule-Based Baseline"],
            index=0
        )
    with c_opt2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_samples_btn = st.button("📁 Load 6 Sample Test Documents", use_container_width=True)

    files_to_process: List[Tuple[str, bytes]] = []

    if uploaded_files:
        for f in uploaded_files:
            files_to_process.append((f.name, f.getvalue()))

    if load_samples_btn:
        sample_names = [
            "invoice_1.pdf", "invoice_2.pdf", "invoice_missing_total.pdf",
            "resume_1.pdf", "resume_missing_phone.pdf", "meeting_minutes.pdf"
        ]
        files_to_process = []
        for sname in sample_names:
            spath = os.path.join(SAMPLES_DIR, sname)
            if os.path.exists(spath):
                with open(spath, "rb") as f:
                    files_to_process.append((sname, f.read()))
        st.session_state["batch_files_loaded"] = files_to_process
        st.info(f"Loaded {len(files_to_process)} standard test files into batch queue.")

    if "batch_files_loaded" in st.session_state and not uploaded_files:
        files_to_process = st.session_state["batch_files_loaded"]

    if files_to_process:
        st.markdown(f"**Ready to process `{len(files_to_process)}` document(s) in batch.**")
        if st.button("⚡ Start Batch Workflow", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_hook(idx, total, fname, status):
                pct = idx / total
                progress_bar.progress(pct)
                status_text.text(f"Processing ({idx}/{total}): {fname} [{status}]")

            summary = batch_proc.process_batch(
                files=files_to_process,
                active_model_name=batch_model,
                progress_callback=progress_hook
            )

            progress_bar.progress(1.0)
            status_text.text(f"Batch completed! Processed {summary['total']} document(s).")

            # Batch Summary Metrics
            st.markdown("### 📊 Batch Execution Summary")
            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                st.metric("Total Batch Items", summary["total"])
            with b2:
                st.metric("Completed", summary["completed"])
            with b3:
                st.metric("Needs Review", summary["needs_review"])
            with b4:
                st.metric("Failed", summary["failed"])
            with b5:
                st.metric("Duplicates", summary["duplicates"])

            # Itemized Results Table
            st.markdown("##### 📋 Itemized Execution Results")
            item_rows = []
            for r in summary["results"]:
                doc_id = r.get("document_id") or (r.get("document", {}).get("id") if r.get("document") else "—")
                item_rows.append({
                    "Document ID": f"#{doc_id}",
                    "Filename": r.get("filename", "unknown"),
                    "Type": r.get("document_type") or (r.get("document", {}).get("document_type") if r.get("document") else "—"),
                    "Status": r.get("status", "Failed"),
                    "Confidence": r.get("confidence") or "—",
                    "Details / Reason": r.get("status_reason") or r.get("error") or "Passed checks"
                })

            st.dataframe(pd.DataFrame(item_rows), use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 4: ⚖️ HUMAN REVIEW QUEUE
# =============================================================================

def render_review_queue_page(
    db: DatabaseManager,
    storage: FileStorageManager,
    workflow: WorkflowEngine,
    audit: AuditService
):
    st.markdown('<div class="main-title">Human Review Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Dedicated Review Interface • Action Approvals & Rejections • Mandatory Rejection Auditing</div>', unsafe_allow_html=True)

    pending_docs = db.get_documents_by_status(STATUS_NEEDS_REVIEW)

    if not pending_docs:
        st.markdown("""
            <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 2rem; text-align: center; margin: 2rem 0;">
                <h3 style="color: #065F46; margin-top: 0;">🎉 All Caught Up!</h3>
                <p style="color: #047857; margin-bottom: 0;">There are currently 0 documents requiring human review. All ingested documents have completed validation.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"**Found `{len(pending_docs)}` document(s) pending human review.**")

    for doc in pending_docs:
        doc_id = doc["id"]
        with st.container():
            st.markdown(f"""
                <div class="review-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #0F172A;">Document #{doc_id}: {doc['original_filename']}</h4>
                        <div>
                            <span class="badge-type">{doc['document_type']}</span>
                            {render_status_badge(doc['status'])}
                        </div>
                    </div>
                    <p style="color: #64748B; font-size: 0.85rem; margin: 0 0 0.8rem 0;">
                        Uploaded: <code>{doc['upload_date'][:19].replace('T', ' ')}</code> |
                        Digest: <code>{doc['file_hash'][:16]}...</code> |
                        {render_confidence_badge(doc.get('confidence_score'))}
                    </p>
                    <div style="background-color: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.8rem;">
                        <strong style="color: #92400E;">⚠️ Reason Flagged for Review:</strong>
                        <span style="color: #78350F;">{doc.get('status_reason', 'Validation checks flagged for review.')}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            col_fields, col_actions = st.columns([3, 2])

            with col_fields:
                st.markdown("##### 📋 Extracted Entities")
                if doc["document_type"] == TYPE_INVOICE:
                    f_data = [
                        ("Invoice Number", doc.get("invoice_number", "Not Found")),
                        ("Company / Vendor", doc.get("company", "Not Found")),
                        ("Total Amount", doc.get("total_amount", "Not Found")),
                    ]
                elif doc["document_type"] == TYPE_RESUME:
                    f_data = [
                        ("Candidate Name", doc.get("candidate_name", "Not Found")),
                        ("Email Address", doc.get("candidate_email", "Not Found")),
                        ("Phone Number", doc.get("candidate_phone", "Not Found")),
                        ("Skills", doc.get("candidate_skills", "Not Found")),
                    ]
                else:
                    f_data = [("Category Notice", "Document classified as 'Other'.")]

                for k, v in f_data:
                    val_color = "#94A3B8" if v == "Not Found" else "#0F172A"
                    st.markdown(f"**{k}:** <span style='color: {val_color}; font-weight: 600;'>{v}</span>", unsafe_allow_html=True)

                with st.expander("📄 View Extracted Text Preview"):
                    st.text_area("Extracted Text", doc.get("text_preview", ""), height=140, disabled=True, key=f"preview_{doc_id}")

            with col_actions:
                st.markdown("##### ⚖️ Reviewer Actions")
                reviewer_note = st.text_input(
                    "Reviewer Note / Decision Reason:",
                    placeholder="Enter approval note or mandatory rejection reason...",
                    key=f"note_{doc_id}"
                )

                c_app, c_rej = st.columns(2)
                with c_app:
                    if st.button("✓ Approve", key=f"btn_app_{doc_id}", type="primary", use_container_width=True):
                        ok, msg = workflow.approve_document(doc_id, reviewer_note=reviewer_note)
                        if ok:
                            st.success(f"Document #{doc_id} Approved & Completed!")
                            st.rerun()
                        else:
                            st.error(msg)

                with c_rej:
                    if st.button("✕ Reject", key=f"btn_rej_{doc_id}", type="secondary", use_container_width=True):
                        # Strict enforcement: note is required
                        if not reviewer_note.strip():
                            st.error("Rejection note is mandatory! Please explain the reason for rejection.")
                        else:
                            ok, msg = workflow.reject_document(doc_id, reviewer_note=reviewer_note)
                            if ok:
                                st.warning(f"Document #{doc_id} Rejected.")
                                st.rerun()
                            else:
                                st.error(msg)

                if st.button("⟳ Re-process Document", key=f"btn_re_{doc_id}", use_container_width=True):
                    ok, msg = workflow.reprocess_document(doc_id, reviewer_note="Queued for reprocessing by reviewer.")
                    if ok:
                        st.info(f"Document #{doc_id} moved back to Processing.")
                        st.rerun()
                    else:
                        st.error(msg)

            # Audit History for this document
            with st.expander(f"📜 View Lifecycle Audit Trail for Document #{doc_id}"):
                history = audit.get_document_history(doc_id)
                for h in history:
                    note_str = f" | Note: {h['reviewer_note']}" if h.get('reviewer_note') else ""
                    st.markdown(f"""
                        <div class="audit-item">
                            <strong>{h['action']}</strong>: <code>{h['previous_status']}</code> → <code>{h['new_status']}</code>
                            <br><small style="color: #64748B;">{h['timestamp'][:19].replace('T', ' ')} | {h.get('reason') or ''}{note_str}</small>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")


# =============================================================================
# VIEW 5: 🔎 SEARCH & FILTER DOCUMENTS
# =============================================================================

def render_search_page(db: DatabaseManager, storage: FileStorageManager):
    st.markdown('<div class="main-title">Search & Filter Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Fast Parameterized Querying Across Workflow States, Document Types, and Extracted Fields</div>', unsafe_allow_html=True)

    query = st.text_input(
        "🔍 Global Search Query:",
        placeholder="Search by filename, company, invoice number, candidate, or full text...",
        help="Parameterized SQL index lookup."
    )

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    with f1:
        doc_type_filter = st.selectbox("Category:", ["All"] + ALL_DOCUMENT_TYPES, index=0)
    with f2:
        status_filter = st.selectbox("Workflow State:", ["All"] + WORKFLOW_STATUSES, index=0)
    with f3:
        sort_order = st.selectbox("Sort Order:", ["Newest First", "Oldest First"], index=0)
    with f4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset", use_container_width=True):
            st.rerun()

    sql_sort = "DESC" if "Newest" in sort_order else "ASC"
    results = db.search_documents(
        search_query=query,
        doc_type=doc_type_filter,
        status=status_filter,
        sort_order=sql_sort,
        limit=100
    )

    st.markdown(f"**Found `{len(results)}` matching document record(s):**")
    if results:
        table_rows = []
        for r in results:
            table_rows.append({
                "ID": f"#{r['id']}",
                "Filename": r["original_filename"],
                "Category": r["document_type"],
                "Workflow Status": r["status"],
                "Company / Name": r["company"] if r["document_type"] == "Invoice" else r["candidate_name"],
                "Total / Email": r["total_amount"] if r["document_type"] == "Invoice" else r["candidate_email"],
                "Upload Date": r["upload_date"][:10]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 6: 📁 DOCUMENT REPOSITORY & AUDIT VIEWER
# =============================================================================

def render_repository_page(
    db: DatabaseManager,
    storage: FileStorageManager,
    workflow: WorkflowEngine,
    audit: AuditService
):
    st.markdown('<div class="main-title">Document Vault & Repository</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Persistent SQLite Repository with Full Entity Inspector and Visual Audit Timeline</div>', unsafe_allow_html=True)

    all_docs = db.get_all_documents(sort_order="DESC")
    if not all_docs:
        st.info("The repository vault is empty. Ingest documents to begin.")
        return

    doc_options = {f"#{d['id']} - {d['original_filename']} ({d['document_type']} | {d['status']})": d['id'] for d in all_docs}
    selected_label = st.selectbox("Select document to inspect:", list(doc_options.keys()))
    selected_id = doc_options[selected_label]

    doc = db.get_document_by_id(selected_id)
    if doc:
        render_single_document_details(doc, storage, db, workflow=workflow, audit=audit)


# =============================================================================
# VIEW 7: 📜 SYSTEM-WIDE AUDIT TRAIL
# =============================================================================

def render_audit_trail_page(db: DatabaseManager, audit: AuditService):
    st.markdown('<div class="main-title">System Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Immutable, Chronological Log of All Automated Pipeline Stages and Human Reviewer Actions</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        action_filter = st.selectbox("Filter by Action:", ["All Actions"] + [
            ACTION_UPLOADED, ACTION_PROCESSING_STARTED, ACTION_CLASSIFIED, ACTION_EXTRACTED,
            ACTION_VALIDATION_PASSED, ACTION_VALIDATION_FAILED, ACTION_ROUTED_TO_REVIEW,
            ACTION_ROUTED_TO_COMPLETED, ACTION_REVIEWER_APPROVED, ACTION_REVIEWER_REJECTED,
            ACTION_WORKFLOW_COMPLETED, ACTION_WORKFLOW_FAILED, ACTION_RETRIED
        ])
    with c2:
        limit_num = st.selectbox("Records limit:", [25, 50, 100, 200], index=1)

    events = db.get_recent_audit_events(limit=limit_num)
    if action_filter != "All Actions":
        events = [e for e in events if e["action"] == action_filter]

    st.markdown(f"**Displaying `{len(events)}` audit log entry/entries:**")
    if events:
        table_rows = []
        for e in events:
            table_rows.append({
                "Audit ID": f"#{e['audit_id']}",
                "Timestamp": e["timestamp"][:19].replace("T", " "),
                "Document": f"#{e['document_id']} ({e.get('original_filename') or '—'})",
                "Action": e["action"],
                "State Transition": f"{e.get('previous_status') or 'None'} → {e['new_status']}",
                "Reason / Routing": e.get("reason") or "—",
                "Reviewer Note": e.get("reviewer_note") or "—"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 8: 📊 WORKFLOW ANALYTICS & ML BENCHMARK
# =============================================================================

def render_analytics_page(db: DatabaseManager, eval_results: Optional[Dict[str, Any]]):
    st.markdown('<div class="main-title">Analytics & ML Model Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Model Performance Metrics • Accuracy, Precision, Recall, and F1-Score Comparisons</div>', unsafe_allow_html=True)

    if eval_results and "comparison_table" in eval_results:
        st.markdown("##### 🏆 Classifier Comparison Benchmark Table")
        st.dataframe(pd.DataFrame(eval_results["comparison_table"]), use_container_width=True, hide_index=True)
        st.success(f"Recommended Model: **{eval_results.get('best_model_name')}** (Evaluated across Accuracy and F1-Score)")

    stats = db.get_statistics()
    st.markdown("---")
    st.markdown("##### 📈 Workflow State Distribution")
    s_df = pd.DataFrame([
        {"Status": "Completed", "Count": stats["completed"]},
        {"Status": "Needs Review", "Count": stats["needs_review"]},
        {"Status": "Approved", "Count": stats["approved"]},
        {"Status": "Rejected", "Count": stats["rejected"]},
        {"Status": "Failed", "Count": stats["failed"]}
    ])
    st.bar_chart(s_df.set_index("Status"))


# =============================================================================
# VIEW 9: ⚙️ SETTINGS & HEALTH
# =============================================================================

def render_settings_page(db: DatabaseManager, storage: FileStorageManager, ocr_engine: OCRProcessor):
    st.markdown('<div class="main-title">System Settings & Health</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Hardware Engine Health • Physical Storage Status • Database Integrity</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔍 Optical Character Recognition (OCR) Engine")
        if ocr_engine.tesseract_available:
            st.success("✓ Tesseract OCR binary is active and ready.")
        else:
            st.warning("⚠️ Tesseract OCR binary not found on PATH. Synthetic & selectable PDF parser active.")

    with c2:
        st.markdown("##### 💾 Storage & Vault Status")
        st.info(f"Database Path: `{db.db_path}`\n\nStorage Root: `{storage.storage_dir}`")

    st.markdown("---")
    if st.button("🔄 Regenerate All 12 Rich Sample PDF Documents", type="secondary"):
        generate_rich_sample_pdfs(SAMPLES_DIR)
        st.success("Sample PDF documents regenerated successfully in samples/ directory.")


# =============================================================================
# HELPER: SINGLE DOCUMENT DETAILS COMPONENT
# =============================================================================

def render_single_document_details(
    doc: Dict[str, Any],
    storage: FileStorageManager,
    db: DatabaseManager,
    workflow: Optional[WorkflowEngine] = None,
    audit: Optional[AuditService] = None,
    live_analysis: Optional[Dict[str, Any]] = None
):
    metadata = doc.get("metadata", {})
    doc_id = doc["id"]

    col_meta, col_fields = st.columns([1, 1])

    with col_meta:
        st.markdown("##### 📌 Document Record Summary")
        st.markdown(f"**Document ID:** `#{doc_id}`")
        st.markdown(f"**Original Filename:** `{doc['original_filename']}`")
        st.markdown(f"**Category:** <span class='badge-type'>{doc['document_type']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Workflow Status:** {render_status_badge(doc['status'])}", unsafe_allow_html=True)
        st.markdown(f"**Status Reason:** `{doc.get('status_reason', '—')}`")
        st.markdown(render_confidence_badge(doc.get("confidence_score")), unsafe_allow_html=True)
        st.markdown(f"**SHA-256 Digest:** `{doc.get('file_hash', '—')[:24]}...`")

        # Download Stored File
        file_bytes = storage.read_file(doc["file_path"])
        if file_bytes:
            st.download_button(
                label=f"📥 Download Stored File",
                data=file_bytes,
                file_name=doc["original_filename"],
                mime="application/pdf" if doc["original_filename"].lower().endswith(".pdf") else "image/png",
                use_container_width=True
            )

    with col_fields:
        st.markdown("##### 📋 Extracted Entities")
        if doc["document_type"] == TYPE_INVOICE:
            fields_data = [
                ("Invoice Number", doc.get("invoice_number", "Not Found")),
                ("Company Name", doc.get("company", "Not Found")),
                ("Total Amount", doc.get("total_amount", "Not Found")),
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
            val_style = "color: #94A3B8; font-style: italic;" if v == "Not Found" else "color: #0F172A; font-weight: 600;"
            st.markdown(f"**{k}:** <span style='{val_style}'>{v}</span>", unsafe_allow_html=True)

    # Document Text Preview
    with st.expander("📄 Document Text Preview", expanded=False):
        st.text_area("Extracted Clean Text", doc.get("text_preview", ""), height=150, disabled=True)

    # Document Audit Timeline
    audit_svc = audit or get_audit_service(db)
    history = audit_svc.get_document_history(doc_id)
    if history:
        with st.expander(f"📜 View Lifecycle Audit Trail ({len(history)} events recorded)", expanded=False):
            for h in history:
                note_str = f" | Note: {h['reviewer_note']}" if h.get('reviewer_note') else ""
                st.markdown(f"""
                    <div class="audit-item">
                        <strong>{h['action']}</strong>: <code>{h['previous_status']}</code> → <code>{h['new_status']}</code>
                        <br><small style="color: #64748B;">{h['timestamp'][:19].replace('T', ' ')} | {h.get('reason') or ''}{note_str}</small>
                    </div>
                """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTROLLER & APPLICATION ENTRYPOINT
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI Document Intelligence & Workflow Platform",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_custom_css()

    db, storage, clf, ocr_engine, processor, batch_proc, workflow, audit, eval_results = initialize_system(full=True)

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 📄 AI Document Intelligence")
        st.caption("Advanced Document Workflow & Automation Platform • Week 5")
        st.markdown("---")

        nav_options = [
            "🏠 Dashboard",
            "📤 Ingest & Workflow",
            "⚡ Batch Processing",
            "⚖️ Human Review Queue",
            "🔎 Search & Filter",
            "📁 Document Vault",
            "📜 Audit Trail",
            "📊 Analytics & Benchmark",
            "⚙️ Settings & Health"
        ]

        # Read or set session state for nav selection
        default_index = 0
        if "nav_selection" in st.session_state and st.session_state["nav_selection"] in nav_options:
            default_index = nav_options.index(st.session_state["nav_selection"])

        page = st.radio("Navigation:", nav_options, index=default_index)
        st.session_state["nav_selection"] = page

        st.markdown("---")
        stats = db.get_statistics()
        st.markdown(f"**Total Vault:** `{stats['total']}` files")
        st.markdown(f"**Completed:** `{stats['completed']}`")
        st.markdown(f"**Needs Review:** `{stats['needs_review']}`")
        st.markdown(f"**Audit Logs:** `{stats['total_audit_events']}`")
        st.markdown("---")
        st.caption("AI/ML Document Intelligence Platform\nDeveloped by Sakthibalan S")

    # Route Selected Page
    if page == "🏠 Dashboard":
        render_dashboard(db, storage, workflow)
    elif page == "📤 Ingest & Workflow":
        render_upload_page(db, storage, processor, clf)
    elif page == "⚡ Batch Processing":
        render_batch_page(db, storage, batch_proc, clf)
    elif page == "⚖️ Human Review Queue":
        render_review_queue_page(db, storage, workflow, audit)
    elif page == "🔎 Search & Filter":
        render_search_page(db, storage)
    elif page == "📁 Document Vault":
        render_repository_page(db, storage, workflow, audit)
    elif page == "📜 Audit Trail":
        render_audit_trail_page(db, audit)
    elif page == "📊 Analytics & Benchmark":
        render_analytics_page(db, eval_results)
    elif page == "⚙️ Settings & Health":
        render_settings_page(db, storage, ocr_engine)


if __name__ == "__main__":
    main()
