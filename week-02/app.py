"""
Zyroo AI Document Intelligence Platform — Main Application
AI/ML Internship Task 2 • Masterpiece Build

Features:
- Document Upload (.pdf, .png, .jpg, .jpeg) with full validation
- High-Speed Text Extraction via PyMuPDF + OCR fallback
- Dual-Engine Classification (Rule-Based + Scikit-Learn TF-IDF Logistic Regression)
- Structured Entity Extraction for Invoices (Number, Date, Company, Total) and Resumes (Name, Email, Phone, Skills)
- Full Explainability & Feature Attribution
- Interactive 1-Click Sample Gallery for rapid evaluation
- JSON and CSV Export
"""

import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from src.config import (
    THEME,
    SAMPLES_DIR,
    DOC_TYPE_INVOICE,
    DOC_TYPE_RESUME,
    DOC_TYPE_OTHER
)
from src.validator import validate_document
from src.extractor import DocumentExtractor
from src.classifier import HybridClassifier
from src.parser import DocumentParser
from src.sample_generator import generate_samples

# Configure Streamlit Page
st.set_page_config(
    page_title="Zyroo AI Document Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching Zyroo Masterpiece Theme Palette
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    /* Top Hero Header */
    .hero-container {{
        background: linear-gradient(135deg, {THEME['primary']} 0%, {THEME['secondary']} 100%);
        border-radius: 16px;
        padding: 32px 36px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(11, 16, 32, 0.15);
        border: 1px solid rgba(0, 217, 255, 0.2);
    }}
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #FFFFFF 0%, {THEME['accent_cyan']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 16px;
    }}
    .badge-pill {{
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(0, 217, 255, 0.12);
        color: {THEME['accent_cyan']};
        border: 1px solid rgba(0, 217, 255, 0.3);
        margin-right: 8px;
    }}
    .badge-purple {{
        background: rgba(124, 58, 237, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }}

    /* Metric KPI Cards */
    .metric-card {{
        background: white;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }}
    .metric-label {{
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {THEME['primary']};
    }}
    .metric-sub {{
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 4px;
    }}

    /* Extracted Entity Cards */
    .entity-card {{
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        border-left: 4px solid {THEME['accent_cyan']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 12px;
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
    }}
    .entity-label {{
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .entity-val {{
        font-size: 1.15rem;
        font-weight: 600;
        color: {THEME['text_dark']};
        margin-top: 4px;
    }}
    .entity-meta {{
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 4px;
    }}

    /* Skill Badge */
    .skill-pill {{
        display: inline-block;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 4px;
        background: #F1F5F9;
        color: {THEME['text_dark']};
        border: 1px solid #E2E8F0;
    }}
    .skill-pill-ai {{
        background: #EDE9FE;
        color: #6D28D9;
        border-color: #DDD6FE;
    }}
    .skill-pill-code {{
        background: #E0F2FE;
        color: #0369A1;
        border-color: #BAE6FD;
    }}
    .skill-pill-cloud {{
        background: #ECFDF5;
        color: #047857;
        border-color: #A7F3D0;
    }}

    /* Pipeline Step Tracker */
    .pipeline-step {{
        display: inline-flex;
        align-items: center;
        margin-right: 18px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #10B981;
    }}
    .pipeline-step span {{
        margin-right: 6px;
    }}

    /* Warning callout */
    .warning-box {{
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 14px 18px;
        color: #92400E;
        font-size: 0.9rem;
        margin: 12px 0;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State & Singletons
@st.cache_resource
def get_pipeline():
    """Initializes and caches the processing pipeline components."""
    generate_samples()
    extractor = DocumentExtractor()
    classifier = HybridClassifier()
    parser = DocumentParser()
    return extractor, classifier, parser


extractor, classifier, parser = get_pipeline()

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "uploaded_doc_bytes" not in st.session_state:
    st.session_state.uploaded_doc_bytes = None
if "uploaded_doc_name" not in st.session_state:
    st.session_state.uploaded_doc_name = None


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 10px 0;">
        <h2 style="color: {THEME['primary']}; margin: 0; font-size: 1.4rem;">Zyroo Platform</h2>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 4px;">AI/ML Internship • Task 2</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ Engine Configuration")

    classification_mode = st.radio(
        "Classification Engine:",
        options=["Hybrid (Rules + ML)", "Rule-Based Only", "ML (TF-IDF + Logistic Reg)"],
        index=0,
        help="Select the classification strategy: Hybrid combines deterministic keyword scoring with a trained machine learning model."
    )

    # Engine internal mapping
    mode_map = {
        "Hybrid (Rules + ML)": "Hybrid",
        "Rule-Based Only": "Rule-Based",
        "ML (TF-IDF + Logistic Reg)": "ML"
    }
    selected_mode = mode_map[classification_mode]

    st.markdown("---")
    st.subheader("🔍 OCR Status")
    if extractor.has_pytesseract:
        st.success(f"Tesseract OCR: Active\n`{extractor.tesseract_cmd}`")
    else:
        st.info("PyMuPDF Direct Extraction: Active\n(OCR engine optional on host)")

    st.markdown("---")
    st.markdown("### 📋 Official Task Criteria")
    st.markdown("""
    - [x] **File Upload**: PDF, JPG, PNG
    - [x] **Text Extraction**: PyMuPDF & OCR
    - [x] **Doc Classification**: Invoice, Resume, Other
    - [x] **Information Extraction**:
      - Invoice: Number, Date, Company, Total
      - Resume: Name, Email, Phone, Skills
    - [x] **Explainable Output & Verification**
    """)

    st.markdown("---")
    st.markdown("### 🌐 Official Zyroo Links")
    st.markdown("""
    - [Zyroo Website](https://zyroo.org)
    - [Internship Programs](https://zyroo.org/internships)
    - [Zyroo LinkedIn](https://www.linkedin.com/company/zyr0-co/)
    - [Zyroo Community](https://chat.whatsapp.com/EfivEcFI4cJ8pWnbW9OmWh)
    """)


# =============================================================================
# HERO SECTION
# =============================================================================
st.markdown(f"""
<div class="hero-container">
    <div class="badge-pill">ZYROO AI/ML TASK 2</div>
    <div class="badge-pill badge-purple">PRODUCTION MVP</div>
    <h1 class="hero-title">AI Document Intelligence & Workflow Platform</h1>
    <p class="hero-subtitle">
        Automated ingestion, text extraction, document categorization (Invoice / Resume / Other), 
        and high-accuracy entity extraction with full feature explainability.
    </p>
    <div style="margin-top: 10px;">
        <span class="badge-pill" style="background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2);">
            ⚡ PyMuPDF Native
        </span>
        <span class="badge-pill" style="background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2);">
            🤖 TF-IDF + Logistic Regression
        </span>
        <span class="badge-pill" style="background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2);">
            💼 Multi-Currency Parsing
        </span>
        <span class="badge-pill" style="background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2);">
            🎯 Explainable AI
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# INPUT SELECTOR: UPLOAD vs 1-CLICK SAMPLE GALLERY
# =============================================================================
input_tabs = st.tabs(["🚀 1-Click Sample Showcase", "📁 Upload Custom Document"])

doc_bytes = None
doc_name = None

# Tab 1: 1-Click Sample Showcase
with input_tabs[0]:
    st.markdown("#### Test instantly with official Task 2 sample documents:")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-top: 4px solid {THEME['accent_cyan']};">
            <div class="metric-label">Official Spec Sample</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #0F172A; margin: 6px 0;">Invoice #1</div>
            <div class="metric-sub">ABC Technologies<br>PKR 125,000 • INV-1024</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Invoice #1", use_container_width=True, key="btn_inv1"):
            st.session_state.selected_sample = "invoice_001.pdf"
            st.session_state.uploaded_doc_bytes = None

    with s_col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-top: 4px solid {THEME['accent_purple']};">
            <div class="metric-label">Commercial Cloud</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #0F172A; margin: 6px 0;">Invoice #2</div>
            <div class="metric-sub">CloudScale Systems<br>$4,850.00 • INV-2026-889</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Invoice #2", use_container_width=True, key="btn_inv2"):
            st.session_state.selected_sample = "invoice_002.pdf"
            st.session_state.uploaded_doc_bytes = None

    with s_col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-top: 4px solid #10B981;">
            <div class="metric-label">AI/ML Candidate</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #0F172A; margin: 6px 0;">Resume #1</div>
            <div class="metric-sub">Alex Morgan<br>Senior AI/ML Engineer</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Resume #1", use_container_width=True, key="btn_res1"):
            st.session_state.selected_sample = "resume_001.pdf"
            st.session_state.uploaded_doc_bytes = None

    with s_col4:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-top: 4px solid #F59E0B;">
            <div class="metric-label">Edge Case / Other</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #0F172A; margin: 6px 0;">Memo / Notes</div>
            <div class="metric-sub">Architecture Memo<br>Unstructured Document</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Other Doc", use_container_width=True, key="btn_other"):
            st.session_state.selected_sample = "sample_other.pdf"
            st.session_state.uploaded_doc_bytes = None

# Tab 2: Live File Uploader
with input_tabs[1]:
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, PNG, JPG, JPEG):",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload up to 10 MB document. Both native text PDFs and scanned documents/images are accepted."
    )
    if uploaded_file is not None:
        st.session_state.uploaded_doc_bytes = uploaded_file.read()
        st.session_state.uploaded_doc_name = uploaded_file.name
        st.session_state.selected_sample = None

# Determine active file
if st.session_state.uploaded_doc_bytes is not None:
    doc_bytes = st.session_state.uploaded_doc_bytes
    doc_name = st.session_state.uploaded_doc_name
elif st.session_state.selected_sample is not None:
    sample_file_path = SAMPLES_DIR / st.session_state.selected_sample
    if sample_file_path.exists():
        doc_bytes = sample_file_path.read_bytes()
        doc_name = sample_file_path.name
else:
    # Default initial document: sample_invoice_01.pdf
    default_path = SAMPLES_DIR / "invoice_001.pdf"
    if default_path.exists():
        doc_bytes = default_path.read_bytes()
        doc_name = default_path.name

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# PIPELINE EXECUTION & PROCESSING
# =============================================================================
if doc_bytes is not None and doc_name is not None:
    start_time = time.perf_counter()

    # Step 1: Validation
    val_result = validate_document(doc_bytes, doc_name)
    if not val_result.is_valid:
        st.error(f"⚠️ Validation Failed: {val_result.error_message}")
        st.stop()

    # Step 2: Extraction
    extraction_result = extractor.extract_from_bytes(doc_bytes, doc_name)

    # Step 3: Classification
    classification_result = classifier.classify(extraction_result.text, mode=selected_mode)

    # Step 4: Parsing
    parse_result = parser.parse(extraction_result.text, classification_result.document_type)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    # Pipeline Breadcrumb Tracker
    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;">
        <span class="pipeline-step"><span>✓</span> 1. Validated ({val_result.file_type}, {val_result.file_size_kb} KB)</span>
        <span class="pipeline-step"><span>✓</span> 2. Extracted ({extraction_result.page_count} pg, {extraction_result.char_count} chars)</span>
        <span class="pipeline-step"><span>✓</span> 3. Classified ({classification_result.document_type} - {classification_result.confidence*100:.0f}%)</span>
        <span class="pipeline-step"><span>✓</span> 4. Extracted Entities ({parse_result.fields_found_count}/{parse_result.total_expected_fields})</span>
        <span class="pipeline-step" style="color: {THEME['accent_purple']};"><span>✓</span> Pipeline Complete ({elapsed_ms} ms)</span>
    </div>
    """, unsafe_allow_html=True)

    # Warnings from extraction
    if extraction_result.warnings:
        for w in extraction_result.warnings:
            st.markdown(f"<div class='warning-box'>⚠️ <strong>Notice:</strong> {w}</div>", unsafe_allow_html=True)

    # =========================================================================
    # KPI METRIC CARDS
    # =========================================================================
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    # Color code doc type
    doc_type_color = {
        DOC_TYPE_INVOICE: "#0284C7",
        DOC_TYPE_RESUME: "#7C3AED",
        DOC_TYPE_OTHER: "#D97706"
    }.get(classification_result.document_type, "#64748B")

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Identified Document Type</div>
            <div class="metric-value" style="color: {doc_type_color};">{classification_result.document_type}</div>
            <div class="metric-sub">{doc_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        conf_pct = round(classification_result.confidence * 100, 1)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AI Confidence Score</div>
            <div class="metric-value">{conf_pct}%</div>
            <div class="metric-sub">{classification_result.confidence_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Field Completeness</div>
            <div class="metric-value">{parse_result.completeness_score}%</div>
            <div class="metric-sub">{parse_result.fields_found_count} of {parse_result.total_expected_fields} entities matched</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Extraction Latency</div>
            <div class="metric-value">{elapsed_ms} ms</div>
            <div class="metric-sub">{extraction_result.extraction_status}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # TABBED RESULT PRESENTATION
    # =========================================================================
    tab_overview, tab_explain, tab_text, tab_export = st.tabs([
        "📊 Structured Entities",
        "🧠 AI Explainability",
        "📄 Raw Document Text",
        "💾 Export & Payload"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: STRUCTURED ENTITIES
    # -------------------------------------------------------------------------
    with tab_overview:
        st.markdown("### Extracted Information")

        if classification_result.document_type == DOC_TYPE_INVOICE:
            # Invoice Grid
            inv_col1, inv_col2 = st.columns(2)
            with inv_col1:
                f_no = parse_result.fields.get("Invoice Number")
                st.markdown(f"""
                <div class="entity-card">
                    <div class="entity-label">Invoice Number</div>
                    <div class="entity-val">{f_no.value if f_no else 'Not found'}</div>
                    <div class="entity-meta">Confidence: {f_no.confidence*100:.0f}% • Status: {f_no.validation_status}</div>
                </div>
                """, unsafe_allow_html=True)

                f_date = parse_result.fields.get("Date")
                st.markdown(f"""
                <div class="entity-card">
                    <div class="entity-label">Invoice Date</div>
                    <div class="entity-val">{f_date.value if f_date else 'Not found'}</div>
                    <div class="entity-meta">Confidence: {f_date.confidence*100:.0f}% • Status: {f_date.validation_status}</div>
                </div>
                """, unsafe_allow_html=True)

            with inv_col2:
                f_comp = parse_result.fields.get("Company Name")
                st.markdown(f"""
                <div class="entity-card">
                    <div class="entity-label">Company / Client Name</div>
                    <div class="entity-val">{f_comp.value if f_comp else 'Not found'}</div>
                    <div class="entity-meta">Confidence: {f_comp.confidence*100:.0f}% • Status: {f_comp.validation_status}</div>
                </div>
                """, unsafe_allow_html=True)

                f_tot = parse_result.fields.get("Total Amount")
                st.markdown(f"""
                <div class="entity-card" style="border-left-color: #10B981;">
                    <div class="entity-label">Total Amount Due</div>
                    <div class="entity-val" style="color: #047857; font-size: 1.35rem;">{f_tot.value if f_tot else 'Not found'}</div>
                    <div class="entity-meta">Confidence: {f_tot.confidence*100:.0f}% • Status: {f_tot.validation_status}</div>
                </div>
                """, unsafe_allow_html=True)

        elif classification_result.document_type == DOC_TYPE_RESUME:
            # Resume Grid
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                f_name = parse_result.fields.get("Name")
                st.markdown(f"""
                <div class="entity-card">
                    <div class="entity-label">Candidate Name</div>
                    <div class="entity-val">{f_name.value if f_name else 'Not found'}</div>
                    <div class="entity-meta">Confidence: {f_name.confidence*100:.0f}% • Status: {f_name.validation_status}</div>
                </div>
                """, unsafe_allow_html=True)

                f_email = parse_result.fields.get("Email")
                st.markdown(f"""
                <div class="entity-card">
                    <div class="entity-label">Email Address</div>
                    <div class="entity-val">{f_email.value if f_email else 'Not found'}</div>
                    <div class="entity-meta">Confidence: {f_email.confidence*100:.0f}% • Status: {f_email.validation_status}</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                f_phone = parse_result.fields.get("Phone")
                st.markdown(f"""
                <div class="entity-card">
                    <div class="entity-label">Phone Number</div>
                    <div class="entity-val">{f_phone.value if f_phone else 'Not found'}</div>
                    <div class="entity-meta">Confidence: {f_phone.confidence*100:.0f}% • Status: {f_phone.validation_status}</div>
                </div>
                """, unsafe_allow_html=True)

                f_skills = parse_result.fields.get("Skills")
                st.markdown(f"""
                <div class="entity-card" style="border-left-color: {THEME['accent_purple']};">
                    <div class="entity-label">Identified Skills Count</div>
                    <div class="entity-val" style="color: #6D28D9;">{len(parse_result.all_skills_flat)} Technologies Detected</div>
                    <div class="entity-meta">{f_skills.context_snippet if f_skills else ''}</div>
                </div>
                """, unsafe_allow_html=True)

            # Categorized Skills Display
            if parse_result.skills_by_category:
                st.markdown("#### 🛠️ Categorized Skills Matrix")
                for category, skills in parse_result.skills_by_category.items():
                    pill_class = "skill-pill-ai" if "AI" in category else ("skill-pill-code" if "Programming" in category else "skill-pill-cloud")
                    pills_html = "".join([f'<span class="skill-pill {pill_class}">{s}</span>' for s in skills])
                    st.markdown(f"**{category}:**<br>{pills_html}", unsafe_allow_html=True)
                    st.write("")

        else:
            # Other Document
            st.info("ℹ️ Document classified as 'Other' (general document). No structured invoice or resume entities apply.")
            st.markdown(f"**Explanation:** {classification_result.explanation}")

    # -------------------------------------------------------------------------
    # TAB 2: AI EXPLAINABILITY & FEATURE ATTRIBUTION
    # -------------------------------------------------------------------------
    with tab_explain:
        st.markdown("### Why did the system reach this decision?")
        st.markdown(f"""
        **Decision Summary:** `{classification_result.explanation}`  
        **Classification Strategy:** `{classification_result.method}`
        """)

        ex_col1, ex_col2 = st.columns(2)

        with ex_col1:
            st.markdown("#### Keyword Signals (Rule-Based)")
            if classification_result.top_keywords:
                kw_data = [
                    {"Keyword": k.keyword.title(), "Impact Score": k.total_contribution, "Matches": k.count}
                    for k in classification_result.top_keywords
                ]
                df_kw = pd.DataFrame(kw_data)
                fig_kw = px.bar(
                    df_kw,
                    x="Impact Score",
                    y="Keyword",
                    orientation="h",
                    color="Impact Score",
                    color_continuous_scale=["#00D9FF", "#0B1020"],
                    title="Top Weighted Keyword Matches"
                )
                fig_kw.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.write("No strong domain keywords detected (expected for Other documents).")

        with ex_col2:
            st.markdown("#### ML Probability Distribution")
            if classification_result.probabilities:
                prob_data = [
                    {"Class": k, "Probability (%)": round(v * 100, 2)}
                    for k, v in classification_result.probabilities.items()
                ]
                df_prob = pd.DataFrame(prob_data)
                fig_prob = px.pie(
                    df_prob,
                    names="Class",
                    values="Probability (%)",
                    color="Class",
                    color_discrete_map={
                        DOC_TYPE_INVOICE: "#0284C7",
                        DOC_TYPE_RESUME: "#7C3AED",
                        DOC_TYPE_OTHER: "#F59E0B"
                    },
                    hole=0.4,
                    title="Class Probability Distribution"
                )
                fig_prob.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.write("Probability distribution unavailable for pure rule mode.")

        # ML Feature Attribution Table
        if classification_result.top_ml_features:
            st.markdown("#### Machine Learning Feature Importance (TF-IDF Attribution)")
            feat_df = pd.DataFrame(classification_result.top_ml_features, columns=["N-Gram Token", "Feature Weight (TF-IDF × Coef)"])
            st.dataframe(feat_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: RAW DOCUMENT TEXT & DIAGNOSTICS
    # -------------------------------------------------------------------------
    with tab_text:
        st.markdown("### Raw Extracted Text")
        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("Page Count", extraction_result.page_count)
        t_col2.metric("Character Count", extraction_result.char_count)
        t_col3.metric("Extraction Method", "PyMuPDF Direct" if not extraction_result.used_ocr else "OCR Engine")

        search_query = st.text_input("Filter / Search in extracted text:", placeholder="Type to search text...")
        display_text = extraction_result.text
        if search_query:
            matched_lines = [l for l in display_text.split("\n") if search_query.lower() in l.lower()]
            display_text = "\n".join(matched_lines) if matched_lines else f"No lines matched '{search_query}'."

        st.text_area("Document Content:", value=display_text, height=350)

    # -------------------------------------------------------------------------
    # TAB 4: EXPORT DATA
    # -------------------------------------------------------------------------
    with tab_export:
        st.markdown("### Export & Downstream Integration")

        export_payload = {
            "document_metadata": {
                "filename": doc_name,
                "file_type": val_result.file_type,
                "file_size_kb": val_result.file_size_kb,
                "extraction_latency_ms": elapsed_ms,
                "extraction_status": extraction_result.extraction_status,
                "used_ocr": extraction_result.used_ocr
            },
            "classification": {
                "document_type": classification_result.document_type,
                "confidence": classification_result.confidence,
                "confidence_label": classification_result.confidence_label,
                "method": classification_result.method,
                "explanation": classification_result.explanation
            },
            "extraction_results": parse_result.to_dict()
        }

        json_str = json.dumps(export_payload, indent=2)

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                label="📥 Download JSON Payload",
                data=json_str,
                file_name=f"{Path(doc_name).stem}_extracted.json",
                mime="application/json",
                use_container_width=True
            )

        with exp_col2:
            # Flatten fields for CSV
            csv_rows = []
            for k, v in parse_result.fields.items():
                csv_rows.append({
                    "Document": doc_name,
                    "Type": classification_result.document_type,
                    "Field": k,
                    "Value": str(v.value),
                    "Found": v.found,
                    "Confidence": v.confidence,
                    "Status": v.validation_status
                })
            df_csv = pd.DataFrame(csv_rows)
            st.download_button(
                label="📥 Download CSV Summary",
                data=df_csv.to_csv(index=False),
                file_name=f"{Path(doc_name).stem}_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("#### JSON Preview")
        st.json(export_payload)

else:
    st.info("Please select a sample document above or upload a PDF/image to begin analysis.")
