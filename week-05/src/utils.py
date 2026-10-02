"""
Utility Module
Provides dataset loading, export formatters (JSON & CSV),
and sample document generation utilities.
"""

import os
import json
import csv
import io
from typing import Dict, Any, List, Tuple
import pymupdf

from .classifier import CLASS_INVOICE, CLASS_RESUME, CLASS_OTHER


def load_dataset_from_dir(directory: str) -> Tuple[List[str], List[str]]:
    """
    Loads text files and their corresponding ground truth labels
    from a partitioned directory structure (invoices/, resumes/, other/).
    """
    texts = []
    labels = []

    class_mapping = [
        ("invoices", CLASS_INVOICE),
        ("resumes", CLASS_RESUME),
        ("other", CLASS_OTHER)
    ]

    for sub_dir, label in class_mapping:
        folder = os.path.join(directory, sub_dir)
        if not os.path.isdir(folder):
            continue

        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            texts.append(content)
                            labels.append(label)
                except Exception:
                    continue

    return texts, labels


def export_result_to_json(result: Dict[str, Any]) -> str:
    """Formats the normalized analysis result into indented JSON."""
    return json.dumps(result, indent=2, ensure_ascii=False)


def export_result_to_csv(result: Dict[str, Any]) -> str:
    """Exports extracted fields and metadata into CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Attribute", "Value"])
    writer.writerow(["Filename", result.get("filename", "")])
    writer.writerow(["File Type", result.get("file_type", "")])
    writer.writerow(["Document Type", result.get("document_type", "")])
    writer.writerow(["Confidence", result.get("confidence", "")])
    writer.writerow(["Classification Model", result.get("classification_model", "")])
    writer.writerow(["OCR Used", result.get("ocr_used", False)])
    writer.writerow(["Word Count", result.get("word_count", 0)])

    fields = result.get("fields", {})
    if isinstance(fields, dict):
        for k, v in fields.items():
            writer.writerow([f"Field: {k}", v])

    return output.getvalue()


def generate_rich_sample_pdfs(samples_dir: str) -> None:
    """Generates authentic, diverse PDF test files with varying layouts and edge cases."""
    os.makedirs(samples_dir, exist_ok=True)

    # 1. invoice_1.pdf (Standard Consulting Invoice)
    doc1 = pymupdf.open()
    page1 = doc1.new_page(width=595, height=842)
    page1.insert_text((50, 60), "TechCorp Solutions Inc.", fontsize=18, color=(0.1, 0.2, 0.5))
    page1.insert_text((50, 80), "100 Innovation Boulevard, Suite 400 | contact@techcorpsolutions.com", fontsize=9, color=(0.4, 0.4, 0.4))
    page1.insert_text((380, 60), "INVOICE", fontsize=22, color=(0.1, 0.1, 0.1))
    page1.insert_text((380, 85), "Invoice Number: INV-2026-001", fontsize=10)
    page1.insert_text((380, 100), "Date: 15/03/2026", fontsize=10)
    page1.insert_text((380, 115), "Due Date: 30/03/2026", fontsize=10)
    page1.draw_line((50, 135), (545, 135), color=(0.8, 0.8, 0.8), width=1)
    page1.insert_text((50, 160), "Bill To: Global Enterprises Ltd.", fontsize=11)
    page1.insert_text((50, 180), "Cloud Architecture Consulting: $ 1,200.00", fontsize=10)
    page1.insert_text((50, 200), "API Integration Services: $ 250.00", fontsize=10)
    page1.insert_text((350, 240), "Total Amount: $ 1,450.00", fontsize=12, color=(0.1, 0.2, 0.5))
    doc1.save(os.path.join(samples_dir, "invoice_1.pdf"))
    doc1.close()

    # 2. invoice_2.pdf (Retail Tax Invoice with INR)
    doc2 = pymupdf.open()
    page2 = doc2.new_page(width=595, height=842)
    page2.insert_text((50, 60), "Apex Retailers Pvt Ltd", fontsize=18, color=(0.6, 0.1, 0.1))
    page2.insert_text((50, 80), "Plot 42, Industrial Area, Phase II | GSTIN: 29AAAAA0000A1Z5", fontsize=9, color=(0.4, 0.4, 0.4))
    page2.insert_text((380, 60), "TAX INVOICE", fontsize=20, color=(0.1, 0.1, 0.1))
    page2.insert_text((380, 85), "Invoice No: INV-8832", fontsize=10)
    page2.insert_text((380, 100), "Date: 2026-04-10", fontsize=10)
    page2.draw_line((50, 135), (545, 135), color=(0.8, 0.8, 0.8), width=1)
    page2.insert_text((50, 160), "Customer Details: Sharma Electronics", fontsize=11)
    page2.insert_text((50, 185), "Enterprise Server Rack Unit: 65,000", fontsize=10)
    page2.insert_text((50, 205), "Managed Gigabit Network Switch: 13,500", fontsize=10)
    page2.insert_text((340, 245), "Total Amount: Rs. 78,500", fontsize=12, color=(0.6, 0.1, 0.1))
    doc2.save(os.path.join(samples_dir, "invoice_2.pdf"))
    doc2.close()

    # 3. invoice_missing_total.pdf (Missing Field Test Document)
    doc_inv_missing = pymupdf.open()
    p_inv_m = doc_inv_missing.new_page(width=595, height=842)
    p_inv_m.insert_text((50, 60), "Beacon Creative Agency", fontsize=18, color=(0.2, 0.2, 0.2))
    p_inv_m.insert_text((380, 60), "PROVISIONAL INVOICE", fontsize=16)
    p_inv_m.insert_text((380, 85), "Invoice #: BCA-2026-88", fontsize=10)
    p_inv_m.insert_text((380, 100), "Date: 14-08-2026", fontsize=10)
    p_inv_m.insert_text((50, 140), "Billed To: Solstice Brands", fontsize=11)
    p_inv_m.insert_text((50, 170), "Services Rendered: Brand Identity Guidelines, Logo Vector Design", fontsize=10)
    p_inv_m.insert_text((50, 195), "Billing Status: Hours pending final client audit and reconciliation.", fontsize=10)
    p_inv_m.insert_text((50, 215), "Payment Terms: Net 30 days after finalized billing notice.", fontsize=10)
    doc_inv_missing.save(os.path.join(samples_dir, "invoice_missing_total.pdf"))
    doc_inv_missing.close()

    # 4. resume_1.pdf (Standard Software/ML Engineer Resume)
    doc3 = pymupdf.open()
    page3 = doc3.new_page(width=595, height=842)
    page3.insert_text((50, 60), "Alex Smith", fontsize=22, color=(0.1, 0.2, 0.4))
    page3.insert_text((50, 80), "Email: alex.smith@email.com | Phone: +91 98765 43210 | Bengaluru, India", fontsize=10, color=(0.3, 0.3, 0.3))
    page3.draw_line((50, 95), (545, 95), color=(0.2, 0.4, 0.7), width=1.5)
    page3.insert_text((50, 120), "PROFESSIONAL SUMMARY", fontsize=12, color=(0.1, 0.2, 0.4))
    page3.insert_text((50, 140), "Results-driven AI/ML Engineer with 3+ years designing document intelligence pipelines.", fontsize=10)
    page3.insert_text((50, 170), "SKILLS & TECHNICAL EXPERTISE", fontsize=12, color=(0.1, 0.2, 0.4))
    page3.insert_text((50, 190), "Languages: Python, SQL, JavaScript, C++", fontsize=10)
    page3.insert_text((50, 205), "Frameworks & Tools: Streamlit, PyMuPDF, Scikit-learn, PyTorch, Docker, FastAPI, Git", fontsize=10)
    page3.insert_text((50, 235), "WORK EXPERIENCE", fontsize=12, color=(0.1, 0.2, 0.4))
    page3.insert_text((50, 255), "AI Engineer — DataVibe Systems (2023 - Present)", fontsize=11)
    page3.insert_text((50, 290), "EDUCATION", fontsize=12, color=(0.1, 0.2, 0.4))
    page3.insert_text((50, 310), "Bachelor of Engineering in Computer Science, VTU (2017 - 2021)", fontsize=10)
    doc3.save(os.path.join(samples_dir, "resume_1.pdf"))
    doc3.close()

    # 5. resume_missing_phone.pdf (Resume with Missing Phone Field)
    doc_res_missing = pymupdf.open()
    p_res_m = doc_res_missing.new_page(width=595, height=842)
    p_res_m.insert_text((50, 60), "Rachel Adams", fontsize=22, color=(0.2, 0.2, 0.3))
    p_res_m.insert_text((50, 80), "Email: rachel.adams@uxdesign.net | San Francisco, CA", fontsize=10, color=(0.3, 0.3, 0.3))
    p_res_m.draw_line((50, 95), (545, 95), color=(0.5, 0.5, 0.5), width=1)
    p_res_m.insert_text((50, 120), "SUMMARY", fontsize=12)
    p_res_m.insert_text((50, 140), "Senior UX Designer & Frontend Developer passionate about accessible design systems.", fontsize=10)
    p_res_m.insert_text((50, 170), "SKILLS & TOOLS", fontsize=12)
    p_res_m.insert_text((50, 190), "Figma, JavaScript, TypeScript, React, CSS3, HTML5, Storybook, Jest, Git", fontsize=10)
    p_res_m.insert_text((50, 220), "EXPERIENCE", fontsize=12)
    p_res_m.insert_text((50, 240), "Lead Product Designer — CreativeCanvas (2021 - Present)", fontsize=11)
    doc_res_missing.save(os.path.join(samples_dir, "resume_missing_phone.pdf"))
    doc_res_missing.close()

    # 6. meeting_minutes.pdf ('Other' category document)
    doc_other = pymupdf.open()
    p_oth = doc_other.new_page(width=595, height=842)
    p_oth.insert_text((50, 60), "Quarterly Product Strategy Meeting Minutes", fontsize=18, color=(0.1, 0.1, 0.1))
    p_oth.insert_text((50, 85), "Date: September 10, 2026 | Attendees: Engineering Leads, Product Managers", fontsize=10, color=(0.4, 0.4, 0.4))
    p_oth.draw_line((50, 105), (545, 105), color=(0.8, 0.8, 0.8), width=1)
    p_oth.insert_text((50, 130), "Agenda & Objectives:", fontsize=12)
    p_oth.insert_text((50, 155), "1. Review of Q3 engineering roadmap milestones and deliverable timelines.", fontsize=10)
    p_oth.insert_text((50, 175), "2. Customer feedback analysis on automated document classification release.", fontsize=10)
    p_oth.insert_text((50, 195), "3. Architecture spike on database scalability and microservice partitioning.", fontsize=10)
    p_oth.insert_text((50, 225), "Resolutions & Action Items:", fontsize=12)
    p_oth.insert_text((50, 250), "- Engineering leads approved deployment of the new TF-IDF classifier.", fontsize=10)
    p_oth.insert_text((50, 270), "- Follow-up sprint planning scheduled for next Monday at 10:00 AM.", fontsize=10)
    doc_other.save(os.path.join(samples_dir, "meeting_minutes.pdf"))
    doc_other.close()

    # 7. invoice_missing_number.pdf (Missing Invoice Number -> Needs Review)
    doc7 = pymupdf.open()
    p7 = doc7.new_page(width=595, height=842)
    p7.insert_text((50, 60), "CloudWorks Infrastructure LLC", fontsize=18, color=(0.1, 0.3, 0.6))
    p7.insert_text((50, 80), "450 Silicon Avenue, San Jose, CA", fontsize=9, color=(0.4, 0.4, 0.4))
    p7.insert_text((380, 60), "INVOICE STATEMENT", fontsize=18)
    p7.insert_text((380, 85), "Date: 2026-05-18", fontsize=10)
    p7.draw_line((50, 110), (545, 110), color=(0.8, 0.8, 0.8), width=1)
    p7.insert_text((50, 140), "Client: Velocity Networks Inc.", fontsize=11)
    p7.insert_text((50, 170), "Kubernetes Cluster Management (Q2): $ 2,800.00", fontsize=10)
    p7.insert_text((50, 195), "Terraform Infrastructure Provisioning: $ 400.00", fontsize=10)
    p7.insert_text((350, 240), "Total Amount: $ 3,200.00", fontsize=12, color=(0.1, 0.3, 0.6))
    doc7.save(os.path.join(samples_dir, "invoice_missing_number.pdf"))
    doc7.close()

    # 8. invoice_freelance.pdf (Clean Freelance Design Invoice -> Completed)
    doc8 = pymupdf.open()
    p8 = doc8.new_page(width=595, height=842)
    p8.insert_text((50, 60), "Nexus Digital Studio", fontsize=18, color=(0.5, 0.1, 0.3))
    p8.insert_text((50, 80), "hello@nexusdigital.design | Portfolio: nexusdigital.design", fontsize=9, color=(0.4, 0.4, 0.4))
    p8.insert_text((380, 60), "INVOICE", fontsize=22)
    p8.insert_text((380, 85), "Invoice Number: INV-7712", fontsize=10)
    p8.insert_text((380, 100), "Date: 01/06/2026", fontsize=10)
    p8.draw_line((50, 120), (545, 120), color=(0.8, 0.8, 0.8), width=1)
    p8.insert_text((50, 150), "Billed To: Apex FinTech Corp", fontsize=11)
    p8.insert_text((50, 175), "Design System Architecture & Tokens: $ 2,000.00", fontsize=10)
    p8.insert_text((50, 195), "Mobile App High-Fidelity Prototypes: $ 1,500.00", fontsize=10)
    p8.insert_text((350, 235), "Total Amount: $ 3,500.00", fontsize=12, color=(0.5, 0.1, 0.3))
    doc8.save(os.path.join(samples_dir, "invoice_freelance.pdf"))
    doc8.close()

    # 9. resume_invalid_email.pdf (Malformed Email Format -> Needs Review)
    doc9 = pymupdf.open()
    p9 = doc9.new_page(width=595, height=842)
    p9.insert_text((50, 60), "David Miller", fontsize=22, color=(0.1, 0.2, 0.4))
    p9.insert_text((50, 80), "Email: david.miller.missing.at.domain | Phone: +1 555 123 4567", fontsize=10, color=(0.3, 0.3, 0.3))
    p9.draw_line((50, 95), (545, 95), color=(0.2, 0.4, 0.7), width=1.5)
    p9.insert_text((50, 120), "PROFESSIONAL PROFILE", fontsize=12)
    p9.insert_text((50, 140), "Full Stack Developer with experience building scalable enterprise SaaS systems.", fontsize=10)
    p9.insert_text((50, 170), "TECHNICAL SKILLS", fontsize=12)
    p9.insert_text((50, 190), "Python, Django, React, TypeScript, PostgreSQL, Docker, AWS", fontsize=10)
    p9.insert_text((50, 220), "EXPERIENCE", fontsize=12)
    p9.insert_text((50, 240), "Senior Backend Engineer — ScaleFlow Technologies (2022 - Present)", fontsize=11)
    doc9.save(os.path.join(samples_dir, "resume_invalid_email.pdf"))
    doc9.close()

    # 10. resume_senior_dev.pdf (Clean Full-Stack Developer Resume -> Completed)
    doc10 = pymupdf.open()
    p10 = doc10.new_page(width=595, height=842)
    p10.insert_text((50, 60), "Priya Sharma", fontsize=22, color=(0.1, 0.3, 0.3))
    p10.insert_text((50, 80), "Email: priya.sharma@cloudtech.io | Phone: +91 99887 76655 | Pune, India", fontsize=10, color=(0.3, 0.3, 0.3))
    p10.draw_line((50, 95), (545, 95), color=(0.2, 0.5, 0.5), width=1.5)
    p10.insert_text((50, 120), "SUMMARY", fontsize=12)
    p10.insert_text((50, 140), "Lead Cloud Architect with deep expertise in distributed microservices and container orchestration.", fontsize=10)
    p10.insert_text((50, 170), "CORE SKILLS & TECHNOLOGIES", fontsize=12)
    p10.insert_text((50, 190), "Go, Python, Kubernetes, Terraform, AWS, Docker, gRPC, Redis, Kafka", fontsize=10)
    p10.insert_text((50, 220), "EXPERIENCE", fontsize=12)
    p10.insert_text((50, 240), "Principal Cloud Architect — Infinicloud Systems (2020 - Present)", fontsize=11)
    p10.insert_text((50, 280), "EDUCATION", fontsize=12)
    p10.insert_text((50, 300), "Master of Technology in Software Engineering, IIT Bombay", fontsize=10)
    doc10.save(os.path.join(samples_dir, "resume_senior_dev.pdf"))
    doc10.close()

    # 11. project_proposal.pdf (Project Architecture Proposal -> Completed 'Other')
    doc11 = pymupdf.open()
    p11 = doc11.new_page(width=595, height=842)
    p11.insert_text((50, 60), "Technical Project Proposal: Automated Data Ingestion Pipeline", fontsize=16, color=(0.1, 0.1, 0.1))
    p11.insert_text((50, 85), "Author: Cloud Architecture Core Team | Version: 2.1 | Date: October 2026", fontsize=9, color=(0.4, 0.4, 0.4))
    p11.draw_line((50, 105), (545, 105), color=(0.8, 0.8, 0.8), width=1)
    p11.insert_text((50, 130), "Executive Summary:", fontsize=12)
    p11.insert_text((50, 155), "This proposal outlines the high-throughput architecture for processing multi-format enterprise documents.", fontsize=10)
    p11.insert_text((50, 175), "The solution leverages optical character recognition, NLP entity extraction, and stateful workflow control.", fontsize=10)
    p11.insert_text((50, 205), "Key Milestones & Estimated Timelines:", fontsize=12)
    p11.insert_text((50, 230), "Phase 1: Ingestion Gateway & SHA-256 Deduplication (Week 1-2)", fontsize=10)
    p11.insert_text((50, 250), "Phase 2: Document Classification and Rule Engine (Week 3-4)", fontsize=10)
    p11.insert_text((50, 270), "Phase 3: Human Review Queue & Audit Logging System (Week 5)", fontsize=10)
    doc11.save(os.path.join(samples_dir, "project_proposal.pdf"))
    doc11.close()

    # 12. nda_agreement.pdf (Mutual Non-Disclosure Agreement -> Completed 'Other')
    doc12 = pymupdf.open()
    p12 = doc12.new_page(width=595, height=842)
    p12.insert_text((50, 60), "MUTUAL NON-DISCLOSURE AGREEMENT", fontsize=18, color=(0.1, 0.1, 0.1))
    p12.insert_text((50, 85), "Confidentiality and Proprietary Information Protection Agreement", fontsize=10, color=(0.4, 0.4, 0.4))
    p12.draw_line((50, 105), (545, 105), color=(0.8, 0.8, 0.8), width=1)
    p12.insert_text((50, 130), "1. Parties and Purpose:", fontsize=12)
    p12.insert_text((50, 155), "This Agreement is entered into by and between the Disclosing Party and the Receiving Party.", fontsize=10)
    p12.insert_text((50, 175), "The purpose is to explore prospective technology collaboration and commercial partnership.", fontsize=10)
    p12.insert_text((50, 205), "2. Confidential Information Definition:", fontsize=12)
    p12.insert_text((50, 230), "All technical designs, software code, algorithmic models, and business specifications shall remain confidential.", fontsize=10)
    p12.insert_text((50, 250), "Term of Obligation: Three (3) years from the effective date of execution.", fontsize=10)
    doc12.save(os.path.join(samples_dir, "nda_agreement.pdf"))
    doc12.close()

