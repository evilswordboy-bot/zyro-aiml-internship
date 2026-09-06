"""
Sample Document Generator for AI Document Intelligence Platform.
Generates genuine, authentic PDF test documents matching the standard document specification:
1. invoice_001.pdf: Matches official standard invoice example (INV-1024, ABC Technologies, PKR 125,000)
2. invoice_002.pdf: Multi-item commercial cloud invoice (INV-2026-889, $4,850.00)
3. resume_001.pdf: Senior AI/ML Engineer resume with structured skills and contact info
4. sample_other.pdf: Unstructured project meeting notes & guidelines (tests 'Other' classification)
"""

from pathlib import Path
import pymupdf
from .config import SAMPLES_DIR


def generate_samples(target_dir: Path = SAMPLES_DIR) -> dict[str, Path]:
    """Generates sample test documents if they do not already exist."""
    target_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    # 1. Official Sample Invoice #1 (ABC Technologies / PKR 125,000)
    inv1_path = target_dir / "invoice_001.pdf"
    if not inv1_path.exists():
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)  # A4

        # Decorative header band
        page.draw_rect(pymupdf.Rect(0, 0, 595, 80), color=None, fill=(11/255, 16/255, 32/255))
        page.insert_text((40, 48), "ABC TECHNOLOGIES", fontsize=22, color=(1, 1, 1), fontname="helv")
        page.insert_text((40, 68), "Enterprise AI & Cloud Software Solutions", fontsize=10, color=(0/255, 217/255, 255/255), fontname="helv")

        # Invoice Header
        page.insert_text((400, 48), "TAX INVOICE", fontsize=18, color=(1, 1, 1), fontname="helv")
        page.insert_text((400, 66), "Official Invoice Document", fontsize=9, color=(0.8, 0.8, 0.8), fontname="helv")

        # Metadata
        page.insert_text((40, 120), "INVOICE DETAILS", fontsize=12, color=(0.1, 0.1, 0.1), fontname="helv")
        page.draw_line(pymupdf.Point(40, 125), pymupdf.Point(555, 125), color=(0.8, 0.8, 0.8), width=1)

        page.insert_text((40, 150), "Invoice Number: INV-1024", fontsize=11, fontname="helv")
        page.insert_text((40, 170), "Date: 02-09-2026", fontsize=11, fontname="helv")
        page.insert_text((40, 190), "Due Date: 16-09-2026", fontsize=11, fontname="helv")
        page.insert_text((40, 210), "Payment Terms: Net 14 Days", fontsize=11, fontname="helv")

        page.insert_text((320, 150), "BILL TO:", fontsize=11, fontname="helv")
        page.insert_text((320, 170), "ABC Technologies Global Ltd.", fontsize=11, fontname="helv")
        page.insert_text((320, 190), "Corporate Towers, Floor 14", fontsize=10, fontname="helv")
        page.insert_text((320, 210), "contact@abctechnologies.com", fontsize=10, fontname="helv")

        # Items Table
        page.draw_rect(pymupdf.Rect(40, 250, 555, 275), color=None, fill=(0.94, 0.95, 0.98))
        page.insert_text((50, 267), "Description", fontsize=10, color=(0.1, 0.1, 0.1), fontname="helv")
        page.insert_text((350, 267), "Qty", fontsize=10, color=(0.1, 0.1, 0.1), fontname="helv")
        page.insert_text((420, 267), "Rate", fontsize=10, color=(0.1, 0.1, 0.1), fontname="helv")
        page.insert_text((480, 267), "Amount", fontsize=10, color=(0.1, 0.1, 0.1), fontname="helv")

        page.insert_text((50, 305), "AI Document Intelligence MVP Development", fontsize=10, fontname="helv")
        page.insert_text((350, 305), "1", fontsize=10, fontname="helv")
        page.insert_text((420, 305), "PKR 75,000", fontsize=10, fontname="helv")
        page.insert_text((480, 305), "PKR 75,000", fontsize=10, fontname="helv")

        page.insert_text((50, 335), "Automated OCR & Classification Pipeline", fontsize=10, fontname="helv")
        page.insert_text((350, 335), "1", fontsize=10, fontname="helv")
        page.insert_text((420, 335), "PKR 50,000", fontsize=10, fontname="helv")
        page.insert_text((480, 335), "PKR 50,000", fontsize=10, fontname="helv")

        # Total Box
        page.draw_rect(pymupdf.Rect(320, 380, 555, 430), color=(0.1, 0.1, 0.2), fill=(0.96, 0.97, 1.0))
        page.insert_text((340, 400), "Subtotal: PKR 125,000", fontsize=10, fontname="helv")
        page.insert_text((340, 420), "Total Amount: PKR 125,000", fontsize=12, color=(0.05, 0.1, 0.3), fontname="helv")

        # Footer
        page.insert_text((40, 750), "Payment Remittance: Habib Bank Limited | Account: 0029-1928-1100", fontsize=9, color=(0.4, 0.4, 0.4), fontname="helv")
        page.insert_text((40, 765), "Thank you for partnering with ABC Technologies.", fontsize=9, color=(0.4, 0.4, 0.4), fontname="helv")

        doc.save(str(inv1_path))
        doc.close()
    generated_files["invoice_001"] = inv1_path

    # 2. Sample Invoice #2 (CloudScale Inc / $4,850.00)
    inv2_path = target_dir / "invoice_002.pdf"
    if not inv2_path.exists():
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)

        page.draw_rect(pymupdf.Rect(0, 0, 595, 75), color=None, fill=(17/255, 26/255, 51/255))
        page.insert_text((40, 45), "CLOUDSCALE SYSTEMS INC.", fontsize=20, color=(1, 1, 1), fontname="helv")
        page.insert_text((40, 63), "High Performance Cloud Computing & GPUs", fontsize=9, color=(0/255, 217/255, 255/255), fontname="helv")

        page.insert_text((420, 45), "INVOICE", fontsize=20, color=(1, 1, 1), fontname="helv")

        page.insert_text((40, 120), "Invoice Number: INV-2026-889", fontsize=11, fontname="helv")
        page.insert_text((40, 140), "Date: 15-08-2026", fontsize=11, fontname="helv")
        page.insert_text((40, 160), "Due Date: 30-08-2026", fontsize=11, fontname="helv")

        page.insert_text((320, 120), "Billed To: Nexus AI Corp", fontsize=11, fontname="helv")
        page.insert_text((320, 140), "Vendor: CloudScale Systems Inc.", fontsize=11, fontname="helv")
        page.insert_text((320, 160), "Client Account: NX-9021", fontsize=10, fontname="helv")

        page.draw_rect(pymupdf.Rect(40, 200, 555, 225), color=None, fill=(0.94, 0.95, 0.98))
        page.insert_text((50, 217), "Item Description", fontsize=10, fontname="helv")
        page.insert_text((480, 217), "Amount", fontsize=10, fontname="helv")

        page.insert_text((50, 250), "NVIDIA H100 Cloud GPU Cluster (400 Hours)", fontsize=10, fontname="helv")
        page.insert_text((480, 250), "$3,600.00", fontsize=10, fontname="helv")

        page.insert_text((50, 280), "High-Throughput NVMe Object Storage (10 TB)", fontsize=10, fontname="helv")
        page.insert_text((480, 280), "$1,250.00", fontsize=10, fontname="helv")

        page.draw_rect(pymupdf.Rect(320, 330, 555, 375), color=None, fill=(0.95, 0.97, 1.0))
        page.insert_text((340, 355), "Total Amount: $4,850.00", fontsize=12, color=(0.1, 0.15, 0.3), fontname="helv")

        doc.save(str(inv2_path))
        doc.close()
    generated_files["invoice_002"] = inv2_path

    # 3. Sample Resume #1 (Alex Morgan - Senior AI/ML Engineer)
    res1_path = target_dir / "resume_001.pdf"
    if not res1_path.exists():
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)

        # Header
        page.draw_rect(pymupdf.Rect(0, 0, 595, 90), color=None, fill=(11/255, 16/255, 32/255))
        page.insert_text((40, 45), "Alex Morgan", fontsize=24, color=(1, 1, 1), fontname="helv")
        page.insert_text((40, 68), "Senior AI/ML Engineer | Deep Learning & NLP Specialist", fontsize=11, color=(0/255, 217/255, 255/255), fontname="helv")

        # Contact Info
        page.draw_rect(pymupdf.Rect(40, 105, 555, 130), color=None, fill=(0.95, 0.96, 0.98))
        page.insert_text((50, 122), "Email: alex.morgan@example.com   |   Phone: +1-555-019-2834   |   GitHub: github.com/alexmorgan", fontsize=9, fontname="helv")

        # Summary
        page.insert_text((40, 160), "PROFESSIONAL SUMMARY", fontsize=12, color=(0.1, 0.1, 0.2), fontname="helv")
        page.draw_line(pymupdf.Point(40, 165), pymupdf.Point(555, 165), color=(0.8, 0.8, 0.8), width=1)
        summary_text = (
            "Results-driven Machine Learning Engineer with 5+ years of experience designing and deploying production "
            "document intelligence pipelines, transformers, and multimodal AI architectures. Passionate about MLOps."
        )
        page.insert_text((40, 185), summary_text, fontsize=9, fontname="helv")

        # Technical Skills
        page.insert_text((40, 230), "TECHNICAL SKILLS", fontsize=12, color=(0.1, 0.1, 0.2), fontname="helv")
        page.draw_line(pymupdf.Point(40, 235), pymupdf.Point(555, 235), color=(0.8, 0.8, 0.8), width=1)

        page.insert_text((40, 255), "Programming Languages: Python, TypeScript, Java, SQL, C++", fontsize=9, fontname="helv")
        page.insert_text((40, 275), "AI & Machine Learning: PyTorch, TensorFlow, Scikit-Learn, Hugging Face, Transformers, OpenCV, NLP", fontsize=9, fontname="helv")
        page.insert_text((40, 295), "Frameworks & Backend: FastAPI, Flask, Docker, Kubernetes, Streamlit", fontsize=9, fontname="helv")
        page.insert_text((40, 315), "Databases & Cloud: PostgreSQL, Redis, MongoDB, AWS, Google Cloud", fontsize=9, fontname="helv")

        # Work Experience
        page.insert_text((40, 360), "PROFESSIONAL EXPERIENCE", fontsize=12, color=(0.1, 0.1, 0.2), fontname="helv")
        page.draw_line(pymupdf.Point(40, 365), pymupdf.Point(555, 365), color=(0.8, 0.8, 0.8), width=1)

        page.insert_text((40, 390), "Lead AI Engineer - Synthetix Labs (2022 - Present)", fontsize=10, fontname="helv")
        page.insert_text((40, 408), "• Built scalable document understanding pipeline reducing manual processing time by 82%.", fontsize=9, fontname="helv")
        page.insert_text((40, 424), "• Trained custom domain classification models with 98.4% accuracy across millions of PDFs.", fontsize=9, fontname="helv")

        page.insert_text((40, 455), "Machine Learning Researcher - DataCore Systems (2020 - 2022)", fontsize=10, fontname="helv")
        page.insert_text((40, 473), "• Developed entity recognition models using PyTorch and Transformers.", fontsize=9, fontname="helv")
        page.insert_text((40, 489), "• Engineered high-throughput REST APIs handling 5,000 requests per minute with FastAPI.", fontsize=9, fontname="helv")

        # Education
        page.insert_text((40, 530), "EDUCATION", fontsize=12, color=(0.1, 0.1, 0.2), fontname="helv")
        page.draw_line(pymupdf.Point(40, 535), pymupdf.Point(555, 535), color=(0.8, 0.8, 0.8), width=1)
        page.insert_text((40, 555), "Bachelor of Science in Computer Science - University of Washington (2016 - 2020)", fontsize=9, fontname="helv")
        page.insert_text((40, 570), "Graduated with Honors | Focus on Artificial Intelligence & Distributed Systems", fontsize=9, fontname="helv")

        doc.save(str(res1_path))
        doc.close()
    generated_files["resume_001"] = res1_path

    # 4. Sample Other Document (Project Meeting Notes & Architecture Guidelines)
    other_path = target_dir / "sample_other.pdf"
    if not other_path.exists():
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)

        page.insert_text((40, 50), "ENGINEERING TEAM MEETING MINUTES & ARCHITECTURE MEMO", fontsize=14, color=(0.1, 0.1, 0.2), fontname="helv")
        page.insert_text((40, 70), "Date: September 4, 2026 | Facilitator: Technical Operations Team", fontsize=9, color=(0.4, 0.4, 0.4), fontname="helv")
        page.draw_line(pymupdf.Point(40, 80), pymupdf.Point(555, 80), color=(0.8, 0.8, 0.8), width=1)

        memo_lines = [
            "Project Scope and Objectives:",
            "The engineering initiative aims to optimize backend throughput and streamline infrastructure costs.",
            "Key discussion points from the sprint retrospective:",
            "",
            "1. Service Resilience: In the event of downstream API latency spikes, microservices must fall back",
            "   gracefully to cached database read replicas rather than crashing.",
            "2. Memory Footprint: Garbage collection thresholds have been tuned to limit resident memory to 1.5 GB.",
            "3. Security Audit: Periodic vulnerability scanning confirmed zero high or critical CVEs.",
            "",
            "Next Steps & Action Items:",
            "• Jordan: Complete Redis cache benchmarking by Friday.",
            "• Maya: Finalize Prometheus metrics dashboard for monitoring.",
            "• Chris: Coordinate cross-region replication testing in staging cluster."
        ]
        y = 110
        for line in memo_lines:
            page.insert_text((40, y), line, fontsize=10, fontname="helv")
            y += 20

        doc.save(str(other_path))
        doc.close()
    generated_files["sample_other"] = other_path

    return generated_files


if __name__ == "__main__":
    generated = generate_samples()
    print(f"Generated {len(generated)} sample documents in {SAMPLES_DIR}")
