"""
Central Configuration for AI Document Intelligence Platform.
Defines UI themes, supported file formats, classification dictionaries,
and pattern matching configurations.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"

# Visual Identity / Theme Palette
THEME = {
    "primary": "#0B1020",         # Deep obsidian navy
    "secondary": "#111A33",       # Midnight card surface
    "accent_cyan": "#00D9FF",     # Electric cyan
    "accent_purple": "#7C3AED",   # Deep royal violet
    "bg_light": "#F8FAFC",        # Crisp clean canvas
    "surface_white": "#FFFFFF",   # Pure white
    "text_dark": "#0F172A",       # Slate 900
    "text_muted": "#64748B",      # Slate 500
    "success": "#10B981",         # Emerald green
    "warning": "#F59E0B",         # Amber
    "danger": "#EF4444",          # Crimson
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
}

# File Upload Constraints
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Document Classes
DOC_TYPE_INVOICE = "Invoice"
DOC_TYPE_RESUME = "Resume"
DOC_TYPE_OTHER = "Other"

# Keyword Dictionaries with Weights
INVOICE_KEYWORDS = {
    "invoice": 3.5,
    "invoice number": 4.5,
    "tax invoice": 4.5,
    "bill to": 3.5,
    "billed to": 3.5,
    "total amount": 4.0,
    "amount due": 3.5,
    "balance due": 3.5,
    "subtotal": 3.0,
    "due date": 3.0,
    "payment terms": 2.5,
    "purchase order": 2.5,
    "unit price": 2.5,
    "vat": 2.0,
    "gst": 2.0,
    "qty": 2.0,
    "quantity": 2.0,
    "remit to": 2.5,
    "vendor": 2.0,
    "client": 1.5,
    "item": 1.0,
    "pkr": 2.5,
    "usd": 2.0,
}

RESUME_KEYWORDS = {
    "curriculum vitae": 4.5,
    "resume": 4.0,
    "work experience": 4.0,
    "professional experience": 4.0,
    "experience": 3.0,
    "education": 3.5,
    "skills": 3.5,
    "technical skills": 3.5,
    "employment history": 3.5,
    "projects": 2.5,
    "summary": 2.5,
    "professional summary": 3.0,
    "certifications": 3.0,
    "bachelor": 2.5,
    "master": 2.5,
    "degree": 2.0,
    "university": 2.5,
    "college": 2.0,
    "gpa": 2.0,
    "internship": 2.5,
    "achievements": 2.0,
    "languages": 1.5,
    "linkedin": 2.5,
    "github": 2.5,
}

# Extensive Tech Skills Taxonomy for Resume Extraction
SKILLS_TAXONOMY = {
    "Programming Languages": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "C", 
        "Go", "Rust", "Kotlin", "Swift", "R", "PHP", "Ruby", "MATLAB", "SQL"
    ],
    "AI & Machine Learning": [
        "Machine Learning", "Deep Learning", "Natural Language Processing", "NLP",
        "Computer Vision", "PyTorch", "TensorFlow", "Keras", "Scikit-Learn",
        "Pandas", "NumPy", "SciPy", "OpenCV", "Hugging Face", "Transformers",
        "LLMs", "LangChain", "LlamaIndex", "XGBoost", "LightGBM", "Data Science"
    ],
    "Web & Full-Stack": [
        "React", "Next.js", "Vue.js", "Angular", "Node.js", "Express.js",
        "FastAPI", "Flask", "Django", "HTML5", "CSS3", "Tailwind CSS",
        "REST API", "GraphQL", "Streamlit", "WebSockets"
    ],
    "Cloud & DevOps": [
        "AWS", "Amazon Web Services", "Azure", "Google Cloud", "GCP",
        "Docker", "Kubernetes", "CI/CD", "GitHub Actions", "Terraform",
        "Linux", "Bash", "Git", "Nginx", "Microservices"
    ],
    "Databases & Big Data": [
        "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Elasticsearch",
        "Apache Spark", "Hadoop", "Kafka", "Snowflake", "BigQuery"
    ],
    "Analytics & Visualization": [
        "Tableau", "Power BI", "Matplotlib", "Seaborn", "Plotly", "Excel"
    ]
}

# Regex Patterns for Parsing
PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b",
    "invoice_number": [
        r"(?i)\binvoice\s*(?:number|no\.?|num\.?|#)\s*[:#\-]?\s*([A-Za-z0-9\-_/]{3,30})\b",
        r"(?i)\binv[\s.:#\-]+([A-Za-z0-9\-_/]{3,30})\b",
        r"(?i)\bbill\s*(?:number|no\.?|#)\s*[:#\-]?\s*([A-Za-z0-9\-_/]{3,30})\b",
        r"(?i)#\s*([A-Z0-9\-_]{4,20})\b",
    ],
    "date": [
        r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b",
        r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
    ],
    "currency_amount": [
        r"(?i)(?:total(?:\s+amount)?|amount\s+due|balance\s+due|net\s+total)[\s:]*([A-Z]{3}|\$|€|£|₹|Rs\.?|PKR)\s*([\d,]+(?:\.\d{2})?)",
        r"(?i)([A-Z]{3}|\$|€|£|₹|Rs\.?|PKR)\s*([\d,]+(?:\.\d{2})?)\s*(?:total|amount\s+due|due)",
        r"(?i)(?:total(?:\s+amount)?|amount\s+due|balance\s+due)[\s:]*([\d,]+(?:\.\d{2})?)\s*([A-Z]{3}|\$|€|£|₹|Rs\.?|PKR)",
        r"(?i)(?:total(?:\s+amount)?|grand\s+total)[\s:]*([\d,]+(?:\.\d{2})?)",
    ]
}
