"""
AI Document Intelligence Platform — Streamlit Platform Entrypoint
Seamlessly launches the AI Document Intelligence Platform.
"""

import sys
from pathlib import Path

# Add week-02 directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
WEEK2_DIR = CURRENT_DIR / "week-02"

if str(WEEK2_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK2_DIR))

# Execute the Week 2 application
week2_app_path = WEEK2_DIR / "app.py"
if week2_app_path.exists():
    with open(week2_app_path, encoding="utf-8") as f:
        code = compile(f.read(), str(week2_app_path), "exec")
        exec(code, globals())
else:
    import streamlit as st
    st.error("Week 2 application files not found. Please verify the repository structure.")
