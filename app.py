"""
AI Document Intelligence & Workflow Platform — Streamlit Platform Entrypoint
Seamlessly launches the AI Document Intelligence Platform (Week 4 Active).
"""

import sys
import os
from pathlib import Path

# Add week-04 directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
WEEK4_DIR = CURRENT_DIR / "week-04"
WEEK3_DIR = CURRENT_DIR / "week-03"
WEEK2_DIR = CURRENT_DIR / "week-02"

if WEEK4_DIR.exists():
    target_dir = WEEK4_DIR
elif WEEK3_DIR.exists():
    target_dir = WEEK3_DIR
elif WEEK2_DIR.exists():
    target_dir = WEEK2_DIR
else:
    target_dir = CURRENT_DIR

if str(target_dir) not in sys.path:
    sys.path.insert(0, str(target_dir))

# Switch working directory context so relative paths in active week work seamlessly
os.chdir(target_dir)

target_app_path = target_dir / "app.py"
if target_app_path.exists():
    with open(target_app_path, encoding="utf-8") as f:
        code = compile(f.read(), str(target_app_path), "exec")
        exec(code, globals())
else:
    import streamlit as st
    st.error("Application files not found. Please verify the repository structure.")
