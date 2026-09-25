# Screenshots Guide — AI Document Intelligence MVP

This directory stores demonstration screenshots of the application for evaluation and documentation.

## Suggested Demonstration Screenshots

### 1. Upload & Initial State (`01_upload_screen.png`)
- **What to show**: The main Streamlit interface with the title header, sidebar system status indicators (PDF Extraction, OCR Engine, Classification, Field Extraction), and the clean file upload area.
- **Why it matters**: Demonstrates UI/UX design, clear status transparency, and system readiness.

### 2. Invoice Document Analysis (`02_invoice_result.png`)
- **What to show**: The processed result of an invoice document (e.g. `samples/invoice_1.pdf` or `samples/invoice_2.pdf`).
- **Elements highlighted**:
  - File metadata (filename, size, type)
  - Processing checkmarks (File accepted, Text extracted, Document identified, Fields extracted)
  - Document Type badge: **Invoice**
  - Detection Method: **Rule-Based Keyword Classification**
  - Extracted fields: Invoice Number, Date, Company Name, Total Amount.

### 3. Resume Document Analysis (`03_resume_result.png`)
- **What to show**: The processed result of a candidate resume (e.g. `samples/resume_1.pdf`).
- **Elements highlighted**:
  - Document Type badge: **Resume**
  - Extracted candidate fields: Name, Email, Phone, Skills tags.

### 4. Extracted Raw Text & Download (`04_extracted_text.png`)
- **What to show**: The expanded text view showing selectable raw text extracted from PyMuPDF, word count statistics, and the JSON/Text download buttons.

---

## How to Capture & Add Screenshots

1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser at `http://localhost:8501`.
3. Use the **Quick Sample Document** dropdown in the sidebar to load:
   - `invoice_1.pdf`
   - `resume_1.pdf`
4. Use `Win + Shift + S` (Windows Snipping Tool) to capture each view.
5. Save the images into this directory as:
   - `screenshots/01_upload_screen.png`
   - `screenshots/02_invoice_result.png`
   - `screenshots/03_resume_result.png`
   - `screenshots/04_extracted_text.png`
