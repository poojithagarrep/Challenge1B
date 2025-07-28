# main_runner.py

import os
import json
from datetime import datetime
from main import intelligent_document_analyst

# --- Directories ---
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdf")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Persona & Job Description ---
input_json_path = os.path.join(DATA_DIR, "input.json")
with open(input_json_path, "r", encoding="utf-8") as f:
    input_data = json.load(f)
    persona = input_data.get("persona", "").strip()
    job = input_data.get("job", "").strip()

if not persona or not job:
    raise ValueError("Both 'persona' and 'job' must be specified in data/input.json")

# --- Load PDFs from data/pdf/ ---
pdf_paths = [
    os.path.join(PDF_DIR, f)
    for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
]

if not (3 <= len(pdf_paths) <= 10):
    raise ValueError("Please ensure 3 to 10 PDF files exist in the 'data/pdf/' folder.")

# --- Process PDFs ---
result_json = intelligent_document_analyst(
    document_paths=pdf_paths,
    persona_definition=persona,
    job_to_be_done=job,
    debug=False
)

# --- Save Result to o/p/ Folder ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"document_analysis_{timestamp}.json")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(result_json)

print(f"✅ Output saved to: {output_path}")
