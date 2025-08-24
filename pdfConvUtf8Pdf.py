#!/usr/bin/env python3

import os
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER

input_dir = "../../Shooting/Articles"
output_dir = "articles/"

os.makedirs(output_dir, exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def write_text_to_pdf(text, output_path):
    c = canvas.Canvas(output_path, pagesize=LETTER)
    width, height = LETTER
    lines = text.split('\n')
    y = height - 72  # Start near top of page

    for line in lines:
        if y < 72:
            c.showPage()
            y = height - 72
        c.drawString(72, y, line)
        y -= 14
    c.save()

# Process all PDFs in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        text = extract_text_from_pdf(input_path)
        write_text_to_pdf(text, output_path)

print("All PDFs have been re-encoded to UTF-8 compatible text-based PDFs.")
