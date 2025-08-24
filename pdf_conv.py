#!/usr/bin/env python3

import os
from PyPDF2 import PdfReader
import codecs

input_directory = "../../Shooting/Articles"
output_directory = "articlesTxt"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(input_directory, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        output_txt_path = os.path.join(output_directory, txt_filename)

        try:
            reader = PdfReader(pdf_path)
            text_content = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text_content += page.extract_text()

            # Encode to UTF-8
            with codecs.open(output_txt_path, "w", "utf-8") as f:
                f.write(text_content)
            print(f"Successfully extracted and converted '{filename}' to UTF-8.")
        except Exception as e:
            print(f"Error processing '{filename}': {e}")