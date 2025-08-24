#!/usr/bin/env python3

import os
import io
from multiprocessing import Pool, cpu_count
from PIL import Image
import pytesseract
import pymupdf  # aka fitz


class PDFOCRProcessor:
    def __init__(self, input_dir="../Articles", output_dir="articlesV2", num_processes=None, dpi=200):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.num_processes = num_processes or cpu_count()
        self.dpi = dpi  # Adjust for speed vs quality

    def ocr_page(self, page):
        # Render page to image
        pix = page.get_pixmap(dpi=self.dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # OCR the image
        text = pytesseract.image_to_string(img, lang="eng", config="--psm 1")
        return text

    def pdf_to_text(self, pdf_path, output_txt_path):
        try:
            doc = pymupdf.open(pdf_path)
            with open(output_txt_path, "w", encoding="utf-8") as f:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = self.ocr_page(page)
                    f.write(text + "\n")
            doc.close()
            print(f"✅ Processed {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(pdf_path)}: {e}")

    def process_file(self, pdf_filename):
        pdf_path = os.path.join(self.input_dir, pdf_filename)
        txt_filename = os.path.splitext(pdf_filename)[0] + ".txt"
        txt_path = os.path.join(self.output_dir, txt_filename)
        self.pdf_to_text(pdf_path, txt_path)

    def run(self):
        pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".pdf")]
        print(f"🚀 Starting OCR on {len(pdf_files)} PDFs with {self.num_processes} processes...")

        # Multiprocessing pool
        with Pool(processes=self.num_processes) as pool:
            pool.map(self.process_file, pdf_files)

        print(f"🎉 All PDFs processed. Output saved in: {self.output_dir}")


if __name__ == "__main__":
    processor = PDFOCRProcessor(input_dir="../Articles", output_dir="articlesV2", num_processes=4, dpi=200)
    processor.run()
