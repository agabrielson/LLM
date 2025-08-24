#!/usr/bin/env python3

import os
import re
import nltk
import enchant
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (only needed once)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class TextCleaner:
    def __init__(self, input_dir="articlesV2raw/", output_dir="articlesV2/", num_workers=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # NLP tools
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Spell checker (English dictionary)
        self.spell_dict = enchant.Dict("en_US")

        # Workers for multiprocessing
        self.num_workers = num_workers or cpu_count()

    # ---------------- Cleaning steps ----------------
    def remove_html_and_links(self, text: str) -> str:
        # Remove everything inside HTML tags including the tags themselves
        text = re.sub(r"<[^>]*>.*?</[^>]*>", " ", text, flags=re.DOTALL)
        # Remove any remaining self-closing tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove links
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        # Normalize whitespace
        return re.sub(r"\s+", " ", text).strip()

    def remove_special_and_emojis(self, text: str) -> str:
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^a-z\s]", "", text)

    def spell_check(self, text: str) -> str:
        """Correct misspelled words using enchant dictionary"""
        words = text.split()
        corrected = []
        for w in words:
            if self.spell_dict.check(w):
                corrected.append(w)
            else:
                suggestions = self.spell_dict.suggest(w)
                corrected.append(suggestions[0] if suggestions else w)
        return " ".join(corrected)

    def apply_lemmatizer_stopwords(self, text: str) -> str:
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words
        ]
        return " ".join(cleaned_words)

    def clean_text(self, text: str) -> str:
        text = self.remove_html_and_links(text)
        text = self.remove_special_and_emojis(text)
        text = self.remove_punctuation(text)
        text = text.lower()
        text = self.spell_check(text)  # ✅ enchant, no indexer dependency
        text = self.apply_lemmatizer_stopwords(text)
        return text

    import os

    def update_file_extension(self, filename: str, desired_ext: str) -> str:
        """
        Ensures that the filename has the desired file extension.
        
        Args:
            filename (str): Original filename.
            desired_ext (str): Desired extension (with or without dot, e.g., 'txt' or '.txt').
            
        Returns:
            str: Filename with the desired extension.
        """
        # Normalize extension to start with a dot
        if not desired_ext.startswith("."):
            desired_ext = f".{desired_ext}"
        
        base, ext = os.path.splitext(filename)
        
        # Replace if different
        if ext.lower() != desired_ext.lower():
            filename = f"{base}{desired_ext}"
        
        return filename


    # ---------------- File Processing ----------------
    def process_file(self, filename: str):
        input_path = os.path.join(self.input_dir, filename)
        out_filename = self.update_file_extension(filename, ".txt")
        output_path = os.path.join(self.output_dir, out_filename)

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_text = self.clean_text(raw_text)

            # need to change file extension...
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"✅ Processed {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    def run(self, extension = ".txt"):
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(extension)]

        print(f"🚀 Starting processing with {self.num_workers} workers...")
        with Pool(processes=self.num_workers) as pool:
            pool.map(self.process_file, files)

        print(f"🎉 All files processed. Output saved in: {self.output_dir}")


if __name__ == "__main__":
    #cleaner = TextCleaner(input_dir="articlesV2raw/", output_dir="articlesV2/", num_workers=4)
    #cleaner.run(extension=".txt")
    cleaner = TextCleaner(input_dir="downloaded_sites/", output_dir="download_clean/", num_workers=4)
    cleaner.run(extension=".html")
