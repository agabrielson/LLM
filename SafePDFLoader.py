from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document

class SafePDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            return PyPDFLoader(self.file_path).load()
        except Exception as e:
            print(f"Skipping {self.file_path}: {e}")
            return []

# Use SafePDFLoader in DirectoryLoader
#loader = DirectoryLoader(
#    path="./path_to_your_pdfs",
#    loader_cls=SafePDFLoader,
#    silent_errors=True  # Optional
#)

#documents = loader.load()
