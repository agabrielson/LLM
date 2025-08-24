#!/usr/bin/env python3

import os
import time
import shutil
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


class LLMDocumentIndexer:
    def __init__(self, 
                 data_dir="articlesV2/",
                 processed_dir="processed/",
                 model_name="llama3",
                 chunk_size=1000,
                 chunk_overlap=200,
                 vectorstore_dir="./chroma_db"):
        self.data_dir = data_dir
        self.processed_dir=processed_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore_dir = vectorstore_dir

        self.documents = []
        self.text_chunks = []
        self.embeddings = None
        self.vectorstore = None

    def load_documents(self):
        """Load documents from the directory (defaults to self.data_dir)."""

        loader_cls = PyPDFLoader if self.data_dir.endswith(".pdf") else TextLoader
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*",
            loader_cls=loader_cls,
            silent_errors=True
        )
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} documents from {self.data_dir}")

    def split_documents(self):
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split into {len(self.text_chunks)} chunks")

    def create_embeddings(self):
        """Create embeddings for the document chunks."""
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        print(f"Created embeddings using model: {self.model_name}")

    def create_vectorstore(self):
        """Create a vector store from document embeddings."""
        if not self.embeddings or not self.text_chunks:
            raise ValueError("Embeddings or text chunks not initialized.")
        self.vectorstore = Chroma.from_documents(
            self.text_chunks, 
            self.embeddings, 
            persist_directory=self.vectorstore_dir
        )
        print(f"Vector store created and persisted at {self.vectorstore_dir}")

    def add_documents(self, new_documents):
        """Add new documents to the in-memory vectorstore."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized.")
        self.vectorstore.add_documents(documents=new_documents)
        print(f"Added {len(new_documents)} new documents to the vector store")

    def add_documents_with_progress(self, new_documents, batch_size=10):
        """Add new documents with a tqdm progress bar and timer."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized.")

        total = len(new_documents)
        start_time = time.time()

        with tqdm(total=total, desc="Adding documents", unit="doc") as pbar:
            for i in range(0, total, batch_size):
                batch = new_documents[i:i+batch_size]
                self.vectorstore.add_documents(batch)
                pbar.update(len(batch))

        total_time = time.time() - start_time
        print(f"\n✅ Completed {total} documents in {total_time:.2f} seconds")

    def add_to_existing_db(self, batch_size=20):
        """
        Load new documents from a directory and add them to an existing vectorstore.
        Splits and embeds the new documents automatically.
        After adding, move the original files into processed_dir to avoid re-processing.
        """
        if not self.vectorstore:
            # Load existing vectorstore if not already loaded
            self.embeddings = OllamaEmbeddings(model=self.model_name)
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_dir, 
                embedding_function=self.embeddings
            )
            print("Loaded existing vectorstore from disk.")

        # Load new documents
        loader_cls = PyPDFLoader if self.data_dir.endswith(".pdf") else TextLoader
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*",
            loader_cls=loader_cls,
            silent_errors=True
        )
        new_documents = loader.load()
        print(f"Loaded {len(new_documents)} new documents from {self.data_dir}")

        if not new_documents:
            print("No new documents found. Exiting.")
            return

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        new_chunks = text_splitter.split_documents(new_documents)
        print(f"Split new documents into {len(new_chunks)} chunks")

        # Add to vectorstore with progress
        self.add_documents_with_progress(new_chunks, batch_size=batch_size)

        # --- Move processed files ---
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, self.data_dir)  # preserve subfolders
                dest_path = os.path.join(self.processed_dir, rel_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} → {dest_path}")

        print(f"\n✅ All files moved to {self.processed_dir}")

    def index_all(self):
        """
        Run the full pipeline: load, split, embed, and create vectorstore.
        If data_dir is provided, it overrides self.data_dir.
        """
        self.load_documents()
        self.split_documents()
        self.create_embeddings()
        self.create_vectorstore()


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    #data_dir = "articlesV2/"
    data_dir = "download_clean/"
    proc_dir = "prod_data/"

    indexer = LLMDocumentIndexer(
        data_dir=data_dir,
        processed_dir = proc_dir,
        model_name="llama3",
        chunk_size=1000,
        chunk_overlap=200,
        vectorstore_dir="./chroma_db"
    )

    # Initial indexing
    # indexer.index_all()

    # Add new documents to existing vectorstore with progress bar, then move them
    indexer.add_to_existing_db()
