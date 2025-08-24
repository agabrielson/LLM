#!/usr/bin/env python3

import os
import shutil
import time
import argparse
from tqdm import tqdm
import torch

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class LLMDocumentIndexerV2:
    def __init__(self, 
                 data_dir="articlesV2/",
                 processed_dir="processed/",
                 model_name="llama3",
                 embedding_backend="ollama",   # "ollama" or "huggingface"
                 device=None,                  # auto-detect if None
                 chunk_size=1000,
                 chunk_overlap=200,
                 vectorstore_dir="./chroma_db"):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.model_name = model_name
        self.embedding_backend = embedding_backend

        # 🔍 Auto-detect device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 📢 Print device info
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_mem = round(props.total_memory / (1024**3), 2)  # GB
            print(f"🚀 CUDA detected: {gpu_name} ({total_mem} GB VRAM)")
        else:
            print("⚡ Using CPU (no CUDA GPU detected)")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore_dir = vectorstore_dir

        self.documents = []
        self.text_chunks = []
        self.embeddings = None
        self.vectorstore = None

    def load_documents(self):
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split into {len(self.text_chunks)} chunks")

    def create_embeddings(self):
        """Create embeddings with Ollama (CPU) or HuggingFace (CPU/GPU)."""
        if self.embedding_backend == "ollama":
            self.embeddings = OllamaEmbeddings(model=self.model_name)
            print(f"✅ Using Ollama embeddings with model: {self.model_name}")
        elif self.embedding_backend == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device}
            )
            print(f"✅ Using HuggingFace embeddings ({self.model_name}) on {self.device}")
        else:
            raise ValueError("Unknown embedding backend. Use 'ollama' or 'huggingface'.")

    def create_vectorstore(self):
        if not self.embeddings or not self.text_chunks:
            raise ValueError("Embeddings or text chunks not initialized.")
        self.vectorstore = Chroma.from_documents(
            self.text_chunks, 
            self.embeddings, 
            persist_directory=self.vectorstore_dir
        )
        print(f"Vector store created and persisted at {self.vectorstore_dir}")

    def add_documents(self, new_documents):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized.")
        self.vectorstore.add_documents(documents=new_documents)
        print(f"Added {len(new_documents)} new documents to the vector store")

    def add_documents_with_progress(self, new_documents, batch_size=10):
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
        if not self.vectorstore:
            self.create_embeddings()
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_dir, 
                embedding_function=self.embeddings
            )
            print("Loaded existing vectorstore from disk.")

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        new_chunks = text_splitter.split_documents(new_documents)
        print(f"Split new documents into {len(new_chunks)} chunks")

        self.add_documents_with_progress(new_chunks, batch_size=batch_size)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, self.data_dir)
                dest_path = os.path.join(self.processed_dir, rel_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} → {dest_path}")

        print(f"\n✅ All files moved to {self.processed_dir}")

    def index_all(self):
        self.load_documents()
        self.split_documents()
        self.create_embeddings()
        self.create_vectorstore()

    def reset_vectorstore(self):
        """Delete the existing vectorstore directory to start fresh."""
        if os.path.exists(self.vectorstore_dir):
            shutil.rmtree(self.vectorstore_dir)
            print(f"🗑️  Reset vectorstore at {self.vectorstore_dir}")
        else:
            print("⚠️  No existing vectorstore found to reset.")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents with GPU/CPU, backend, and custom options")

    parser.add_argument("--gpu", action="store_true", help="Force use of NVIDIA GPU (CUDA)")
    parser.add_argument("--cpu", action="store_true", help="Force use of CPU only")
    parser.add_argument("--backend", type=str, default="huggingface",
                        choices=["huggingface", "ollama"],
                        help="Embedding backend to use (default: huggingface)")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model name (HF model or Ollama model)")
    parser.add_argument("--data-dir", type=str, default="download_clean/",
                        help="Directory containing documents to index")
    parser.add_argument("--processed-dir", type=str, default="prod_data/",
                        help="Directory to move processed documents into")
    parser.add_argument("--vectorstore-dir", type=str, default="./chroma_db",
                        help="Directory where vectorstore is persisted")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Batch size for adding documents to vectorstore (default: 20)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Chunk size for text splitting (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Overlap size between chunks (default: 200)")
    parser.add_argument("--reset-db", action="store_true",
                        help="Reset (delete) the existing vectorstore before indexing")
    parser.add_argument("--mode", type=str, default="incremental",
                        choices=["full", "incremental"],
                        help="Indexing mode: full = rebuild vectorstore, incremental = add new docs")

    args = parser.parse_args()

    # Decide device
    if args.gpu:
        device = "cuda"
    elif args.cpu:
        device = "cpu"
    else:
        device = None  # auto-detect

    indexer = LLMDocumentIndexerV2(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        model_name=args.model,
        embedding_backend=args.backend,
        device=device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vectorstore_dir=args.vectorstore_dir
    )

    # Reset DB if requested
    if args.reset_db:
        indexer.reset_vectorstore()

    # Run according to mode
    if args.mode == "full":
        print("⚡ Running full indexing (rebuild vectorstore)")
        indexer.index_all()
    elif args.mode == "incremental":
        print("⚡ Running incremental indexing (add new documents)")
        indexer.add_to_existing_db(batch_size=args.batch_size)
