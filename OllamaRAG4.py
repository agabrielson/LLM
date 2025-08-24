#!/usr/bin/env python3

"""
Live RAG + Incremental Indexing REPL with auto-refresh retrievers
- Automatically updates retrievers when new documents are added
- Interactive question/answer interface
- Supports Ollama or HuggingFace embeddings with GPU/CPU
"""

import os
import shutil
import time
import argparse
import torch
from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory

# ---------------- Defaults ----------------
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_EMBED_MODEL = "llama3"
DEFAULT_EMBED_BACKEND = "ollama"
DEFAULT_DATA_DIR = "download_clean/"
DEFAULT_PROCESSED_DIR = "prod_data/"
DEFAULT_VECTORSTORE_DIR = "./chroma_db"
_SESSION_STORE = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
    return _SESSION_STORE[session_id]

# ---------------- Document Indexer ----------------
class LLMDocumentIndexer:
    def __init__(self, data_dir=DEFAULT_DATA_DIR, processed_dir=DEFAULT_PROCESSED_DIR,
                 model_name=DEFAULT_EMBED_MODEL, embedding_backend=DEFAULT_EMBED_BACKEND,
                 device=None, chunk_size=1000, chunk_overlap=200, vectorstore_dir=DEFAULT_VECTORSTORE_DIR):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.model_name = model_name
        self.embedding_backend = embedding_backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore_dir = vectorstore_dir
        self.vectorstore = None
        self.embeddings = None
        print(f"💡 Device: {self.device} | Embedding backend: {self.embedding_backend}")

    def create_embeddings(self):
        if self.embedding_backend.lower() == "ollama":
            self.embeddings = OllamaEmbeddings(model=self.model_name)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={"device": self.device})

    def add_documents_incremental(self):
        loader_cls = PyPDFLoader if self.data_dir.endswith(".pdf") else TextLoader
        loader = DirectoryLoader(self.data_dir, glob="**/*", loader_cls=loader_cls, silent_errors=True)
        new_docs = loader.load()
        if not new_docs:
            return 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        new_chunks = splitter.split_documents(new_docs)

        if not self.vectorstore:
            self.create_embeddings()
            self.vectorstore = Chroma(persist_directory=self.vectorstore_dir, embedding_function=self.embeddings)

        for chunk in new_chunks:
            self.vectorstore.add_documents([chunk])

        # Move processed files
        os.makedirs(self.processed_dir, exist_ok=True)
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, self.data_dir)
                dest_path = os.path.join(self.processed_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)

        print(f"📦 Indexed and moved {len(new_docs)} new documents")
        return len(new_docs)

# ---------------- RAG System ----------------
class OllamaRAG3:
    def __init__(self, llm_model=DEFAULT_LLM_MODEL, embed_model=DEFAULT_EMBED_MODEL,
                 persist_directory=DEFAULT_VECTORSTORE_DIR, embed_backend=DEFAULT_EMBED_BACKEND, device=None):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.persist_directory = persist_directory
        self.embed_backend = embed_backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💡 Initializing RAG on {self.device} using {self.embed_backend}")
        self.llm = OllamaLLM(model=self.llm_model)
        self.chains = {}
        self.vectorstore = None
        self.retriever = None
        self._init_retriever()

    def _init_retriever(self):
        if self.embed_backend.lower() == "ollama":
            embeddings = OllamaEmbeddings(model=self.embed_model)
        else:
            embeddings = HuggingFaceEmbeddings(model_name=self.embed_model, model_kwargs={"device": self.device})

        self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        basic_db = self.vectorstore.as_retriever()
        adv_db = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.retriever = EnsembleRetriever([basic_db, adv_db], weights=[0.5, 0.5])

    def refresh_retriever(self):
        """Re-initialize retriever to reflect newly added documents"""
        self._init_retriever()
        # Clear chains so memory links to updated retriever
        for session_id in self.chains:
            memory = get_memory(session_id)
            self.chains[session_id] = ConversationalRetrievalChain.from_llm(
                llm=self.llm, retriever=self.retriever, memory=memory, return_source_documents=True
            )
        print("🔄 Retriever refreshed to include new documents")

    def _init_chain_for_session(self, session_id):
        memory = get_memory(session_id)
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, retriever=self.retriever, memory=memory, return_source_documents=True
        )
        self.chains[session_id] = chain
        return chain

    def ask(self, query, session_id="default"):
        if session_id not in self.chains:
            self._init_chain_for_session(session_id)
        chain = self.chains[session_id]
        result = chain.invoke({"question": query})
        sources = [os.path.basename(doc.metadata["source"]) for doc in result.get("source_documents", []) if hasattr(doc, "metadata") and "source" in doc.metadata]
        return {"answer": result.get("answer", ""), "sources": sources}

# ---------------- Live REPL ----------------
def start_live_repl(indexer, rag_system, poll_interval=5):
    print("💬 Live RAG REPL started. Type 'exit' to quit.")
    session_id = "live_session"

    while True:
        # Check for new documents
        new_count = indexer.add_documents_incremental()
        if new_count:
            rag_system.refresh_retriever()

        try:
            query = input("\nEnter question ('exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting REPL.")
            break

        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        res = rag_system.ask(query, session_id=session_id)
        print(f"\n📝 Answer: {res['answer']}")
        if res['sources']:
            print(f"📚 Sources: {res['sources']}")

        time.sleep(poll_interval)

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live RAG + Incremental Indexing with auto-refresh retriever")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--embed-backend", default=DEFAULT_EMBED_BACKEND, choices=["ollama","huggingface"])
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--vectorstore-dir", default=DEFAULT_VECTORSTORE_DIR)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--poll-interval", type=int, default=5)
    args = parser.parse_args()

    DEVICE = "cuda" if args.gpu else "cpu" if args.cpu else None

    indexer = LLMDocumentIndexer(
        data_dir=args.data_dir, processed_dir=args.processed_dir,
        model_name=args.embed_model, embedding_backend=args.embed_backend,
        device=DEVICE, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        vectorstore_dir=args.vectorstore_dir
    )

    rag_system = OllamaRAG3(
        llm_model=args.llm_model, embed_model=args.embed_model,
        embed_backend=args.embed_backend, persist_directory=args.vectorstore_dir,
        device=DEVICE
    )

    start_live_repl(indexer, rag_system, poll_interval=args.poll_interval)
