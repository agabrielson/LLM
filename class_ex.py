#!/usr/bin/env python3

from ollama import Ollama
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class RetrievalQA:
    def __init__(self, model_name="llama3", top_k=3):
        # Initialize embedder, vector store and LLM client
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = Client(Settings(chroma_db_impl="duckdb+parquet"))  # in-memory vector DB
        self.collection = self.client.get_or_create_collection(name="my_docs")
        self.llm = Ollama()
        self.model_name = model_name
        self.top_k = top_k

    def add_documents(self, docs, metadatas=None):
        embeddings = self.embedder.encode(docs).tolist()
        self.collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas or [{}]*len(docs))

    def retrieve(self, query):
        q_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=self.top_k)
        return results['documents'][0]

    def generate_prompt(self, query, contexts):
        context_text = "\n---\n".join(contexts)
        prompt = (
            f"Use the following context to answer the question.\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return prompt

    def query(self, query):
        # Retrieve top-k docs
        retrieved_docs = self.retrieve(query)
        
        # Build prompt
        prompt = self.generate_prompt(query, retrieved_docs)
        
        # Query LLaMA 3 via Ollama
        response = self.llm.chat(model=self.model_name, prompt=prompt)
        return response

# Initialize QA system
qa_system = RetrievalQA(top_k=2)

# Query example
question = "tell me about colt firearms?"
answer = qa_system.query(question)

print("Answer:", answer)
