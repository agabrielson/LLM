#!/usr/bin/env python3

"""OllamaRAG Python Module

This module provides a class to initialize and query a Retrieval-Augmented Generation (RAG)
system using Ollama LLM (Gemma3:4b by default) and Chroma vector store.
"""

import os
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever

# Default models -> need to update model for this...
#LLM_MODEL = "gemma3:4b"          # for text generation
#EMBED_MODEL = "nomic-embed-text" # for embeddings
LLM_MODEL = "llama3"
EMBED_MODEL = "llama3"
PERSIST_DIR = "./chroma_db"      # Path to vector DB (built separately)


class OllamaRAG2:
    def __init__(self, llm_model=LLM_MODEL, embed_model=EMBED_MODEL, persist_directory=PERSIST_DIR):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.persist_directory = persist_directory
        self.qa = self._init_existing_llm_model()

    def _init_retrievers(self):
        """Initialize retrievers from Chroma DB."""
        embeddings = OllamaEmbeddings(model=self.embed_model)

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )

        basic_db_retriever = vectorstore.as_retriever()
        adv_db_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Ensemble retriever (weights can be tuned)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[basic_db_retriever, adv_db_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever

    def _init_existing_llm_model(self):
        """Initialize Gemma3:4b with retrievers."""
        retriever = self._init_retrievers()
        llm = OllamaLLM(model=self.llm_model)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",   # simple: stuff retrieved docs into prompt
            return_source_documents=True
        )
        return qa_chain

    @staticmethod
    def replace_file_extension(filename: str, new_extension: str) -> str:
        """Utility: change file extension of a given filename."""
        base_name, _ = os.path.splitext(filename)
        if not new_extension.startswith('.'):
            new_extension = '.' + new_extension
        return base_name + new_extension

    def ask(self, query: str):
        """Query the RAG system and return both the answer and source filenames."""
        result = self.qa.invoke(query)
        #print(type(result))
        answer_text = result.get('result', "")
        source_docs = result.get('source_documents', [])

        # Collect just the source filenames
        source_files = []
        for doc in source_docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_files.append(os.path.basename(doc.metadata['source']))

        return {"answer": answer_text, "sources": source_files}


def initialize_rag_system(llm_model=LLM_MODEL, embed_model=EMBED_MODEL, persist_directory=PERSIST_DIR):
    """Initialize and return an OllamaRAG2 instance."""
    return OllamaRAG2(llm_model=llm_model, embed_model=embed_model, persist_directory=persist_directory)


# Optional test function (can be removed if used purely as a module)
def test_rag():
    rag_system = initialize_rag_system()
    queries = [
        "what is the colt single action army",
        "what calibers have it come in"
        "how do you clean a gun",
        "which bullet caliber is bigger 9mm or 45acp"
    ]

    for query in queries:
        print(f"Query: {query}")
        result = rag_system.ask(query)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}\n")


# Only run tests when executed directly
if __name__ == "__main__":
    test_rag()
