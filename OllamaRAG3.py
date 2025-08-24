#!/usr/bin/env python3

"""OllamaRAG3 Python Module with Modern ChatMessageHistory Memory

This module provides a class to initialize and query a Retrieval-Augmented Generation (RAG)
system using Ollama LLM and Chroma vector store, with fully updated session-based memory.
"""

import os
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory

# Default models
LLM_MODEL = "llama3"
EMBED_MODEL = "llama3"
PERSIST_DIR = "./chroma_db"

# Global session store for memory
_SESSION_STORE = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    """Retrieve or create a modern memory object for a session."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return _SESSION_STORE[session_id]


class OllamaRAG3:
    def __init__(self, llm_model=LLM_MODEL, embed_model=EMBED_MODEL, persist_directory=PERSIST_DIR):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.persist_directory = persist_directory

        self.retriever = self._init_retrievers()
        self.llm = OllamaLLM(model=self.llm_model)
        self.chains = {}  # store chains per session

    def _init_retrievers(self):
        """Initialize retrievers from Chroma DB."""
        embeddings = OllamaEmbeddings(model=self.embed_model)

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )

        basic_db_retriever = vectorstore.as_retriever()
        adv_db_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[basic_db_retriever, adv_db_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever

    def _init_chain_for_session(self, session_id: str) -> ConversationalRetrievalChain:
        """Initialize a chain for a specific session with memory."""
        memory = get_memory(session_id)

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True
        )
        self.chains[session_id] = chain
        return chain

    @staticmethod
    def replace_file_extension(filename: str, new_extension: str) -> str:
        """Utility: change file extension of a given filename."""
        base_name, _ = os.path.splitext(filename)
        if not new_extension.startswith('.'):
            new_extension = '.' + new_extension
        return base_name + new_extension

    def ask(self, query: str, session_id: str = "default"):
        """Query the RAG system and return both the answer and source filenames, with memory."""
        if session_id not in self.chains:
            self._init_chain_for_session(session_id)

        chain = self.chains[session_id]
        result = chain.invoke({"question": query})  # use invoke to avoid deprecated __call__

        answer_text = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        # Collect just the source filenames
        source_files = []
        for doc in source_docs:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                source_files.append(os.path.basename(doc.metadata["source"]))

        return {"answer": answer_text, "sources": source_files}


def initialize_rag_system(llm_model=LLM_MODEL, embed_model=EMBED_MODEL, persist_directory=PERSIST_DIR):
    """Initialize and return an OllamaRAG3 instance with modern memory."""
    return OllamaRAG3(llm_model=llm_model, embed_model=embed_model, persist_directory=persist_directory)


def test_rag():
    rag_system = initialize_rag_system()
    print("Testing memory across queries...\n")

    queries = [
        "what is the colt single action army",
        "what calibers have it come in"
        "how do you clean a gun",
        "which bullet caliber is bigger 9mm or 45acp"
    ]

    session_id = "demo"
    for q in queries:
        result = rag_system.ask(q, session_id=session_id)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}\n")


if __name__ == "__main__":
    test_rag()
