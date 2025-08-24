#!/usr/bin/env python3

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

#model_name = "nomic-embed-text"
model_name = "llama3"
#model_name = "gemma3:4b"

# Define the directory containing your documents
data_dir = "articlesV2/"

# Load documents from the directory
loader = DirectoryLoader(
    data_dir,
    glob="**/*",  # "**/*" Loads all files
    loader_cls=PyPDFLoader if data_dir.endswith(".txt") else TextLoader, # Adjust loader based on file type
    silent_errors=True  #False
)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings using Ollama
embeddings = OllamaEmbeddings(model=model_name) # e.g., "llama3"

# Create a vector store (e.g., ChromaDB) from the embeddings
print(f"The text list has {len(texts)} elements.")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")


# New documents to add
#    new_documents = [
#        Document(page_content="This is a new document added later."),
#        Document(page_content="Another new document for the collection."),
#    ]

    # Add the new documents
#    vectorstore.add_documents(documents=new_documents)