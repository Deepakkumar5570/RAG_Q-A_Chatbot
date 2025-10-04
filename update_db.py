from src.data_loader import load_documents
from src.preprocess import split_documents
from src.embed_store import load_vectorstore
import os

def update_vectorstore():
    # Load new docs
    docs = load_documents()
    chunks = split_documents(docs)

    # Load existing vectorstore
    vectorstore = load_vectorstore()

    # Add new docs
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    print("[INFO] ✅ Vector DB updated with new documents")

if __name__ == "__main__":
    update_vectorstore()
