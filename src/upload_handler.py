import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.preprocess import split_documents
from src.embed_store import load_vectorstore

UPLOAD_DIR = "uploads"

def save_upload(file, filename: str) -> str:
    """Save uploaded file to disk and return path"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file, buffer)

    return file_path

def handle_uploaded_file(file_path: str):
    """Load uploaded PDF/TXT, split, and add to vectorstore"""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("❌ Only .pdf and .txt supported")

    docs = loader.load()
    chunks = split_documents(docs)

    vectorstore = load_vectorstore()
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    print(f"[INFO] ✅ {file_path} added to vectorstore")
    return True
