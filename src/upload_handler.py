import os
import shutil
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.preprocess import split_documents
from src.embed_store import load_vectorstore

UPLOAD_DIR = "uploads"


def save_upload(file, filename: str) -> str:
    """Save uploaded file"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file, buffer)

    return file_path


def handle_uploaded_file(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        print(f"[INFO] Image uploaded: {file_path}")
        return True

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(" Only .pdf, .txt, .jpg, .png supported")

    docs = loader.load()
    if not docs:
        print(f"[WARNING] No docs in uploaded file: {file_path}")
        return False

    chunks = split_documents(docs)
    if not chunks:
        print(f"[WARNING] No content after splitting uploaded file: {file_path}")
        return False

    print(f"[INFO] Total chunks: {len(chunks)}")

    vectorstore = load_vectorstore()

    #  Batch embedding to avoid API timeout
    batch_size = 2
    total_embedded = 0
    max_retries = 4

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_ok = False
        for attempt in range(1, max_retries + 1):
            try:
                vectorstore.add_documents(batch)
                total_embedded += len(batch)
                batch_ok = True
                print(f"[INFO] Embedded batch {i//batch_size + 1} (+{len(batch)} docs) attempt {attempt}")
                break
            except Exception as e:
                print(f"[ERROR] Batch {i//batch_size + 1} attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[INFO] Retrying in {wait}s...")
                    time.sleep(wait)

        if not batch_ok:
            print(f"[ERROR] Skipping batch {i//batch_size + 1} after {max_retries} retries")

        time.sleep(1)

    vectorstore.persist()

    if total_embedded > 0:
        print(f"[INFO] {file_path} added to vectorstore ({total_embedded} chunks)")
        return True

    print(f"[WARN] {file_path} produced no valid embeddings")
    return False
