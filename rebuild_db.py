import os
import shutil
from src.data_loader import load_documents
from src.preprocess import split_documents
from src.embed_store import create_vectorstore
from src.config import VECTOR_DB_DIR

def rebuild_vector_db():
    # 1. Purana DB delete karo
    if os.path.exists(VECTOR_DB_DIR):
        print("[INFO] Removing old vectorstore...")
        shutil.rmtree(VECTOR_DB_DIR)

    # 2. Data load karo
    documents = load_documents()
    print(f"[INFO] Loaded {len(documents)} documents")

    # 3. Chunking
    chunks = split_documents(documents)

    # 4. Naya vectorstore create karo
    create_vectorstore(chunks)
    print("[INFO] ✅ Vector DB rebuilt successfully!")

if __name__ == "__main__":
    rebuild_vector_db()