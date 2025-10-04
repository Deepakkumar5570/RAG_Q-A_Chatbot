
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_documents():
    documents = []

    # Load PDFs
    pdf_dir = "data/pdfs"
    if os.path.exists(pdf_dir):
        for file in os.listdir(pdf_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_dir, file))
                documents.extend(loader.load())

    # Load TXTs
    txt_dir = "data/txts"
    if os.path.exists(txt_dir):
        for file in os.listdir(txt_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(txt_dir, file)
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"[WARNING] Skipping {file_path} due to error: {e}")

    if not documents:
        raise RuntimeError("❌ No documents loaded. Please check your data/pdfs or data/txts folder.")

    print(f"[INFO] Loaded {len(documents)} documents")
    return documents
