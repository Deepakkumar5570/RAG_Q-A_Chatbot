import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """
    Cleaning logic for research papers & text files
    """
    if not text:
        return ""

    # Remove references section (common in research papers)
    text = re.sub(r"References.*", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove figure/table captions (optional cleanup)
    text = re.sub(r"Figure\s*\d+.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Table\s*\d+.*", "", text, flags=re.IGNORECASE)

    # Remove page numbers like "Page 12" or just "12"
    text = re.sub(r"\bPage\s*\d+\b", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Cleans and splits documents into smaller chunks
    """
    # Clean each document
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(docs)} chunks after cleaning")
    return docs
