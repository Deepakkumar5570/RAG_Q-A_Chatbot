import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """
    Cleaning logic for research papers & text files
    """

    if not text:
        return ""

    # Remove references section
    text = re.sub(r"References.*", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove figure/table captions
    text = re.sub(r"Figure\s*\d+.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Table\s*\d+.*", "", text, flags=re.IGNORECASE)

    # Remove page numbers
    text = re.sub(r"\bPage\s*\d+\b", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_documents(documents, chunk_size=300, chunk_overlap=50):
    """
    Clean and split documents into smaller chunks
    """

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    docs = splitter.split_documents(documents)

    print(f"[INFO] Split into {len(docs)} chunks after cleaning")

    return docs