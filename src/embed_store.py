import google.generativeai as genai
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import GEMINI_API_KEY, EMBED_MODEL, VECTOR_DB_DIR

genai.configure(api_key=GEMINI_API_KEY)


def create_vectorstore(docs):

    docs = [doc for doc in docs if doc.page_content.strip()]

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GEMINI_API_KEY
    )

    # Create empty vectorstore
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    batch_size = 2
    max_retries = 4
    total_embedded = 0

    print(f"[INFO] Total chunks to embed: {len(docs)}")

    for i in range(0, len(docs), batch_size):

        batch = docs[i:i+batch_size]
        batch_success = False

        for attempt in range(1, max_retries + 1):
            try:
                vectorstore.add_documents(batch)
                added = len(batch)
                total_embedded += added
                batch_success = True
                print(f"[INFO] Embedded batch {i//batch_size + 1} (attempt {attempt}, +{added} docs)")
                break
            except Exception as e:
                print(f"[ERROR] Batch {i//batch_size + 1} attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    sleep_time = 2 ** attempt
                    print(f"[INFO] Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

        if not batch_success:
            print(f"[ERROR] Skipping batch {i//batch_size + 1} after {max_retries} attempts")

        # brief pause between batches to avoid rate limits
        time.sleep(1)

    if total_embedded == 0:
        print("[WARN] No chunks were successfully embedded into vectorstore.")

    vectorstore.persist()
    print(f"[INFO] Vector DB persisted with {total_embedded} new embeds")
    return vectorstore, total_embedded

    vectorstore.persist()

    print("[INFO] Vector DB created successfully")

    return vectorstore


def load_vectorstore():

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GEMINI_API_KEY
    )

    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )


def summarize_documents(documents):

    try:

        text = " ".join(
            [doc.page_content for doc in documents[:3]]
        )[:4000]

        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(
            f"Summarize the following text in 5 bullet points:\n{text}"
        )

        return response.text

    except Exception as e:

        return f"⚠️ Could not summarize: {e}"