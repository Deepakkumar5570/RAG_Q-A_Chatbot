# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import Chroma
# from src.config import GEMINI_API_KEY, EMBED_MODEL, VECTOR_DB_DIR

# genai.configure(api_key=GEMINI_API_KEY)

# def create_vectorstore(docs):
#     embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
#     vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_DIR)
#     vectorstore.persist()
#     print("[INFO] Vector DB created and persisted")
#     return vectorstore

# def load_vectorstore():
#     embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
#     return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)




import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import GEMINI_API_KEY, EMBED_MODEL, VECTOR_DB_DIR

genai.configure(api_key=GEMINI_API_KEY)

def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GEMINI_API_KEY   # ✅ API key force
    )
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_DIR)
    vectorstore.persist()
    print("[INFO] Vector DB created and persisted")
    return vectorstore

def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GEMINI_API_KEY   # ✅ API key force
    )
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
