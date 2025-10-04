
# import sys
# import os

# # Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import streamlit as st
# from src.data_loader import load_documents
# from src.preprocess import split_documents
# from src.embed_store import create_vectorstore, load_vectorstore
# from src.chatbot import create_chatbot
# from src.upload_handler import save_upload, handle_uploaded_file  # 👈 NEW

# DB_PATH = "chroma_db"
# UPLOAD_DIR = "uploads"

# st.set_page_config(page_title="📚 MultiDoc Chatbot", layout="wide")
# st.title("🤖 Multi-Document RAG Chatbot with Gemini")

# # Ensure upload dir exists
# if not os.path.exists(UPLOAD_DIR):
#     os.makedirs(UPLOAD_DIR)

# # Sidebar upload section
# st.sidebar.header("📤 Upload Documents")
# uploaded_files = st.sidebar.file_uploader(
#     "Upload PDFs or TXTs", type=["pdf", "txt"], accept_multiple_files=True
# )

# if uploaded_files:
#     for f in uploaded_files:
#         file_path = save_upload(f, f.name)   # save to uploads/
#         handle_uploaded_file(file_path)      # process + add to vectorstore
#         st.sidebar.success(f"✅ {f.name} added to knowledge base")

# # Load or create vectorstore
# if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
#     with st.spinner("📥 Loading & processing base documents..."):
#         docs = load_documents()
#         if not docs:
#             st.error("❌ No documents found in data/pdfs or data/txts folder!")
#             st.stop()
#         chunks = split_documents(docs)
#         vectorstore = create_vectorstore(chunks)
# else:
#     vectorstore = load_vectorstore()

# qa_chain = create_chatbot(vectorstore)

# # Initialize chat messages
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display existing chat messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Chat input
# if prompt := st.chat_input("Ask something from your documents..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         try:
#             response = qa_chain.invoke(prompt)
#             if isinstance(response, dict) and "answer" in response:
#                 response_text = response["answer"]
#             else:
#                 response_text = str(response)
#         except Exception as e:
#             response_text = f"⚠️ Error: {e}"
        
#         st.markdown(response_text)

#     st.session_state.messages.append({"role": "assistant", "content": response_text})








# src/app.py
import sys
import os
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.data_loader import load_documents
from src.preprocess import split_documents
from src.embed_store import create_vectorstore, load_vectorstore
from src.chatbot import create_chatbot
from src.upload_handler import save_upload, handle_uploaded_file
from src.config import VECTOR_DB_DIR

# Streamlit page setup
st.set_page_config(page_title="📚 MultiDoc Chatbot", layout="wide")
st.title("🤖 Multi-Document Chatbot with Gemini")

# Ensure directories exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Function to process uploaded files and update vector store
def process_uploaded_files(uploaded_files):
    new_docs = []
    for f in uploaded_files:
        file_path = save_upload(f, f.name)
        handle_uploaded_file(file_path)
        st.success(f"✅ {f.name} added to knowledge base")
        # Load documents from newly uploaded file
        new_docs.extend(load_documents())
    if new_docs:
        chunks = split_documents(new_docs)
        chunks = [doc for doc in chunks if doc.page_content.strip()]
        if chunks:
            st.session_state.vectorstore = create_vectorstore(chunks)
            st.success("✅ Vector store updated with new documents")
        else:
            st.warning("⚠️ Uploaded documents were empty after cleaning")

# Chat container
chat_container = st.container()

# File upload inside chat
with chat_container:
    uploaded_files = st.file_uploader(
        "📤 Upload PDFs/TXTs directly in chat",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        process_uploaded_files(uploaded_files)

# Load or create vectorstore if not exists
if not st.session_state.vectorstore:
    try:
        if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
            with st.spinner("📥 Loading base documents..."):
                docs = load_documents()
                chunks = split_documents(docs)
                chunks = [doc for doc in chunks if doc.page_content.strip()]
                if chunks:
                    st.session_state.vectorstore = create_vectorstore(chunks)
                else:
                    st.warning("❌ No valid documents found to embed")
        else:
            st.session_state.vectorstore = load_vectorstore()
    except Exception as e:
        st.error(f"❌ Vector store creation failed: {e}")

# Initialize chatbot
qa_chain = None
if st.session_state.vectorstore:
    qa_chain = create_chatbot(st.session_state.vectorstore)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if qa_chain:
    if prompt := st.chat_input("💬 Ask something or upload more documents..."):
        # Detect if user wants to upload via message (optional)
        if prompt.lower().startswith("upload"):
            st.info("Use the upload button above to add documents.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    response = qa_chain.invoke(prompt)
                    response_text = response.get("answer") if isinstance(response, dict) else str(response)
                except Exception as e:
                    response_text = f"⚠️ Error: {e}"

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("ℹ️ Chat will be available once documents are loaded or uploaded.")
