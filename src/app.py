# import sys
# import os

# # Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import streamlit as st
# from src.data_loader import load_documents
# from src.preprocess import split_documents
# from src.embed_store import create_vectorstore, load_vectorstore
# from src.chatbot import create_chatbot

# DB_PATH = "chroma_db"

# st.set_page_config(page_title="📚 MultiDoc Chatbot", layout="wide")
# st.title("🤖 Multi-Document RAG Chatbot with Gemini")

# # Load or create vectorstore
# if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
#     with st.spinner("📥 Loading & processing documents..."):
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
#             # Extract only the 'answer' field if it's a dict
#             if isinstance(response, dict) and "answer" in response:
#                 response_text = response["answer"]
#             else:
#                 response_text = str(response)
#         except Exception as e:
#             response_text = f"⚠️ Error: {e}"
        
#         st.markdown(response_text)

#     st.session_state.messages.append({"role": "assistant", "content": response_text})







import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.data_loader import load_documents
from src.preprocess import split_documents
from src.embed_store import create_vectorstore, load_vectorstore
from src.chatbot import create_chatbot
from src.upload_handler import save_upload, handle_uploaded_file  # 👈 NEW

DB_PATH = "chroma_db"
UPLOAD_DIR = "uploads"

st.set_page_config(page_title="📚 MultiDoc Chatbot", layout="wide")
st.title("🤖 Multi-Document RAG Chatbot with Gemini")

# Ensure upload dir exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Sidebar upload section
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs or TXTs", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        file_path = save_upload(f, f.name)   # save to uploads/
        handle_uploaded_file(file_path)      # process + add to vectorstore
        st.sidebar.success(f"✅ {f.name} added to knowledge base")

# Load or create vectorstore
if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
    with st.spinner("📥 Loading & processing base documents..."):
        docs = load_documents()
        if not docs:
            st.error("❌ No documents found in data/pdfs or data/txts folder!")
            st.stop()
        chunks = split_documents(docs)
        vectorstore = create_vectorstore(chunks)
else:
    vectorstore = load_vectorstore()

qa_chain = create_chatbot(vectorstore)

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something from your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = qa_chain.invoke(prompt)
            if isinstance(response, dict) and "answer" in response:
                response_text = response["answer"]
            else:
                response_text = str(response)
        except Exception as e:
            response_text = f"⚠️ Error: {e}"
        
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
