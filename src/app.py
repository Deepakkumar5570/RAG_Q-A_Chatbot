import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_documents
from src.preprocess import split_documents
from src.embed_store import create_vectorstore, load_vectorstore
from src.chatbot import create_chatbot, create_llm
from src.upload_handler import save_upload, handle_uploaded_file
from src.config import VECTOR_DB_DIR

st.set_page_config(page_title="📚 MultiDoc Chatbot", layout="wide")
st.title("🤖 MultiDoc Chatbot (Gemini style)")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def vectorstore_has_items(vs):
    try:
        docs = vs.similarity_search("test", k=1)
        if docs:
            return True
    except Exception:
        pass
    try:
        if hasattr(vs, "_collection"):
            return vs._collection.count() > 0
    except Exception:
        pass
    return False

def process_uploaded_files(uploaded_files):
    for f in uploaded_files:
        file_path = save_upload(f, f.name)
        ok = handle_uploaded_file(file_path)
        if ok:
            st.success(f"✅ {f.name} added") 
        else:
            st.warning(f"⚠️ {f.name} had no usable text")
    try:
        vs = load_vectorstore()
        if vectorstore_has_items(vs):
            st.session_state.vectorstore = vs
        else:
            st.session_state.vectorstore = None
    except Exception as e:
        st.error(f"Error reloading vectorstore: {e}")

with st.sidebar.expander("📤 Upload docs", expanded=True):
    uploaded_files = st.file_uploader(
        "PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if uploaded_files:
        process_uploaded_files(uploaded_files)

if not st.session_state.vectorstore:
    try:
        if not os.listdir(VECTOR_DB_DIR):
            with st.spinner("Loading base docs..."):
                docs = load_documents()
                chunks = split_documents(docs)
                chunks = [d for d in chunks if d.page_content.strip()]
                if chunks:
                    vs, embedded = create_vectorstore(chunks)
                    if embedded > 0:
                        st.session_state.vectorstore = vs
                    else:
                        st.warning("No chunks were embedded.")
                else:
                    st.info("No docs found. Upload one.")
        else:
            vs = load_vectorstore()
            if vectorstore_has_items(vs):
                st.session_state.vectorstore = vs
            else:
                st.session_state.vectorstore = None
    except Exception as e:
        st.error(f"Vector store error: {e}")

qa_chain = create_chatbot(st.session_state.vectorstore) if st.session_state.vectorstore else None
llm_agent = create_llm() if not qa_chain else None

if st.button("Clear chat"):
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.form(key="chat_form", clear_on_submit=False):
    query_text = st.text_input("💬 Ask something:", key="query_text")
    submit = st.form_submit_button("Send")

if submit and query_text:
    st.session_state.messages.append({"role": "user", "content": query_text})

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            response_text = ""
            try:
                if qa_chain:
                    res = qa_chain.invoke(query_text)
                    if isinstance(res, dict):
                        response_text = res.get("answer", "")
                        if not response_text and llm_agent:
                            response_text = llm_agent(query_text)
                    else:
                        response_text = str(res)
                elif llm_agent:
                    response_text = llm_agent(query_text)
                else:
                    response_text = "⚠️ Chat backend not available."
            except Exception as e:
                if llm_agent:
                    response_text = llm_agent(query_text)
                else:
                    response_text = f"Error: {e}"

            if not response_text:
                response_text = "⚠️ No answer could be generated. Please try again."

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})