import os
import streamlit as st

# Use secrets if available, else fallback to environment variable
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (AttributeError, KeyError):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Model Config
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"
VECTOR_DB_DIR = "chroma_db"
