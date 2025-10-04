import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model Config
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"
VECTOR_DB_DIR = "chroma_db"
