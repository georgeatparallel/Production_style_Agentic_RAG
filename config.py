import os
from dotenv import load_dotenv

# 1. Load variables from .env file (primarily for local D: drive development)
load_dotenv()

# ==========================================
# API KEYS & SERVER CONFIG
# ==========================================
# Ensure your Railway Variable matches "HUGGINGFACEHUB_API_TOKEN"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Explicit opt-in; existing deployments continue to use DuckDuckGo.
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").strip().lower()

# Railway assigns a dynamic PORT; default to 8000 for local testing
PORT = int(os.getenv("PORT", 8000))

# ==========================================
# PATHS & DIRECTORIES (Absolute for Docker)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder where the FAISS vector database is stored
DB_PATH = os.path.join(BASE_DIR, "faiss_index")

# Folder for temporary PDF processing
TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")

# Auto-create directories on startup to prevent "Folder Not Found" errors
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# ==========================================
# MODEL SETTINGS
# ==========================================
# The LLM "Brain"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# The Embedding "Translator" (Converts text to math for FAISS)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# RAG PARAMETERS (Fine-Tuning)
# ==========================================
TOP_K = 3            # Number of document chunks to retrieve per question
CHUNK_SIZE = 800      # Character count for each text piece
CHUNK_OVERLAP = 80    # Buffer to maintain context between pieces
