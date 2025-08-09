import os
from dotenv import load_dotenv

# Load variables from .env if it exists
load_dotenv()

class Config:
    # --- App Config ---
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "replace-this-secret")

    # --- Database Config ---
    POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://user:password@localhost:5432/policy_llm")
    DB_NAME = os.getenv("DB_NAME", "policy_llm")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))

    # --- Embedding Model ---
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # --- LLM / Local Model Config ---
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "google/flan-t5-base")
    USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

    # --- LLM Mode Switch ---
    LLM_MODE = os.getenv("LLM_MODE", "local")  # "local" or "gemini"

    # --- Gemini API Config ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # --- Indexer & FAISS ---
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/embeddings/faiss_index")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", 100))

    # --- Cache Management ---
    SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", 30))
    CACHE_MAX_QUERIES = int(os.getenv("CACHE_MAX_QUERIES", 50))  # LRU cache

    # --- Logging & Monitoring ---
    ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # --- Webhook ---
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

    # --- HackRx Authentication ---
    API_KEY = os.getenv("API_KEY", "")
