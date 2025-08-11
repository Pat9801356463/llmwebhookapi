import os

class Config:
    # --- App Config ---
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "replace-this-secret")

    # --- Database Config ---
    # Prefer Railway-provided vars if available
    DB_NAME = os.getenv("PGDATABASE", os.getenv("DB_NAME", "railway"))
    DB_USER = os.getenv("PGUSER", os.getenv("DB_USER", "postgres"))
    DB_PASSWORD = os.getenv("PGPASSWORD", os.getenv("DB_PASSWORD", "zUCCqLepmlTwrQhxDwpwWnAUvHqbWgkt"))
    DB_HOST = os.getenv("PGHOST", os.getenv("DB_HOST", "postgres.railway.internal"))
    DB_PORT = int(os.getenv("PGPORT", os.getenv("DB_PORT", 5432)))

    # Unified Postgres URI
    POSTGRES_URI = os.getenv(
        "DATABASE_URL",  # Railway-provided
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # --- Embedding Model ---
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- LLM / Local Model Config ---
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "google/flan-t5-base")
    USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

    # --- LLM Mode Switch ---
    LLM_MODE = os.getenv("LLM_MODE", "gemini")  

    # --- Gemini API Config ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB5tSjidvsJdNW1kRfAaw5pXoSiKe-iubY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # --- Cohere API Config ---
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    COHERE_MODEL = os.getenv("COHERE_MODEL", "command-xlarge-nightly")

    # --- Indexer & FAISS ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", 100))

    # --- Cache Management ---
    SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", 30))
    CACHE_MAX_QUERIES = int(os.getenv("CACHE_MAX_QUERIES", 50))

    # --- Logging & Monitoring ---
    ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # --- Webhook ---
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

    # --- HackRx Authentication ---
    API_KEY = os.getenv("API_KEY", "ce10025c4a5a05ce5776d7c1c8136640412c48de7a7ddcc1c7d80ce1dfb13cff")

