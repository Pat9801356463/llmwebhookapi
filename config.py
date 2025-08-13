# config.py
import os

class Config:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "replace-this-secret")

    DB_NAME = os.getenv("PGDATABASE", os.getenv("DB_NAME", "railway"))
    DB_USER = os.getenv("PGUSER", os.getenv("DB_USER", "postgres"))
    DB_PASSWORD = os.getenv("PGPASSWORD", os.getenv("DB_PASSWORD", "postgres"))
    DB_HOST = os.getenv("PGHOST", os.getenv("DB_HOST", "postgres.railway.internal"))
    DB_PORT = int(os.getenv("PGPORT", os.getenv("DB_PORT", 5432)))

    POSTGRES_URI = os.getenv(
        "DATABASE_URL",
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "google/flan-t5-base")
    USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

    LLM_MODE = os.getenv("LLM_MODE", "gemini")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    COHERE_MODEL = os.getenv("COHERE_MODEL", "command-xlarge-nightly")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", 100))

    SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", 30))
    CACHE_MAX_QUERIES = int(os.getenv("CACHE_MAX_QUERIES", 50))

    ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

    API_KEY = os.getenv("API_KEY", "")

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp_starter")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "policy-llm-index")
