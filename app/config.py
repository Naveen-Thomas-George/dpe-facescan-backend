# app/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Event Config
    EVENT_SLUG: str = "christ-sports-2025"

    # --- UPDATED: Azure Storage Config ---
    # --- Azure Storage Config ---
    AZURE_STORAGE_CONNECTION_STRING: str | None = None
    STORAGE_ACCOUNT_NAME: str | None = None
    AZURE_PHOTO_CONTAINER: str = "photos"
    AZURE_INDEX_CONTAINER: str = "indexes"
    
    # Database (use Railway Postgres URL here for local too)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data.db")

    # Search Config
    FAISS_METRIC: str = "cosine"
    MATCH_THRESHOLD: float = 0.45
    TOP_K: int = 50

    # App/Security
    MAX_UPLOAD_MB: int = 8
    SECRET_KEY: str = "change-me"

    # Paths (local cache for embeddings/indexes)
    MEDIA_ROOT: str = os.getenv("MEDIA_ROOT", "/tmp/media")


settings = Settings()

# Create local directories for temporary processing if they don't exist
os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
os.makedirs(os.path.join(settings.MEDIA_ROOT, "indices"), exist_ok=True)
os.makedirs(os.path.join(settings.MEDIA_ROOT, "embeddings"), exist_ok=True)