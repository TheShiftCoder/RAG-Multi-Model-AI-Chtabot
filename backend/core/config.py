"""Application Configuration Module.

Loads settings from environment variables using Pydantic Settings.
Configures database connection, file storage, security limits,
and multimodal model provider parameters.
"""

from typing import List, Optional
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
GENERATED_DIR = STORAGE_DIR / "generated"
VECTOR_INDEX_DIR = STORAGE_DIR / "vector_index"


class Settings(BaseSettings):
    """Application settings with environment variable fallbacks."""
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore"
    )

    PROJECT_NAME: str = "Secure Multimodal AI Companion"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:password@localhost:3306/multimodal_ai"
    )
    SQLITE_FALLBACK_URL: str = f"sqlite:///{BASE_DIR / 'multimodal_ai.db'}"

    # Storage Paths
    BASE_STORAGE_PATH: str = str(STORAGE_DIR)
    UPLOAD_PATH: str = str(UPLOAD_DIR)
    GENERATED_PATH: str = str(GENERATED_DIR)
    VECTOR_INDEX_PATH: str = str(VECTOR_INDEX_DIR)

    # File Upload Limits & Formats
    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB limit
    ALLOWED_DOCUMENT_EXTENSIONS: List[str] = ["pdf", "txt"]
    ALLOWED_IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
    ALLOWED_AUDIO_EXTENSIONS: List[str] = ["wav", "mp3", "m4a"]

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    SIMILARITY_THRESHOLD: float = 0.35  # Cosine similarity threshold for RAG
    MAX_RETRIEVAL_CHUNKS: int = 4

    # Providers Configuration ("mock", "openai", "gemini")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "fallback")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "fallback")
    STT_PROVIDER: str = os.getenv("STT_PROVIDER", "fallback")
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "gtts")  # gtts / fallback / openai
    IMAGE_GEN_PROVIDER: str = os.getenv("IMAGE_GEN_PROVIDER", "fallback")

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Security & CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


# Ensure storage directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
