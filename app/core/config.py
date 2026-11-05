
# app/core/config.py

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================
    # Application
    # ========================================================

    app_name: str = "RAG Document Assistant"
    app_version: str = "1.0.0"
    environment: Literal[
        "development", "testing", "production"
    ] = "development"

    debug: bool = True

    # ========================================================
    # Storage
    # ========================================================

    upload_dir: Path = BASE_DIR / "storage" / "uploads"
    processed_dir: Path = BASE_DIR / "storage" / "processed"
    vector_store_dir: Path = BASE_DIR / "storage" / "vector_store"

    collection_name: str = "Assit_documents"

    # ========================================================
    # Embedding
    # ========================================================

    embed_model: str = "intfloat/multilingual-e5-large-instruct"
    embed_batch_size: int = Field(default=64, gt=0)

    # ========================================================
    # Chunking
    # ========================================================

    chunk_method: str = "recursive"
    chunk_size: int = Field(default=300, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)

    # ========================================================
    # Retrieval
    # ========================================================

    top_k: int = Field(default=5, gt=0)
    retrieval_top_k: int = Field(default=5, gt=0)

    # Preserve your existing threshold.
    # Its interpretation depends on your Retriever implementation.
    min_score: float = 0.8

    # ========================================================
    # Reranking
    # ========================================================

    rerank_model: str = (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )

    # ========================================================
    # LLM
    # ========================================================

    default_llm: Literal[
        "ollama", "openai", "claude"
    ] = "ollama"

    # Ollama
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2:3b"

    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"

    # ========================================================
    # Generation
    # ========================================================

    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=2048, gt=0)

    # ========================================================
    # Security
    # ========================================================

    max_file_size_mb: int = Field(default=50, gt=0)

    allowed_file_extensions: set[str] = {
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".txt",
        ".md",
    }

    api_key: str = ""

    # ========================================================
    # API
    # ========================================================

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # ========================================================
    # Logging
    # ========================================================

    log_level: str = "INFO"

    # ========================================================
    # Computed properties
    # ========================================================

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


# ============================================================
# Settings singleton
# ============================================================

@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


if __name__ == "__main__":
    print("Configuration loaded successfully")
    print(f"Application: {settings.app_name}")
    print(f"Embedding: {settings.embed_model}")
    print(f"LLM: {settings.default_llm}")
    print(f"Uploads: {settings.upload_dir}")
    print(f"Vector store: {settings.vector_store_dir}")
