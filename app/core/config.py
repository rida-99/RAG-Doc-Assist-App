
# app/core/config.py

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# APPLICATION SETTINGS
# ============================================================

class Settings(BaseSettings):
    """
    Central configuration for the RAG Document Assistant.

    Configuration priority:
        1. Environment variables
        2. .env file
        3. Default values
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # --------------------------------------------------------
    # 1. Application
    # --------------------------------------------------------

    app_name: str = "RAG Document Assistant"
    app_version: str = "1.0.0"

    environment: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    debug: bool = True

    # --------------------------------------------------------
    # 2. Storage
    # --------------------------------------------------------

    upload_dir: Path = BASE_DIR / "storage" / "uploads"

    processed_dir: Path = BASE_DIR / "storage" / "processed"

    vector_store_dir: Path = BASE_DIR / "storage" / "vector_store"

    collection_name: str = "Assit_documents"

    # --------------------------------------------------------
    # 3. Document ingestion
    # --------------------------------------------------------

    max_file_size_mb: int = Field(default=50, gt=0)

    allowed_file_extensions: set[str] = {
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".txt",
        ".md",
    }

    # --------------------------------------------------------
    # 4. Embedding
    # --------------------------------------------------------

    embed_model: str = (
        "intfloat/multilingual-e5-large-instruct"
    )

    embed_batch_size: int = Field(default=64, gt=0)

    # --------------------------------------------------------
    # 5. Chunking
    # --------------------------------------------------------

    chunk_method: Literal[
        "recursive",
        "fixed",
    ] = "recursive"

    chunk_size: int = Field(default=300, gt=0)

    chunk_overlap: int = Field(default=50, ge=0)

    # --------------------------------------------------------
    # 6. Retrieval
    # --------------------------------------------------------

    # Legacy parameter: preserved for compatibility.
    top_k: int = Field(default=5, gt=0)

    # Number of final chunks returned by the Retriever.
    retrieval_top_k: int = Field(default=5, gt=0)

    # Number of initial candidates retrieved from ChromaDB.
    fetch_k: int = Field(default=15, gt=0)

    # Maximum accepted cosine distance.
    # Lower distance means greater similarity.
    max_distance: float = Field(default=0.35, ge=0, le=2)

    # Legacy threshold: preserved until its usage is reviewed.
    min_score: float = Field(default=0.8, ge=0, le=1)

    # --------------------------------------------------------
    # 7. Reranking
    # --------------------------------------------------------

    rerank_model: str = (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )

    # --------------------------------------------------------
    # 8. LLM Provider
    # --------------------------------------------------------

    default_llm: Literal[
        "ollama",
        "openai",
        "claude",
        "anthropic",
    ] = "ollama"

    # --------------------------------------------------------
    # 8.1 Ollama
    # --------------------------------------------------------

    ollama_url: str = (
        "http://localhost:11434/api/generate"
    )

    ollama_model: str = "llama3.2:3b"

    # --------------------------------------------------------
    # 8.2 OpenAI
    # --------------------------------------------------------

    openai_api_key: str | None = None

    openai_model: str = "gpt-4o-mini"

    # --------------------------------------------------------
    # 8.3 Anthropic / Claude
    # --------------------------------------------------------

    anthropic_api_key: str | None = None

    anthropic_model: str = "claude-3-5-sonnet-latest"

    # --------------------------------------------------------
    # 9. LLM Generation
    # --------------------------------------------------------

    temperature: float = Field(default=0.2, ge=0, le=2)

    max_tokens: int = Field(default=1024, gt=0)

    # --------------------------------------------------------
    # 10. Security
    # --------------------------------------------------------

    # Legacy API key: preserved for existing modules.
    api_key: str = ""

    # --------------------------------------------------------
    # 11. API
    # --------------------------------------------------------

    api_host: str = "127.0.0.1"

    api_port: int = Field(default=8000, gt=0, le=65535)

    # --------------------------------------------------------
    # 12. Logging
    # --------------------------------------------------------

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    # ========================================================
    # VALIDATION
    # ========================================================

    @model_validator(mode="after")
    def validate_configuration(self):
        """Validate relationships between configuration values."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if self.fetch_k < self.retrieval_top_k:
            raise ValueError(
                "fetch_k must be greater than or equal to "
                "retrieval_top_k."
            )

        if not self.collection_name.strip():
            raise ValueError(
                "collection_name cannot be empty."
            )

        return self

    # ========================================================
    # COMPUTED PROPERTIES
    # ========================================================

    @property
    def max_file_size_bytes(self) -> int:
        """Convert maximum upload size from MB to bytes."""

        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


# ============================================================
# SETTINGS FACTORY
# ============================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    This avoids reloading the .env file every time
    configuration is requested.
    """

    return Settings()


# ============================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================

settings = get_settings()


# ============================================================
# CONFIGURATION TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("RAG DOCUMENT ASSISTANT - CONFIGURATION")
    print("=" * 55)

    print(f"Application : {settings.app_name}")
    print(f"Version     : {settings.app_version}")
    print(f"Environment : {settings.environment}")

    print("\n--- Storage ---")
    print(f"Uploads     : {settings.upload_dir}")
    print(f"Processed   : {settings.processed_dir}")
    print(f"Vector DB   : {settings.vector_store_dir}")
    print(f"Collection  : {settings.collection_name}")

    print("\n--- Embedding ---")
    print(f"Model       : {settings.embed_model}")
    print(f"Batch size  : {settings.embed_batch_size}")

    print("\n--- Chunking ---")
    print(f"Method      : {settings.chunk_method}")
    print(f"Chunk size  : {settings.chunk_size}")
    print(f"Overlap     : {settings.chunk_overlap}")

    print("\n--- Retrieval ---")
    print(f"Top K       : {settings.retrieval_top_k}")
    print(f"Fetch K     : {settings.fetch_k}")
    print(f"Max distance: {settings.max_distance}")

    print("\n--- Reranking ---")
    print(f"Model       : {settings.rerank_model}")

    print("\n--- LLM ---")
    print(f"Provider    : {settings.default_llm}")
    print(f"Temperature : {settings.temperature}")
    print(f"Max tokens  : {settings.max_tokens}")

    print("\n--- Security ---")
    print(f"Max upload  : {settings.max_file_size_mb} MB")
    print(f"Extensions  : {sorted(settings.allowed_file_extensions)}")

    print("=" * 55)
    print("Configuration loaded successfully.")
    print("=" * 55)
