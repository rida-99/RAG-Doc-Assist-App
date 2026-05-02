# app/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Storage
    upload_dir: Path = Path("storage/uploads")
    processed_dir: Path = Path("storage/processed")
    vector_store_path: Path = Path("storage/vector_store")
    collection_name: str = "Assit_documents"

    # Embedding
    embed_model: str = "intfloat/multilingual-e5-large-instruct"
    embed_batch_size: int = 64

    # Chunking
    chunk_method: str = "recursive"
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5
    min_score: float = 0.8

    # LLM
    default_llm: str = "claude"  # ollama | openai | claude
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2:3b"

    # Security
    max_file_size_mb: int = 50
    allowed_file_extensions: set[str] = {".pdf", ".docx", ".pptx", ".html", ".txt"}
    api_key: str = ""               # set in .env

    class Config:
        env_file = ".env"

settings = Settings()

if __name__ == "__main__":
    print(f"Configuration loaded: {settings}")