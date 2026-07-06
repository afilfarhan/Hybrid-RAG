"""
Hybrid RAG - Core utilities
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "INFO"

    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Vector store settings
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/vector_store"

    # Redis settings
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour

    # Retrieval settings
    retrieval_top_k: int = 5
    retrieval_dense_weight: float = 0.7
    retrieval_sparse_weight: float = 0.3

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 51

    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # Compliance settings
    data_residency: str = "us"  # us, eu, sa (for PDPL)
    enable_audit_logging: bool = True


settings = Settings()
