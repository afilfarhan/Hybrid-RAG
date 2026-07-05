"""Configuration management for Hybrid RAG system."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(default='', description='OpenAI API key')
    openai_model: str = Field(default='gpt-4o', description='OpenAI model to use')
    openai_embedding_model: str = Field(default='text-embedding-3-large', description='OpenAI embedding model')
    
    # Vector Store Configuration
    vector_store_type: str = Field(default='chroma', description='Vector store type (chroma, pinecone, etc.)')
    chroma_persist_dir: str = Field(default='./data/chroma', description='ChromaDB persist directory')
    chroma_collection_name: str = Field(default='hybrid_rag_docs', description='ChromaDB collection name')
    
    # Embedding Settings
    embedding_dimension: int = Field(default=3072, description='Embedding vector dimension')
    chunk_size: int = Field(default=512, description='Chunk size in tokens')
    chunk_overlap: int = Field(default=50, description='Chunk overlap in tokens')
    
    # Retrieval Settings
    retrieval_top_k: int = Field(default=5, description='Number of chunks to retrieve')
    retrieval_score_threshold: float = Field(default=0.7, description='Minimum similarity score threshold')
    hybrid_retrieval_weight: float = Field(default=0.7, description='Weight for hybrid retrieval (0.0-1.0)')
    
    # Generation Settings
    temperature: float = Field(default=0.2, description='LLM temperature')
    max_tokens: int = Field(default=1000, description='Maximum tokens in response')
    system_prompt: str = Field(
        default="You are a helpful assistant that answers questions based solely on the provided context. Always cite your sources and say 'I don't have information on that' if the answer isn't in the context.",
        description='System prompt for the LLM'
    )
    
    # Caching
    cache_enabled: bool = Field(default=True, description='Enable caching')
    cache_ttl: int = Field(default=3600, description='Cache TTL in seconds')
    redis_url: str = Field(default='redis://localhost:6379', description='Redis connection URL')
    
    # Logging
    log_level: str = Field(default='INFO', description='Logging level')
    log_file: str = Field(default='./logs/app.log', description='Log file path')
    
    # Security
    security_enabled: bool = Field(default=True, description='Enable security features')
    pii_redaction: bool = Field(default=True, description='Enable PII redaction')
    access_control_enabled: bool = Field(default=True, description='Enable access control')
    
    # Evaluation
    evaluation_enabled: bool = Field(default=True, description='Enable evaluation tracking')
    evaluation_metrics: List[str] = Field(
        default_factory=lambda: ['faithfulness', 'relevance', 'hallucination_rate'],
        description='Evaluation metrics to track'
    )
    
    # Data sources
    data_sources_dir: str = Field(default='./data/sources', description='Directory for data sources')
    sample_docs_dir: str = Field(default='./data/sample_docs', description='Directory for sample documents')
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_api_key(cls, v):
        """Validate API key is set."""
        if not v:
            raise ValueError('OPENAI_API_KEY must be set')
        return v
    
    @field_validator('vector_store_type')
    @classmethod
    def validate_vector_store(cls, v):
        """Validate vector store type."""
        valid_types = ['chroma', 'pinecone', 'qdrant', 'weaviate']
        if v not in valid_types:
            raise ValueError(f'vector_store_type must be one of {valid_types}')
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        return v.upper() if v.upper() in valid_levels else 'INFO'


settings = Settings()
