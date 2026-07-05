"""Custom exceptions for Hybrid RAG system."""


class RAGError(Exception):
    """Base exception for RAG system errors."""
    pass


class ConfigurationError(RAGError):
    """Configuration-related errors."""
    pass


class RetrievalError(RAGError):
    """Retrieval-related errors."""
    pass


class GenerationError(RAGError):
    """Generation-related errors."""
    pass


class IngestionError(RAGError):
    """Ingestion-related errors."""
    pass


class EmbeddingError(RAGError):
    """Embedding-related errors."""
    pass


class CacheError(RAGError):
    """Cache-related errors."""
    pass


class SecurityError(RAGError):
    """Security-related errors."""
    pass


class EvaluationError(RAGError):
    """Evaluation-related errors."""
    pass
