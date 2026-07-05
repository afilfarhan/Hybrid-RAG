"""
Hybrid RAG - Base retriever interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.ingestion.base import Document


class RetrievedChunk:
    """Represents a retrieved chunk with metadata."""

    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        score: float,
        source: str,
    ):
        self.content = content
        self.metadata = metadata
        self.score = score
        self.source = source

    def __repr__(self) -> str:
        return f"RetrievedChunk(score={self.score:.4f}, source={self.source})"


class BaseRetriever(ABC):
    """Base class for retrievers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        pass

    @abstractmethod
    async def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve using hybrid search (dense + sparse)."""
        pass
