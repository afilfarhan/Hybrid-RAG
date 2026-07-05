"""
Hybrid RAG - Base chunker interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.ingestion.base import Document


class Chunk:
    """Represents a chunk of a document."""

    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_index: int = 0,
        chunk_id: Optional[str] = None,
    ):
        self.content = content
        self.metadata = metadata
        self.chunk_index = chunk_index
        self.chunk_id = chunk_id

    def __repr__(self) -> str:
        return f"Chunk(index={self.chunk_index}, content_length={len(self.content)})"


class BaseChunker(ABC):
    """Base class for chunkers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def chunk(self, document: Document) -> List[Chunk]:
        """Chunk a single document."""
        pass

    @abstractmethod
    async def chunk_batch(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        pass
