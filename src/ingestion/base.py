"""
Hybrid RAG - Base classes for ingestion pipeline
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Document:
    """Represents a document in the system."""

    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, metadata={self.metadata})"


class BaseIngestionPipeline(ABC):
    """Base class for ingestion pipelines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def ingest(self, source: Any) -> List[Document]:
        """Ingest documents from a source."""
        pass

    @abstractmethod
    async def ingest_all(self) -> List[Document]:
        """Ingest all documents from configured sources."""
        pass

    @abstractmethod
    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        pass

    @abstractmethod
    async def update(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Update a document."""
        pass
