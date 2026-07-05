"""
Hybrid RAG - Vector store interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseVectorStore(ABC):
    """Base class for vector stores."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the vector store."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the vector store."""
        pass

    @abstractmethod
    async def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        """Add vectors to the store."""
        pass

    @abstractmethod
    async def search(
        self, query_vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, doc_id: str) -> bool:
        """Delete vectors by document ID."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all vectors from the store."""
        pass
