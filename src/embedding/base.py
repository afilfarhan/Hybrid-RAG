"""
Hybrid RAG - Embedding service interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseEmbeddingService(ABC):
    """Base class for embedding services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.dimension: Optional[int] = None

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass

    @abstractmethod
    async def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass
