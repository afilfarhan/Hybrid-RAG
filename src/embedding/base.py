"""Base class for embedding services."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize embedding service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get('model_name', 'text-embedding-3-large')
        self.dimension = config.get('dimension', 3072)
        
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def get_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Dimension of embedding vectors
        """
        pass
