"""
In-memory embedding service for testing.
"""

import numpy as np
from typing import List, Optional


class InMemoryEmbeddingService:
    """Simple in-memory embedding service using random vectors."""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self._embeddings: dict[str, List[float]] = {}
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text (deterministic based on content)."""
        if text in self._embeddings:
            return self._embeddings[text]
        
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension).tolist()
        self._embeddings[text] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        return [self.embed(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension
