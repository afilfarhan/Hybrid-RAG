"""
Hybrid RAG - Local in-house embedding service using sentence-transformers
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding.base import BaseEmbeddingService


class InHouseEmbeddingService(BaseEmbeddingService):
    """Local embedding service using sentence-transformers models.
    
    Supports bilingual (Arabic/English) embedding models:
    - multilingual-MiniLM-L12-v2: General purpose, lightweight
    - BAAI/bge-small-en-v1.5: English-optimized, high quality
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: Multilingual
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        model_name = config.get("model_name", "sentence-transformers/multilingual-MiniLM-L12-v2")
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed(self, text: str) -> List[float]:
        """Embed a single text using local model."""
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batch."""
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        return embeddings.tolist()

    async def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        return self._dimension or 384  # Default for MiniLM
