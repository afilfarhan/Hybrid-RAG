"""
Hybrid RAG - OpenAI embedding service
"""

from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI

from src.embedding.base import BaseEmbeddingService


class OpenAIEmbeddingService(BaseEmbeddingService):
    """Embedding service using OpenAI models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        api_key = config.get("api_key", "")
        model_name = config.get("model_name", "text-embedding-3-small")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self.dimension: Optional[int] = None

    async def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        response = await self.client.embeddings.create(
            model=self.model_name, input=text
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        response = await self.client.embeddings.create(
            model=self.model_name, input=texts
        )
        return [item.embedding for item in response.data]

    async def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.dimension is None:
            # Test embedding to get dimension
            test_text = "test"
            embedding = await self.embed(test_text)
            self.dimension = len(embedding)
        return self.dimension
