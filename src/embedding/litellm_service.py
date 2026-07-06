"""
Hybrid RAG - LiteLLM embedding service
"""

from typing import Any, Dict, List, Optional

import litellm

from src.core.provider_config import get_provider_model, DEFAULT_MODELS
from src.embedding.base import BaseEmbeddingService


class LiteLMEmbeddingService(BaseEmbeddingService):
    """Embedding service using LiteLLM with support for multiple providers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Handle provider-based model naming
        model_name = config.get("model_name")
        provider = config.get("provider")
        
        if provider and not model_name:
            model_name = DEFAULT_MODELS.get(provider, {}).get("embedding")
        
        if provider and model_name:
            model_name = get_provider_model(provider, "embedding", model_name)
        
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", None)

        self.model_name = model_name or "text-embedding-3-small"
        self.dimension: Optional[int] = None

        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url

    async def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        response = litellm.embedding(model=self.model_name, input=text)
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        response = litellm.embedding(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    async def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.dimension is None:
            test_text = "test"
            embedding = await self.embed(test_text)
            self.dimension = len(embedding)
        return self.dimension
