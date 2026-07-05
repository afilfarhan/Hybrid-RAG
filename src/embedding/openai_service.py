"""OpenAI embedding service implementation."""

from typing import List, Dict, Any, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService:
    """OpenAI embedding service implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI embedding service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model_name', 'text-embedding-3-large')
        self.dimension = config.get('dimension', 3072)
        self.client = None
        
    async def _get_client(self):
        """Get OpenAI client.
        
        Returns:
            OpenAI client instance
        """
        if self.client is None:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        return self.client
    
    async def embed(self, text: str) -> List[float]:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        client = await self._get_client()
        
        try:
            response = await client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        client = await self._get_client()
        
        try:
            response = await client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    async def get_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Dimension of embedding vectors
        """
        return self.dimension
