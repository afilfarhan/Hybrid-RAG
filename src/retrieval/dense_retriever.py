"""
Hybrid RAG - Dense retriever using vector search
"""

from typing import Any, Dict, List, Optional

from src.embedding.base import BaseEmbeddingService
from src.embedding.vector_store import BaseVectorStore
from src.retrieval.base import BaseRetriever, RetrievedChunk


class DenseRetriever(BaseRetriever):
    """Dense retriever using vector similarity search."""

    def __init__(
        self,
        embedding_service: BaseEmbeddingService,
        vector_store: BaseVectorStore,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = config.get("top_k", 5)

    async def retrieve(
        self, query: str, top_k: Optional[int] = None, filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks using dense vector search."""
        if top_k is None:
            top_k = self.top_k

        # Embed the query
        query_embedding = await self.embedding_service.embed(query)

        # Search in vector store
        results = await self.vector_store.search(
            query_vector=query_embedding, top_k=top_k, filter=filter
        )

        # Format results
        chunks = []
        for result in results:
            chunks.append(
                RetrievedChunk(
                    content=result["metadata"].get("content", ""),
                    metadata=result["metadata"],
                    score=1.0 - result["score"],  # Convert distance to similarity
                    source=result["metadata"].get("source", "unknown"),
                )
            )

        return chunks

    async def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve using dense search only (no sparse yet)."""
        return await self.retrieve(query, top_k, filter)
