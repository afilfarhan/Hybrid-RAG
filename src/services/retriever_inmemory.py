"""
In-memory retriever for testing.
"""

from typing import List, Dict, Any, Optional
from src.services.embedding_inmemory import InMemoryEmbeddingService
from src.services.vector_store_inmemory import InMemoryVectorStore


class InMemoryRetriever:
    """Simple in-memory retriever using dense similarity search."""
    
    def __init__(
        self,
        embedding_service: InMemoryEmbeddingService,
        vector_store: InMemoryVectorStore,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    def retrieve(self, query: str, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self.embedding_service.embed(query)
        results = self.vector_store.search(query_embedding, top_k=self.top_k, filter=filter)
        return results
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to the vector store."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)
        
        from src.services.vector_store_inmemory import VectorChunk
        
        vector_chunks = [
            VectorChunk(
                id=chunk.get("id", f"chunk_{i}"),
                text=chunk["text"],
                embedding=embeddings[i],
                metadata=chunk.get("metadata", {}),
            )
            for i, chunk in enumerate(chunks)
        ]
        
        self.vector_store.add_batch(vector_chunks)
