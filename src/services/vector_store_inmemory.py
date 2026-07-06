"""
In-memory vector store for testing.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class VectorChunk:
    """Simple chunk class without dataclass to avoid circular imports."""
    
    def __init__(self, id: str, text: str, embedding: List[float], metadata: Dict[str, Any]):
        self.id = id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata


class InMemoryVectorStore:
    """Simple in-memory vector store using numpy for similarity search."""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self._chunks: Dict[str, VectorChunk] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
    
    def add(self, chunk) -> None:
        """Add a chunk to the store."""
        self._chunks[chunk.id] = chunk
        self._embeddings_matrix = None
    
    def add_batch(self, chunks) -> None:
        """Add multiple chunks to the store."""
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        self._embeddings_matrix = None
    
    def search(
        self, query_embedding: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if not self._chunks:
            return []
        
        query = np.array(query_embedding)
        
        ids = list(self._chunks.keys())
        embeddings = np.array([self._chunks[id].embedding for id in ids])
        
        similarities = np.dot(embeddings, query) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self._chunks[ids[idx]]
            if filter is None or self._matches_filter(chunk.metadata, filter):
                results.append({
                    "id": chunk.id,
                    "text": chunk.text,
                    "score": float(similarities[idx]),
                    "metadata": chunk.metadata,
                })
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if metadata.get(key) != value:
                return False
        return True
    
    def clear(self) -> None:
        """Clear all chunks."""
        self._chunks.clear()
        self._embeddings_matrix = None
    
    def get_chunk(self, chunk_id: str):
        """Get a specific chunk by ID."""
        return self._chunks.get(chunk_id)
    
    def get_all_chunks(self):
        """Get all chunks."""
        return list(self._chunks.values())
