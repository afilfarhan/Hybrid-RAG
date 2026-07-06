"""
In-Memory Services for Development/Testing

These services provide fallback functionality when real services are unavailable.
"""

import numpy as np
from typing import List, Dict, Any
from .base import EmbeddingService, VectorStore, RetrievalService, GenerationService


class InMemoryEmbeddingService(EmbeddingService):
    """Simple in-memory embedding service using a pre-trained model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return [e.tolist() for e in embeddings]


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using numpy for similarity search."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
    
    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a document to the store."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode([text])[0].tolist()
        
        doc_id = f"doc_{len(self.documents)}"
        self.documents.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata
        })
        self.embeddings.append(embedding)
        
        return doc_id
    
    def add_batch(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to the store."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)
        
        doc_ids = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            doc_id = f"doc_{len(self.documents)}"
            self.documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata
            })
            self.embeddings.append(embedding.tolist())
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode([query])[0]
        
        if not self.embeddings:
            return []
        
        # Calculate cosine similarity
        embeddings_array = np.array(self.embeddings)
        query_array = np.array(query_embedding)
        
        similarities = np.dot(embeddings_array, query_array) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_array) + 1e-8
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                **self.documents[idx],
                "similarity": float(similarities[idx])
            })
        
        return results


class InMemoryRetrievalService(RetrievalService):
    """Simple retrieval service using in-memory vector store."""
    
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        k = top_k or self.top_k
        return self.vector_store.search(query, k)


class InMemoryGenerationService(GenerationService):
    """Simple in-memory generation service using a template-based approach."""
    
    def __init__(self, model_name: str = "in-memory"):
        self.model_name = model_name
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate a response based on query and retrieved contexts."""
        if not contexts:
            return {
                "text": "I don't have enough information to answer this question.",
                "confidence": 0.0,
                "model": self.model_name
            }
        
        # Extract context texts
        context_texts = [c["text"] for c in contexts]
        combined_context = "\n\n".join(context_texts)
        
        # Simple template-based response
        response = f"Based on the available information:\n\n{combined_context}\n\n"
        response += f"Answer: This information is derived from {len(contexts)} relevant documents."
        
        # Calculate confidence based on similarity scores
        avg_similarity = np.mean([c.get("similarity", 0) for c in contexts]) if contexts else 0
        confidence = min(avg_similarity, 0.95)
        
        return {
            "text": response,
            "confidence": float(confidence),
            "model": self.model_name
        }
