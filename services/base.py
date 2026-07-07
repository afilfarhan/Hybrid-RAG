"""
Hybrid RAG - Core Services Layer

This module provides the core RAG services with a simple, clean interface.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings."""
        pass
    
    def get_dimension(self) -> int:
        """Get embedding dimension (optional, for initialization)."""
        test_text = "test"
        embedding = self.embed(test_text)
        return len(embedding)


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a document to the store."""
        pass
    
    @abstractmethod
    def add_batch(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to the store."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass


class RetrievalService(ABC):
    """Abstract base class for retrieval services."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        pass


class GenerationService(ABC):
    """Abstract base class for generation services."""
    
    @abstractmethod
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate a response based on query and retrieved contexts."""
        pass


class RAGService:
    """Main RAG service that orchestrates all components."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        retrieval_service: RetrievalService,
        generation_service: GenerationService
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query through the full RAG pipeline."""
        # Retrieve relevant documents
        contexts = self.retrieval_service.retrieve(question, top_k)
        
        # Generate response
        result = self.generation_service.generate(question, contexts)
        
        return {
            "answer": result["text"],
            "citations": contexts,
            "confidence": result.get("confidence", 0.0),
            "metadata": {
                "model": result.get("model", "unknown"),
                "used_contexts": len(contexts)
            }
        }
    
    def ingest_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Ingest a document into the vector store."""
        return self.vector_store.add(text, metadata)
