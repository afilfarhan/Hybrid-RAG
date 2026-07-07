"""
ChromaDB Vector Store Implementation

Production-ready vector store using ChromaDB with persistent storage.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from .base import VectorStore


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(
        self,
        persist_path: str = "./data/vector_store",
        collection_name: str = "documents",
        embedding_function: Optional[Any] = None,
        dimension: int = 384
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_path: Path to persist the database
            collection_name: Name of the collection
            embedding_function: Custom embedding function (optional)
            dimension: Dimension of embeddings (default: 384 for MiniLM)
        """
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Create persist directory
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Use default embedding function if not provided
        if embedding_function is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = "all-MiniLM-L6-v2"
                self.embedding_model = SentenceTransformer(model_name)
                
                def custom_embed(texts: List[str]) -> List[List[float]]:
                    return self.embedding_model.encode(texts).tolist()
                
                # Use SentenceTransformerEmbeddingFunction for ChromaDB
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        else:
            self.embedding_function = embedding_function
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        
        self._ensure_dimension()
    
    def _ensure_dimension(self):
        """Ensure collection has correct dimension."""
        # ChromaDB will handle this automatically with embedding_function
        pass
    
    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a single document to the store."""
        import uuid
        
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata]
        )
        
        return doc_id
    
    def add_batch(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to the store."""
        import uuid
        
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        
        self.collection.add(
            ids=doc_ids,
            documents=texts,
            metadatas=metadatas
        )
        
        return doc_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1.0 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
    
    def clear(self) -> bool:
        """Clear all documents from the store."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "dimension": self.dimension
        }
    
    def persist(self):
        """Persist the database to disk (ChromaDB auto-persists)."""
        # ChromaDB automatically persists to disk when using PersistentClient
        # This method is kept for compatibility but does nothing
        pass
    
    def disconnect(self):
        """Close the connection."""
        pass
