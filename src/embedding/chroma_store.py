"""
Hybrid RAG - ChromaDB vector store implementation
"""

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.embedding.vector_store import BaseVectorStore


class ChromaDBVectorStore(BaseVectorStore):
    """Vector store using ChromaDB."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.collection_name = config.get("collection_name", "hybrid_rag")
        self.persist_path = config.get("persist_path", "./data/vector_store")

        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None

    async def connect(self) -> bool:
        """Connect to ChromaDB."""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from ChromaDB."""
        try:
            if self.client:
                self.client = None
            return True
        except Exception as e:
            print(f"Error disconnecting from ChromaDB: {e}")
            return False

    async def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        """Add vectors to the store."""
        if not self.collection:
            raise RuntimeError("Vector store not connected")

        ids = [f"chunk_{i}_{m.get('chunk_id', '')}" for i, m in enumerate(metadata)]

        self.collection.add(
            embeddings=vectors,
            metadatas=metadata,
            ids=ids,
        )

        return ids

    async def search(
        self, query_vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self.collection:
            raise RuntimeError("Vector store not connected")

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter if filter else None,
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "score": float(results["distances"][0][i]),
                    "metadata": results["metadatas"][0][i],
                }
            )

        return formatted_results

    async def delete(self, doc_id: str) -> bool:
        """Delete vectors by document ID."""
        if not self.collection:
            raise RuntimeError("Vector store not connected")

        # Get all items with this document ID
        results = self.collection.get(where={"doc_id": doc_id})

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return True

        return False

    async def clear(self) -> bool:
        """Clear all vectors from the store."""
        if not self.collection:
            raise RuntimeError("Vector store not connected")

        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            return True
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            return False
