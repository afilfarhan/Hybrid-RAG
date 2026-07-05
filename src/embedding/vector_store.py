"""Vector store integration for Hybrid RAG system."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import hashlib

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store integration for Hybrid RAG system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.store_type = config.get('store_type', 'chroma')
        self.collection_name = config.get('collection_name', 'hybrid_rag_docs')
        self.persist_dir = Path(config.get('persist_dir', './data/chroma'))
        self.client = None
        self.collection = None
        
    async def connect(self):
        """Connect to vector store."""
        if self.store_type == 'chroma':
            import chromadb
            from chromadb.config import Settings
            
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Connected to ChromaDB at {self.persist_dir}")
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    async def disconnect(self):
        """Disconnect from vector store."""
        if self.client:
            self.client = None
            self.collection = None
            logger.info("Disconnected from vector store")
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Add documents to vector store.
        
        Args:
            documents: List of document chunks
            embeddings: List of embeddings
            
        Returns:
            List of document IDs
        """
        if not self.collection:
            await self.connect()
        
        ids = []
        metadatas = []
        texts = []
        
        for doc, embedding in zip(documents, embeddings):
            doc_id = self._generate_id(doc)
            ids.append(doc_id)
            
            metadata = {
                'source_type': doc.get('metadata', {}).get('source_type', 'unknown'),
                'source': doc.get('metadata', {}).get('file_path', doc.get('metadata', {}).get('url', 'unknown')),
                **doc.get('metadata', {})
            }
            metadatas.append(metadata)
            texts.append(doc.get('text', ''))
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(ids)} documents to vector store")
        return ids
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List of matching documents
        """
        if not self.collection:
            await self.connect()
        
        where_clause = None
        if metadata_filter:
            where_clause = self._build_where_clause(metadata_filter)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        return self._format_results(results)
    
    async def delete(self, doc_ids: List[str]) -> bool:
        """Delete documents from vector store.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful
        """
        if not self.collection:
            await self.connect()
        
        self.collection.delete(ids=doc_ids)
        logger.info(f"Deleted {len(doc_ids)} documents from vector store")
        return True
    
    async def get_count(self) -> int:
        """Get number of documents in store.
        
        Returns:
            Number of documents
        """
        if not self.collection:
            await self.connect()
        
        return self.collection.count()
    
    def _generate_id(self, doc: Dict[str, Any]) -> str:
        """Generate unique document ID.
        
        Args:
            doc: Document dictionary
            
        Returns:
            Unique document ID
        """
        content = doc.get('text', '')
        source = doc.get('metadata', {}).get('source', '')
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        
        return f"doc_{content_hash}"
    
    def _build_where_clause(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Build where clause for metadata filtering.
        
        Args:
            metadata_filter: Metadata filter
            
        Returns:
            Where clause dictionary
        """
        where_clause = {}
        
        for key, value in metadata_filter.items():
            if isinstance(value, str):
                where_clause[key] = value
            elif isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                where_clause[key] = value
        
        return where_clause
    
    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format query results.
        
        Args:
            results: Raw query results
            
        Returns:
            Formatted results
        """
        formatted = []
        
        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for i, (doc_id, doc_text, metadata, distance) in enumerate(zip(
            ids, documents, metadatas, distances
        )):
            formatted.append({
                'id': doc_id,
                'text': doc_text,
                'metadata': metadata,
                'score': 1.0 - distance,
                'rank': i + 1
            })
        
        return formatted
