"""Dense retriever using vector embeddings."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retriever using vector embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize dense retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.top_k = config.get('top_k', 5)
        self.score_threshold = config.get('score_threshold', 0.7)
        self.embedding_service = config.get('embedding_service')
        self.vector_store = config.get('vector_store')
        
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using dense embeddings.
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents
        """
        query_embedding = await self.embedding_service.embed(query)
        
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )
        
        results = self._filter_results(results)
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    async def retrieve_with_filter(
        self,
        query: str,
        metadata_filter: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve documents with metadata filtering.
        
        Args:
            query: User query
            metadata_filter: Metadata filter
            
        Returns:
            List of relevant documents
        """
        query_embedding = await self.embedding_service.embed(query)
        
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            metadata_filter=metadata_filter
        )
        
        results = self._filter_results(results)
        
        logger.info(f"Retrieved {len(results)} documents with filter for query: {query[:50]}...")
        return results
    
    def _filter_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter results by score threshold.
        
        Args:
            results: Raw results
            
        Returns:
            Filtered results
        """
        return [r for r in results if r.get('score', 0) >= self.score_threshold]
