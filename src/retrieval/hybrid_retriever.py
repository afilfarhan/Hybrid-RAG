"""Hybrid retriever combining dense and sparse search."""

from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.top_k = config.get('top_k', 5)
        self.score_threshold = config.get('score_threshold', 0.7)
        self.dense_weight = config.get('dense_weight', 0.7)
        self.embedding_service = config.get('embedding_service')
        self.vector_store = config.get('vector_store')
        
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid search.
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents
        """
        query_embedding = await self.embedding_service.embed(query)
        
        dense_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k * 2
        )
        
        sparse_results = await self._sparse_search(query, self.top_k * 2)
        
        combined_results = self._combine_results(
            dense_results,
            sparse_results,
            self.dense_weight
        )
        
        combined_results = self._filter_results(combined_results)
        
        logger.info(f"Retrieved {len(combined_results)} documents (hybrid) for query: {query[:50]}...")
        return combined_results
    
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
        
        dense_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k * 2,
            metadata_filter=metadata_filter
        )
        
        sparse_results = await self._sparse_search(query, self.top_k * 2)
        
        combined_results = self._combine_results(
            dense_results,
            sparse_results,
            self.dense_weight
        )
        
        combined_results = self._filter_results(combined_results)
        
        logger.info(f"Retrieved {len(combined_results)} documents (hybrid+filter) for query: {query[:50]}...")
        return combined_results
    
    async def _sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform sparse keyword search.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Sparse search results
        """
        import re
        
        query_terms = query.lower().split()
        
        all_docs = await self._get_all_documents()
        
        scores = []
        for doc in all_docs:
            text = doc.get('text', '').lower()
            score = self._bm25_score(text, query_terms)
            if score > 0:
                scores.append({
                    'id': doc.get('id'),
                    'text': doc.get('text'),
                    'metadata': doc.get('metadata'),
                    'score': score
                })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def _bm25_score(self, text: str, query_terms: List[str], k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score.
        
        Args:
            text: Document text
            query_terms: Query terms
            k1: BM25 parameter
            b: BM25 parameter
            
        Returns:
            BM25 score
        """
        words = text.lower().split()
        doc_len = len(words)
        avg_doc_len = sum(len(w) for w in words) / max(len(words), 1)
        
        score = 0
        for term in query_terms:
            if term in words:
                term_freq = words.count(term)
                idf = 1.0
                score += idf * (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_len / avg_doc_len))
        
        return score
    
    async def _get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from vector store.
        
        Returns:
            List of all documents
        """
        count = await self.vector_store.get_count()
        
        if count == 0:
            return []
        
        results = self.vector_store.collection.get(
            include=['documents', 'metadatas'],
            limit=min(count, 1000)
        )
        
        formatted = []
        for i, doc_id in enumerate(results.get('ids', [])):
            formatted.append({
                'id': doc_id,
                'text': results['documents'][i] if i < len(results['documents']) else '',
                'metadata': results['metadatas'][i] if i < len(results['metadatas']) else {}
            })
        
        return formatted
    
    def _combine_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine dense and sparse results using RRF.
        
        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results
            dense_weight: Weight for dense results
            
        Returns:
            Combined results
        """
        all_results = {}
        
        for i, doc in enumerate(dense_results):
            doc_id = doc.get('id')
            rank = i + 1
            score = doc.get('score', 0)
            
            if doc_id not in all_results:
                all_results[doc_id] = {
                    'id': doc_id,
                    'text': doc.get('text'),
                    'metadata': doc.get('metadata'),
                    'dense_score': score,
                    'sparse_score': 0,
                    'combined_score': 0
                }
            
            all_results[doc_id]['dense_score'] = score
        
        for i, doc in enumerate(sparse_results):
            doc_id = doc.get('id')
            rank = i + 1
            score = doc.get('score', 0)
            
            if doc_id not in all_results:
                all_results[doc_id] = {
                    'id': doc_id,
                    'text': doc.get('text'),
                    'metadata': doc.get('metadata'),
                    'dense_score': 0,
                    'sparse_score': score,
                    'combined_score': 0
                }
            
            all_results[doc_id]['sparse_score'] = score
        
        for doc_id, doc in all_results.items():
            combined_score = (
                dense_weight * doc['dense_score'] +
                (1 - dense_weight) * doc['sparse_score']
            )
            doc['combined_score'] = combined_score
            doc['score'] = combined_score
        
        results = list(all_results.values())
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
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
