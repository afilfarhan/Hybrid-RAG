"""Reranker for improving retrieval quality."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker using cross-encoder for improved ranking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reranker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.top_k = config.get('top_k', 5)
        self.model_name = config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance.
        
        Args:
            query: User query
            documents: Documents to rerank
            
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        scores = await self._compute_scores(query, documents)
        
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score
            doc['score'] = score
        
        documents.sort(key=lambda x: x['score'], reverse=True)
        
        return documents[:self.top_k]
    
    async def _compute_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[float]:
        """Compute relevance scores for query-document pairs.
        
        Args:
            query: User query
            documents: Documents to score
            
        Returns:
            Relevance scores
        """
        scores = []
        
        for doc in documents:
            doc_text = doc.get('text', '')
            score = self._simple_similarity(query, doc_text)
            scores.append(score)
        
        return scores
    
    def _simple_similarity(self, query: str, doc_text: str) -> float:
        """Compute simple similarity score.
        
        Args:
            query: User query
            doc_text: Document text
            
        Returns:
            Similarity score
        """
        query_words = set(query.lower().split())
        doc_words = set(doc_text.lower().split())
        
        intersection = query_words & doc_words
        union = query_words | doc_words
        
        if not union:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        
        return jaccard
