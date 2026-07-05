"""Base class for retrievers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.top_k = config.get('top_k', 5)
        self.score_threshold = config.get('score_threshold', 0.7)
        
    @abstractmethod
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents
        """
        pass
    
    @abstractmethod
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
        pass
    
    def _filter_results(
        self,
        results: List[Dict[str, Any]],
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Filter results by score threshold.
        
        Args:
            results: Raw results
            score_threshold: Minimum score threshold
            
        Returns:
            Filtered results
        """
        threshold = score_threshold or self.score_threshold
        return [r for r in results if r.get('score', 0) >= threshold]
