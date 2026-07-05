"""Base class for tracing."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTracer(ABC):
    """Abstract base class for tracing."""
    
    def __init__(self, config: dict):
        """Initialize tracer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
    @abstractmethod
    async def trace_query(
        self,
        query: str,
        steps: Dict[str, Any]
    ) -> str:
        """Trace a query through the RAG pipeline.
        
        Args:
            query: User query
            steps: Pipeline steps with results
            
        Returns:
            Trace ID
        """
        pass
    
    @abstractmethod
    async def trace_feedback(
        self,
        trace_id: str,
        feedback: Dict[str, Any]
    ) -> bool:
        """Record feedback for a traced query.
        
        Args:
            trace_id: Trace ID
            feedback: Feedback data
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a traced query by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace data or None
        """
        pass
