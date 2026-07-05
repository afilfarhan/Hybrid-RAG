"""Base class for generators."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract base class for generators."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 1000)
        
    @abstractmethod
    async def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate answer from query and context.
        
        Args:
            query: User query
            context: Retrieved context documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer with metadata
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ):
        """Generate answer with streaming.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Yields:
            Streaming response chunks
        """
        pass
