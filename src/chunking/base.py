"""Base class for chunkers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for chunkers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text into smaller units.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        pass
    
    @abstractmethod
    async def chunk_async(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Async version of chunk.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        pass
    
    def _create_chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0,
        total_chunks: int = 1
    ) -> Dict[str, Any]:
        """Create a chunk dictionary.
        
        Args:
            text: Chunk text
            metadata: Original metadata
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            
        Returns:
            Chunk dictionary
        """
        chunk = {
            'text': text,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'metadata': metadata or {}
        }
        
        if metadata:
            chunk['metadata'] = {**metadata}
        
        return chunk
