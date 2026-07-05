"""Recursive character-based chunker."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """Chunker that recursively splits text by characters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize recursive chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.separators = config.get('separators', ['\n\n', '\n', ' ', ''])
        
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text recursively by separators.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        lines = text.split('\n')
        
        for line in lines:
            line_length = len(line) + 1
            
            if current_length + line_length <= self.chunk_size:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
                current_length += line_length
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))
                
                if line_length > self.chunk_size:
                    chunks.extend(self._split_long_line(line, metadata, len(chunks)))
                    current_chunk = ""
                    current_length = 0
                else:
                    current_chunk = line
                    current_length = line_length
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))
        
        return chunks
    
    def _split_long_line(self, line: str, metadata: Optional[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """Split a long line into multiple chunks.
        
        Args:
            line: Long line to split
            metadata: Metadata to attach
            start_index: Starting chunk index
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        words = line.split()
        current_chunk = ""
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            
            if current_length + word_length <= self.chunk_size:
                if current_chunk:
                    current_chunk += ' ' + word
                else:
                    current_chunk = word
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata, start_index + len(chunks)))
                current_chunk = word
                current_length = word_length
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata, start_index + len(chunks)))
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0
    ) -> Dict[str, Any]:
        """Create a chunk dictionary.
        
        Args:
            text: Chunk text
            metadata: Original metadata
            chunk_index: Index of this chunk
            
        Returns:
            Chunk dictionary
        """
        return {
            'text': text,
            'chunk_index': chunk_index,
            'total_chunks': 0,
            'metadata': metadata or {}
        }
