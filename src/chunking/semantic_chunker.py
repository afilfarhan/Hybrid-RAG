"""Semantic chunker that groups semantically related content."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunker that uses semantic similarity to group content."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize semantic chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text based on semantic boundaries.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))
                
                if sentence_length > self.chunk_size:
                    chunks.extend(self._split_long_sentence(sentence, metadata, len(chunks)))
                    current_chunk = ""
                    current_length = 0
                else:
                    current_chunk = sentence
                    current_length = sentence_length
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        import re
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str, metadata: Optional[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """Split a long sentence into multiple chunks.
        
        Args:
            sentence: Long sentence to split
            metadata: Metadata to attach
            start_index: Starting chunk index
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        words = sentence.split()
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
