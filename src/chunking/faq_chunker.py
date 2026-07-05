"""FAQ-specific chunker that keeps Q&A pairs together."""

from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class FAQChunker:
    """Chunker designed for FAQ content."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FAQ chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk FAQ content keeping Q&A pairs together.
        
        Args:
            text: FAQ text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        faqs = self._extract_faqs(text)
        chunks = []
        
        for i, faq in enumerate(faqs):
            chunk = self._create_chunk(faq, metadata, i)
            chunks.append(chunk)
        
        return chunks
    
    def _extract_faqs(self, text: str) -> List[str]:
        """Extract Q&A pairs from text.
        
        Args:
            text: Text containing FAQs
            
        Returns:
            List of Q&A pairs
        """
        faqs = []
        
        patterns = [
            r'(Q[ues]?estion[:\s]+.*?A[ns]?swer[:\s]+.*?)(?=(Q[ues]?estion|A[ns]?swer|$))',
            r'(Q[:\s]+.*?A[:\s]+.*?)(?=(Q|A|$))',
            r'(\d+\.\s*.*?\n.*?)(?=\n\d+\.|\Z)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        faqs.append(match[0].strip())
                    else:
                        faqs.append(match.strip())
                break
        
        if not faqs:
            faqs = [text]
        
        return faqs
    
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
