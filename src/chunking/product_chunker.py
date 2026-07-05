"""Product-specific chunker that keeps product records atomic."""

from typing import List, Dict, Any, Optional
import re
import json
import logging

logger = logging.getLogger(__name__)


class ProductChunker:
    """Chunker designed for product catalog data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize product chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 0)
        
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk product data keeping records atomic.
        
        Args:
            text: Product data to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        products = self._extract_products(text)
        chunks = []
        
        for i, product in enumerate(products):
            chunk = self._create_chunk(product, metadata, i)
            chunks.append(chunk)
        
        return chunks
    
    def _extract_products(self, text: str) -> List[str]:
        """Extract product records from text.
        
        Args:
            text: Text containing product data
            
        Returns:
            List of product records
        """
        products = []
        
        patterns = [
            r'Product\s*\d+[:\s]+.*?(?=Product\s*\d+|$)',
            r'\{.*?\}',
            r'Product Name:.*?(?=Product Name:|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    products.append(match.strip())
                break
        
        if not products:
            products = [text]
        
        return products
    
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
