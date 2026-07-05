"""
Hybrid RAG - Product chunker
"""

from typing import Any, Dict, List, Optional
import re

from src.chunking.base import BaseChunker, Chunk
from src.ingestion.base import Document


class ProductChunker(BaseChunker):
    """Chunk product catalogs, keeping product records atomic."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 512)

    async def chunk(self, document: Document) -> List[Chunk]:
        """Chunk product catalog, keeping each product as one chunk."""
        text = document.content
        chunks = []

        # Split by product sections
        products = self._split_by_products(text)

        for i, product in enumerate(products):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(products),
                    "chunk_id": f"{document.doc_id}_product_{i}",
                    "chunk_type": "product",
                }
            )

            chunks.append(
                Chunk(
                    content=product.strip(),
                    metadata=chunk_metadata,
                    chunk_index=i,
                )
            )

        return chunks

    async def chunk_batch(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = await self.chunk(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _split_by_products(self, text: str) -> List[str]:
        """Split text into product records."""
        # Pattern to match product sections
        # Looks for lines starting with **Product: or **Product ID:
        product_pattern = r"(\*\*Product:\*\*.*?)(?=\*\*Product:|\Z)"

        matches = re.findall(product_pattern, text, re.DOTALL)

        if matches:
            return matches

        # Fallback: split by empty lines and look for product markers
        sections = re.split(r"\n\s*\n", text)
        products = []

        for section in sections:
            if any(
                marker in section
                for marker in ["Product ID:", "**Product:", "Product:"]
            ):
                products.append(section)

        return products if products else [text]
