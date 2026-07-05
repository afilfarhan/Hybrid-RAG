"""
Hybrid RAG - FAQ chunker
"""

from typing import Any, Dict, List, Optional

from src.chunking.base import BaseChunker, Chunk
from src.ingestion.base import Document


class FAQChunker(BaseChunker):
    """Chunk FAQ documents, keeping Q&A pairs together."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 512)

    async def chunk(self, document: Document) -> List[Chunk]:
        """Chunk FAQ document preserving Q&A pairs."""
        text = document.content
        chunks = []

        # Split by question patterns
        faq_items = self._split_by_qa(text)

        for i, faq_item in enumerate(faq_items):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(faq_items),
                    "chunk_id": f"{document.doc_id}_faq_{i}",
                    "chunk_type": "faq",
                }
            )

            chunks.append(
                Chunk(
                    content=faq_item.strip(),
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

    def _split_by_qa(self, text: str) -> List[str]:
        """Split text into Q&A pairs."""
        # Patterns for questions
        patterns = [
            r"\*\*Q:\*\*.*?\*\*A:\*\*.*?(?=\*\*Q:|\Z)",  # **Q:** ... **A:** ...
            r"Q:\s*.*?A:\s*.*?(?=\nQ:|\Z)",  # Q: ... A: ...
            r"Q:\s*.*?(?=\nA:|\n\n)",  # Q: ...
        ]

        items = []
        lines = text.split("\n")

        current_qa = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Check if this is a new question
            if line.startswith(("**Q:**", "Q:", "Q:")) or (
                len(line) < 100 and line.endswith("?")
            ):
                if current_qa:
                    items.append("\n".join(current_qa))
                current_qa = [line]
            else:
                current_qa.append(line)

        if current_qa:
            items.append("\n".join(current_qa))

        return items
