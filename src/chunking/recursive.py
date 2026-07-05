"""
Hybrid RAG - Recursive character chunker
"""

import re
from typing import Any, Dict, List, Optional

from src.chunking.base import BaseChunker, Chunk
from src.ingestion.base import Document


class RecursiveCharacterChunker(BaseChunker):
    """Chunk documents using recursive character splitting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 51)
        self.separators = config.get(
            "separators", ["\n\n", "\n", " ", ""]
        )  # Split by paragraphs, then lines, then words

    async def chunk(self, document: Document) -> List[Chunk]:
        """Chunk a single document using recursive character splitting."""
        text = document.content
        chunks = []

        # Split text recursively
        split_texts = self._split_text(text, self.chunk_size, self.chunk_overlap)

        for i, chunk_text in enumerate(split_texts):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(split_texts),
                    "chunk_id": f"{document.doc_id}_chunk_{i}",
                }
            )

            chunks.append(
                Chunk(
                    content=chunk_text,
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

    def _split_text(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Split text into chunks using recursive character splitting."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Find a good split point within the chunk
            if end < len(text):
                # Try to split at separators
                split_text = text[start:end]

                for sep in self.separators[:-1]:  # Exclude empty string
                    if sep in split_text:
                        last_sep_pos = split_text.rfind(sep)
                        if last_sep_pos > len(split_text) * 0.5:  # Only if in second half
                            end = start + last_sep_pos + len(sep)
                            break

            chunks.append(text[start:end])

            # Move to next chunk with overlap
            start = max(end - chunk_overlap, start + 1)

        return chunks
