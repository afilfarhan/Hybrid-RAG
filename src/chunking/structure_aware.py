"""
Hybrid RAG - Structure-aware chunker
"""

from typing import Any, Dict, List, Optional

from src.chunking.base import BaseChunker, Chunk
from src.ingestion.base import Document


class StructureAwareChunker(BaseChunker):
    """Chunk documents respecting their structure (headings, sections)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 51)

    async def chunk(self, document: Document) -> List[Chunk]:
        """Chunk a document by preserving its structure."""
        text = document.content
        chunks = []

        # Split by headings (Markdown-style)
        sections = self._split_by_headings(text)

        current_chunk = ""
        chunk_index = 0

        for section in sections:
            section_lines = section.strip().split("\n")

            if not section_lines:
                continue

            # Check if adding this section would exceed chunk size
            if len(current_chunk) + len(section) > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk(current_chunk, document, chunk_index))
                chunk_index += 1
                current_chunk = ""

            current_chunk += section + "\n\n"

        # Add remaining content
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, document, chunk_index))

        return chunks

    async def chunk_batch(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = await self.chunk(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _split_by_headings(self, text: str) -> List[str]:
        """Split text by Markdown headings (#, ##, ###)."""
        # Pattern to match Markdown headings
        heading_pattern = r"^(#{1,6})\s+(.+)$"
        lines = text.split("\n")

        sections = []
        current_section = []

        for line in lines:
            if line.strip() and line[0] == "#":
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections

    def _create_chunk(
        self, content: str, document: Document, chunk_index: int
    ) -> Chunk:
        """Create a chunk with metadata."""
        chunk_metadata = document.metadata.copy()
        chunk_metadata.update(
            {
                "chunk_index": chunk_index,
                "total_chunks": 0,  # Will be updated later
                "chunk_id": f"{document.doc_id}_chunk_{chunk_index}",
                "chunk_type": "structure_aware",
            }
        )

        return Chunk(
            content=content.strip(),
            metadata=chunk_metadata,
            chunk_index=chunk_index,
        )
