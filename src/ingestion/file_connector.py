"""
Hybrid RAG - File-based ingestion connector
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion.base import BaseIngestionPipeline, Document


class TextConnector(BaseIngestionPipeline):
    """Ingest text files (txt, md, html, docx)."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".docx"}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.source_dir = Path(config.get("source_dir", "./data"))
        self.supported_extensions = set(
            config.get("supported_extensions", [".txt", ".md", ".html"])
        )

    async def ingest(self, source: str) -> List[Document]:
        """Ingest a single file."""
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported: {self.supported_extensions}"
            )

        content = await self._read_file(file_path)
        doc_id = self._generate_id(file_path, content)

        return [
            Document(
                content=content,
                metadata={
                    "source": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "doc_id": doc_id,
                    "ingested_at": self._get_timestamp(),
                },
            )
        ]

    async def ingest_all(self) -> List[Document]:
        """Ingest all supported files from source directory."""
        documents = []

        for file_path in self.source_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    docs = await self.ingest(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")

        return documents

    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ID (not applicable for file connector)."""
        return False

    async def update(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Update a document (not applicable for file connector)."""
        return False

    async def _read_file(self, file_path: Path) -> str:
        """Read file content."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _generate_id(self, file_path: Path, content: str) -> str:
        """Generate document ID."""
        from src.core.utils import generate_id

        return generate_id(f"{file_path}:{content[:100]}")

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from src.core.utils import format_timestamp

        return format_timestamp()
