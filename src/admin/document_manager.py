"""
Hybrid RAG - Admin tools
"""

from typing import Any, Dict, List, Optional

from src.ingestion.base import BaseIngestionPipeline, Document


class DocumentManager:
    """Manager for documents in the knowledge base."""

    def __init__(
        self,
        ingestion_pipeline: BaseIngestionPipeline,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.ingestion_pipeline = ingestion_pipeline
        self.config = config or {}

    async def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add a new document."""
        doc = Document(content=content, metadata=metadata)
        documents = await self.ingestion_pipeline.ingest(doc)
        return documents[0].doc_id if documents else ""

    async def update_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update an existing document."""
        return await self.ingestion_pipeline.update(doc_id, content, metadata)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        return await self.ingestion_pipeline.delete(doc_id)

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        # TODO: Implement document listing
        return []

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        # TODO: Implement document retrieval
        return None
