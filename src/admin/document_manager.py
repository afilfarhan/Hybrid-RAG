"""Document manager for admin operations."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class DocumentManager:
    """Document manager for admin operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize document manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vector_store = config.get('vector_store')
        self.data_dir = Path(config.get('data_dir', './data/sources'))
        
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a new document to the system.
        
        Args:
            content: Document content
            metadata: Optional metadata
            
        Returns:
            Document info
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        doc_id = await self._generate_doc_id(content)
        
        document = {
            'text': content,
            'metadata': metadata or {}
        }
        
        await self.vector_store.add_documents([document], [[0.0] * 100])
        
        logger.info(f"Added document: {doc_id}")
        
        return {
            'doc_id': doc_id,
            'status': 'success',
            'content_preview': content[:100] + '...'
        }
    
    async def update_document(
        self,
        doc_id: str,
        new_content: str,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an existing document.
        
        Args:
            doc_id: Document ID
            new_content: New content
            new_metadata: Optional new metadata
            
        Returns:
            Update result
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        await self.vector_store.delete([doc_id])
        
        document = {
            'text': new_content,
            'metadata': new_metadata or {}
        }
        
        await self.vector_store.add_documents([document], [[0.0] * 100])
        
        logger.info(f"Updated document: {doc_id}")
        
        return {
            'doc_id': doc_id,
            'status': 'success'
        }
    
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Delete result
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        await self.vector_store.delete([doc_id])
        
        logger.info(f"Deleted document: {doc_id}")
        
        return {
            'doc_id': doc_id,
            'status': 'success'
        }
    
    async def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List documents.
        
        Args:
            limit: Maximum number of documents
            
        Returns:
            List of document info
        """
        if not self.vector_store:
            return []
        
        count = await self.vector_store.get_count()
        
        return {
            'total': count,
            'limit': limit,
            'documents': []
        }
    
    async def _generate_doc_id(self, content: str) -> str:
        """Generate document ID.
        
        Args:
            content: Document content
            
        Returns:
            Document ID
        """
        import hashlib
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        return f"doc_{content_hash}"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status.
        
        Returns:
            System status dictionary
        """
        if not self.vector_store:
            return {
                'status': 'error',
                'message': 'Vector store not initialized'
            }
        
        try:
            count = await self.vector_store.get_count()
            
            return {
                'status': 'healthy',
                'document_count': count,
                'data_dir': str(self.data_dir)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check results
        """
        status = await self.get_status()
        
        return {
            'component': 'document_manager',
            'healthy': status['status'] == 'healthy',
            'details': status
        }
