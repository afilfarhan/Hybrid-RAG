"""Base class for ingestion pipelines."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseIngestionPipeline(ABC):
    """Abstract base class for ingestion pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ingestion pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sources: List[Dict[str, Any]] = []
        self.processed_count = 0
        self.failed_count = 0
        
    @abstractmethod
    async def add_source(self, source_config: Dict[str, Any]) -> bool:
        """Add a data source to the pipeline.
        
        Args:
            source_config: Configuration for the data source
            
        Returns:
            True if source was added successfully
        """
        pass
    
    @abstractmethod
    async def ingest(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """Ingest data from sources.
        
        Args:
            source_id: Optional specific source ID to ingest from
            
        Returns:
            Ingestion results summary
        """
        pass
    
    @abstractmethod
    async def ingest_all(self) -> Dict[str, Any]:
        """Ingest data from all configured sources.
        
        Returns:
            Ingestion results summary
        """
        pass
    
    @abstractmethod
    async def remove_source(self, source_id: str) -> bool:
        """Remove a data source from the pipeline.
        
        Args:
            source_id: ID of the source to remove
            
        Returns:
            True if source was removed successfully
        """
        pass
    
    @abstractmethod
    async def get_sources(self) -> List[Dict[str, Any]]:
        """Get list of all configured sources.
        
        Returns:
            List of source configurations
        """
        pass
    
    def _log_ingestion_result(self, source: str, status: str, details: Dict[str, Any]):
        """Log ingestion result.
        
        Args:
            source: Source identifier
            status: Ingestion status
            details: Additional details
        """
        logger.info(f"Ingestion {status} for source '{source}': {details}")
