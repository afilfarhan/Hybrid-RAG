"""Base class for admin tools."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAdminTool(ABC):
    """Abstract base class for admin tools."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize admin tool.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get system status.
        
        Returns:
            System status dictionary
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check results
        """
        pass
