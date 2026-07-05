"""Base class for caching."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract base class for caching."""
    
    def __init__(self, config: dict):
        """Initialize cache.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ttl = config.get('ttl', 3600)
        
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cached values.
        
        Returns:
            True if successful
        """
        pass
