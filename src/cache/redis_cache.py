"""
Hybrid RAG - Caching module
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class BaseCache(ABC):
    """Base class for caching."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.ttl = config.get("ttl", 3600)  # Default 1 hour

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any) -> bool:
        """Set a value in the cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass


class RedisCache(BaseCache):
    """Cache using Redis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisCache")
        redis_url = config.get("url", "redis://localhost:6379")
        self.client = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if not REDIS_AVAILABLE:
            return None
        try:
            value = self.client.get(key)
            return value.decode("utf-8") if value else None
        except Exception:
            return None

    async def set(self, key: str, value: Any) -> bool:
        """Set a value in the cache."""
        if not REDIS_AVAILABLE:
            return False
        try:
            self.client.setex(key, self.ttl, str(value))
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        if not REDIS_AVAILABLE:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception:
            return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        if not REDIS_AVAILABLE:
            return False
        try:
            self.client.flushdb()
            return True
        except Exception:
            return False
