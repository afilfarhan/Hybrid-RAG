"""Redis cache implementation."""

from typing import Any, Optional
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache implementation."""
    
    def __init__(self, config: dict):
        """Initialize Redis cache.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.redis_url = config.get('redis_url', 'redis://localhost:6379')
        self.ttl = config.get('ttl', 3600)
        self.client = None
        
    async def connect(self):
        """Connect to Redis."""
        import redis.asyncio as redis
        
        self.client = await redis.from_url(
            self.redis_url,
            encoding='utf-8',
            decode_responses=True
        )
        logger.info("Connected to Redis")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Disconnected from Redis")
    
    async def _get_key(self, key: str) -> str:
        """Generate cache key.
        
        Args:
            key: Original key
            
        Returns:
            Hashed cache key
        """
        return f"rag:{hashlib.md5(key.encode()).hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.client:
            await self.connect()
        
        cache_key = await self._get_key(key)
        value = await self.client.get(cache_key)
        
        if value:
            logger.debug(f"Cache hit for key: {key[:50]}...")
            return json.loads(value)
        
        logger.debug(f"Cache miss for key: {key[:50]}...")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
            
        Returns:
            True if successful
        """
        if not self.client:
            await self.connect()
        
        cache_key = await self._get_key(key)
        ttl = ttl or self.ttl
        
        await self.client.setex(
            cache_key,
            ttl,
            json.dumps(value)
        )
        
        logger.debug(f"Cached value for key: {key[:50]}...")
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        if not self.client:
            await self.connect()
        
        cache_key = await self._get_key(key)
        await self.client.delete(cache_key)
        
        logger.debug(f"Deleted cache entry for key: {key[:50]}...")
        return True
    
    async def clear(self) -> bool:
        """Clear all cached values.
        
        Returns:
            True if successful
        """
        if not self.client:
            await self.connect()
        
        await self.client.flushdb()
        logger.info("Cleared all cache entries")
        return True
