"""
Tests for Hybrid RAG caching
"""

import pytest
from src.cache.redis_cache import RedisCache, REDIS_AVAILABLE


@pytest.fixture
def redis_cache():
    """Skip tests if Redis is not available or server not running."""
    if not REDIS_AVAILABLE:
        pytest.skip("redis package not installed")
    
    try:
        cache = RedisCache({"url": "redis://localhost:6379", "ttl": 60})
        cache.client.ping()
        return cache
    except Exception:
        pytest.skip("Redis server not available")


@pytest.mark.asyncio
async def test_redis_cache_set_get(redis_cache):
    """Test basic cache set and get."""
    result = await redis_cache.set("test_key", "test_value")
    assert result is True
    
    value = await redis_cache.get("test_key")
    assert value == "test_value"


@pytest.mark.asyncio
async def test_redis_cache_delete(redis_cache):
    """Test cache delete."""
    await redis_cache.set("delete_test", "value")
    
    result = await redis_cache.delete("delete_test")
    assert result is True
    
    value = await redis_cache.get("delete_test")
    assert value is None


@pytest.mark.asyncio
async def test_redis_cache_nonexistent_key(redis_cache):
    """Test getting a non-existent key."""
    value = await redis_cache.get("nonexistent_key")
    assert value is None
