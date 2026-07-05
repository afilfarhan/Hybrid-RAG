"""
Tests for Hybrid RAG API endpoints
"""

from src.api.endpoints import router


def test_router_exists():
    """Test that router is properly defined."""
    assert router is not None
    assert hasattr(router, "routes")
    assert len(router.routes) > 0


def test_routes_defined():
    """Test that expected routes are defined."""
    route_paths = [str(route.path) for route in router.routes]
    assert "/health" in route_paths
    assert "/" in route_paths
    assert "/status" in route_paths
    assert "/query" in route_paths
    assert "/feedback" in route_paths
