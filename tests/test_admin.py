"""
Tests for Hybrid RAG admin tools
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.admin.document_manager import DocumentManager
from src.admin.feedback_handler import FeedbackHandler


@pytest.mark.asyncio
async def test_document_manager_add():
    """Test adding a document."""
    mock_pipeline = AsyncMock()
    mock_pipeline.ingest.return_value = [MagicMock(doc_id="doc_123")]
    
    manager = DocumentManager(mock_pipeline)
    
    doc_id = await manager.add_document("content", {"source": "test.txt"})
    
    assert doc_id == "doc_123"


@pytest.mark.asyncio
async def test_document_manager_update():
    """Test updating a document."""
    mock_pipeline = AsyncMock()
    mock_pipeline.update.return_value = True
    
    manager = DocumentManager(mock_pipeline)
    
    result = await manager.update_document("doc_123", "new content", {"source": "test.txt"})
    
    assert result is True


@pytest.mark.asyncio
async def test_document_manager_delete():
    """Test deleting a document."""
    mock_pipeline = AsyncMock()
    mock_pipeline.delete.return_value = True
    
    manager = DocumentManager(mock_pipeline)
    
    result = await manager.delete_document("doc_123")
    
    assert result is True


@pytest.mark.asyncio
async def test_feedback_handler_submit():
    """Test submitting feedback."""
    handler = FeedbackHandler()
    
    result = await handler.submit_feedback(
        query_id="query_123",
        rating=5,
        comment="Great answer!",
    )
    
    assert result is True


@pytest.mark.asyncio
async def test_feedback_handler_invalid_rating():
    """Test that invalid ratings are rejected."""
    handler = FeedbackHandler()
    
    result = await handler.submit_feedback(
        query_id="query_123",
        rating=6,  # Invalid rating
    )
    
    assert result is False


@pytest.mark.asyncio
async def test_feedback_handler_get_statistics():
    """Test getting feedback statistics."""
    handler = FeedbackHandler()
    
    # Submit some feedback
    await handler.submit_feedback("q1", 5)
    await handler.submit_feedback("q2", 4)
    await handler.submit_feedback("q3", 3)
    
    stats = await handler.get_statistics()
    
    assert stats["total"] == 3
    assert stats["avg_rating"] == 4.0
    assert stats["rating_distribution"]["5"] == 1
