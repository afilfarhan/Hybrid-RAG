"""
Tests for Hybrid RAG ingestion
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.ingestion.file_connector import TextConnector
from src.ingestion.base import Document


@pytest.mark.asyncio
async def test_text_connector_single_file():
    """Test ingesting a single file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test content")
        temp_path = f.name
    
    try:
        connector = TextConnector({"source_dir": os.path.dirname(temp_path)})
        docs = await connector.ingest(temp_path)
        
        assert len(docs) == 1
        assert docs[0].content == "Test content"
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_text_connector_directory():
    """Test ingesting all files from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        file1 = Path(tmpdir) / "test1.txt"
        file2 = Path(tmpdir) / "test2.txt"
        
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        connector = TextConnector({"source_dir": tmpdir})
        docs = await connector.ingest_all()
        
        assert len(docs) == 2


@pytest.mark.asyncio
async def test_text_connector_unsupported_extension():
    """Test that unsupported extensions raise error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write("Test content")
        temp_path = f.name
    
    try:
        connector = TextConnector({"source_dir": os.path.dirname(temp_path)})
        
        with pytest.raises(ValueError):
            await connector.ingest(temp_path)
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_text_connector_file_not_found():
    """Test that non-existent files raise error."""
    connector = TextConnector({"source_dir": "."})
    
    with pytest.raises(FileNotFoundError):
        await connector.ingest("/nonexistent/file.txt")
