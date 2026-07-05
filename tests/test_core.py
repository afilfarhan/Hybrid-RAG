"""
Tests for Hybrid RAG core utilities
"""

import pytest
from src.core.utils import generate_id, clean_text, format_timestamp, chunk_list


class TestGenerateId:
    def test_generates_consistent_id(self):
        """Test that same text produces same ID."""
        text = "test document"
        id1 = generate_id(text)
        id2 = generate_id(text)
        assert id1 == id2

    def test_different_texts_different_ids(self):
        """Test that different texts produce different IDs."""
        text1 = "test document 1"
        text2 = "test document 2"
        id1 = generate_id(text1)
        id2 = generate_id(text2)
        assert id1 != id2

    def test_id_length(self):
        """Test that generated ID has expected length."""
        text = "test"
        id_val = generate_id(text)
        assert len(id_val) == 32  # MD5 hash is 32 hex chars


class TestCleanText:
    def test_removes_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        text = "hello    world   test"
        cleaned = clean_text(text)
        assert cleaned == "hello world test"

    def test_trims_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        text = "  hello world  "
        cleaned = clean_text(text)
        assert cleaned == "hello world"

    def test_preserves_content(self):
        """Test that actual content is preserved."""
        text = "Hello World!"
        cleaned = clean_text(text)
        assert cleaned == "Hello World!"


class TestFormatTimestamp:
    def test_iso_format(self):
        """Test that timestamp is in ISO format."""
        timestamp = format_timestamp()
        assert "T" in timestamp
        assert timestamp.endswith("Z")


class TestChunkList:
    def test_chunks_equal_size(self):
        """Test that list is chunked into equal sizes."""
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        chunks = chunk_list(lst, 3)
        assert len(chunks) == 3
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7, 8, 9]

    def test_chunks_unequal_size(self):
        """Test that last chunk can be smaller."""
        lst = [1, 2, 3, 4, 5]
        chunks = chunk_list(lst, 2)
        assert len(chunks) == 3
        assert chunks[0] == [1, 2]
        assert chunks[1] == [3, 4]
        assert chunks[2] == [5]

    def test_empty_list(self):
        """Test that empty list returns empty chunks."""
        lst = []
        chunks = chunk_list(lst, 3)
        assert chunks == []

    def test_single_element(self):
        """Test that single element list works."""
        lst = [1]
        chunks = chunk_list(lst, 3)
        assert chunks == [[1]]
