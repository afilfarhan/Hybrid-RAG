"""
Tests for Hybrid RAG chunking
"""

import pytest
from src.ingestion.base import Document
from src.chunking.recursive import RecursiveCharacterChunker
from src.chunking.structure_aware import StructureAwareChunker
from src.chunking.faq import FAQChunker
from src.chunking.product import ProductChunker


@pytest.mark.asyncio
async def test_recursive_chunker_basic():
    """Test basic chunking functionality."""
    doc = Document(
        content="This is a test document. " * 20,
        metadata={"source": "test.txt"},
    )
    
    chunker = RecursiveCharacterChunker({"chunk_size": 100, "chunk_overlap": 10})
    chunks = await chunker.chunk(doc)
    
    assert len(chunks) > 0
    assert chunks[0].metadata["chunk_index"] == 0


@pytest.mark.asyncio
async def test_structure_aware_chunker():
    """Test structure-aware chunking."""
    content = """# Section 1
Content for section 1.

# Section 2
Content for section 2.
"""
    
    doc = Document(
        content=content,
        metadata={"source": "doc.md"},
    )
    
    chunker = StructureAwareChunker({"chunk_size": 512})
    chunks = await chunker.chunk(doc)
    
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_faq_chunker():
    """Test FAQ chunking."""
    content = """**Q:** What is your return policy?
**A:** We offer a 30-day return policy.

**Q:** How do I track my order?
**A:** You can track your order using the tracking link in your confirmation email.
"""
    
    doc = Document(
        content=content,
        metadata={"source": "faq.md"},
    )
    
    chunker = FAQChunker()
    chunks = await chunker.chunk(doc)
    
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_product_chunker():
    """Test product chunking."""
    content = """**Product:** Premium Headphones
**Product ID:** WH-001
**Price:** $299.99

**Product:** Smart Watch
**Product ID:** SW-002
**Price:** $399.99
"""
    
    doc = Document(
        content=content,
        metadata={"source": "products.md"},
    )
    
    chunker = ProductChunker()
    chunks = await chunker.chunk(doc)
    
    assert len(chunks) >= 2
