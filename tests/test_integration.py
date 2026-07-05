"""
Hybrid RAG - Integration tests
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.ingestion.file_connector import TextConnector
from src.chunking.recursive import RecursiveCharacterChunker
from src.embedding.openai_service import OpenAIEmbeddingService
from src.embedding.chroma_store import ChromaDBVectorStore
from src.retrieval.dense_retriever import DenseRetriever


@pytest.mark.asyncio
async def test_full_ingestion_pipeline():
    """Test full ingestion pipeline with file connector and chunker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test document
        doc_path = Path(tmpdir) / "test.md"
        doc_path.write_text(
            "# Test Document\n\nThis is a test document with multiple sections.\n\n## Section 1\n\nContent for section 1.\n\n## Section 2\n\nContent for section 2."
        )
        
        # Ingest
        connector = TextConnector({"source_dir": tmpdir})
        documents = await connector.ingest_all()
        
        assert len(documents) == 1
        assert documents[0].content.startswith("# Test Document")
        
        # Chunk
        chunker = RecursiveCharacterChunker({"chunk_size": 100})
        chunks = await chunker.chunk(documents[0])
        
        assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_vector_store_operations():
    """Test basic vector store operations."""
    import sys
    if sys.platform == "win32":
        pytest.skip("ChromaDB temp directory cleanup fails on Windows")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaDBVectorStore(
            {"collection_name": "test_collection", "persist_path": tmpdir}
        )
        
        # Connect
        connected = await store.connect()
        assert connected is True
        
        # Add vectors
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [{"source": "test1"}, {"source": "test2"}]
        
        ids = await store.add(vectors, metadata)
        assert len(ids) == 2
        
        # Search
        results = await store.search([0.1, 0.2, 0.3], top_k=2)
        assert len(results) >= 1
        
        # Disconnect
        disconnected = await store.disconnect()
        assert disconnected is True


@pytest.mark.asyncio
async def test_dense_retriever():
    """Test dense retriever integration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock embedding service
        class MockEmbeddingService:
            async def embed(self, text):
                return [0.1] * 1536  # Mock embedding dimension
            
            async def embed_batch(self, texts):
                return [[0.1] * 1536] * len(texts)
            
            async def get_dimension(self):
                return 1536
        
        # Create mock vector store
        class MockVectorStore:
            async def connect(self):
                return True
            
            async def search(self, query_vector, top_k=5, filter=None):
                return [
                    {
                        "id": "chunk_1",
                        "score": 0.1,
                        "metadata": {
                            "content": "Test chunk content",
                            "source": "test.txt",
                            "chunk_id": "chunk_1",
                        },
                    }
                ]
        
        # Create retriever
        retriever = DenseRetriever(
            embedding_service=MockEmbeddingService(),
            vector_store=MockVectorStore(),
            config={"top_k": 5},
        )
        
        # Retrieve
        chunks = await retriever.retrieve("test query", top_k=5)
        
        assert len(chunks) == 1
        assert chunks[0].content == "Test chunk content"
