"""Integration tests for Hybrid RAG system."""

import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIntegration:
    """Integration tests for the RAG system."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test the full RAG pipeline."""
        from src.core.config import Settings
        from src.ingestion.file_connector import TextConnector
        from src.chunking.structure_chunker import StructureAwareChunker
        from src.embedding.openai_service import OpenAIEmbeddingService
        from src.embedding.vector_store import VectorStore
        
        config = Settings()
        
        # Test ingestion
        text_connector = TextConnector({
            'source_dir': './data/sample_docs',
            'supported_extensions': ['.md']
        })
        
        files = text_connector._get_files()
        assert len(files) == 0  # No files yet
        
        # Test chunking
        chunker = StructureAwareChunker({'chunk_size': 100})
        text = "# Test\n\nContent here."
        chunks = chunker.chunk(text)
        assert len(chunks) > 0
        
        # Test vector store
        vector_store = VectorStore({
            'store_type': 'chroma',
            'collection_name': 'test_collection',
            'persist_dir': './data/chroma_test'
        })
        
        # This would normally connect to the database
        # For now, just test initialization
        assert vector_store.store_type == 'chroma'


class TestSampleData:
    """Tests for sample data generation."""
    
    def test_sample_documents(self):
        """Test sample document creation."""
        from pathlib import Path
        
        sample_docs_dir = Path('./data/sample_docs')
        sample_docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample document
        test_file = sample_docs_dir / 'test.md'
        test_file.write_text('# Test Document\n\nContent here.')
        
        assert test_file.exists()
        assert test_file.read_text() == '# Test Document\n\nContent here.'
        
        # Cleanup
        test_file.unlink()
        
        # Cleanup directory if empty
        try:
            sample_docs_dir.rmdir()
        except:
            pass
