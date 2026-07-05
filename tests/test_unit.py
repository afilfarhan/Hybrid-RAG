# Sample unit tests for Hybrid RAG system.

import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChunking:
    """Tests for chunking module."""
    
    def test_recursive_chunker(self):
        """Test recursive chunker."""
        from src.chunking.recursive_chunker import RecursiveChunker
        
        config = {'chunk_size': 100, 'chunk_overlap': 10}
        chunker = RecursiveChunker(config)
        
        text = "This is a test document. " * 20
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'text' in chunk
            assert 'chunk_index' in chunk
    
    def test_structure_chunker(self):
        """Test structure-aware chunker."""
        from src.chunking.structure_chunker import StructureAwareChunker
        
        config = {'chunk_size': 100, 'chunk_overlap': 10}
        chunker = StructureAwareChunker(config)
        
        text = "# Heading 1\n\nContent here.\n\n# Heading 2\n\nMore content."
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
    
    def test_faq_chunker(self):
        """Test FAQ chunker."""
        from src.chunking.faq_chunker import FAQChunker
        
        config = {'chunk_size': 512}
        chunker = FAQChunker(config)
        
        text = "Q: What is your return policy?\nA: 30 days."
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
    
    def test_text_preprocessor(self):
        """Test text preprocessor."""
        from src.chunking.preprocessor import TextPreprocessor
        
        config = {
            'remove_html': True,
            'normalize_whitespace': True,
            'lowercase': False
        }
        preprocessor = TextPreprocessor(config)
        
        text = "<p>Hello World!</p>"
        cleaned = preprocessor.preprocess(text)
        
        assert "Hello World!" in cleaned


class TestRetrieval:
    """Tests for retrieval module."""
    
    def test_dense_retriever(self):
        """Test dense retriever initialization."""
        from src.retrieval.dense_retriever import DenseRetriever
        from unittest.mock import MagicMock
        
        mock_embedding_service = MagicMock()
        mock_vector_store = MagicMock()
        
        config = {'top_k': 5, 'score_threshold': 0.7}
        retriever = DenseRetriever(mock_embedding_service, mock_vector_store, config)
        
        assert retriever.top_k == 5
        assert retriever.embedding_service == mock_embedding_service
        assert retriever.vector_store == mock_vector_store
    
    def test_hybrid_retriever(self):
        """Test hybrid retriever initialization."""
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        config = {'top_k': 5, 'dense_weight': 0.7}
        retriever = HybridRetriever(config)
        
        assert retriever.top_k == 5
        assert 0 <= retriever.dense_weight <= 1


class TestGeneration:
    """Tests for generation module."""
    
    def test_guardrails(self):
        """Test guardrails."""
        from src.generation.guardrails import Guardrails
        
        config = {'confidence_threshold': 0.5, 'max_hallucination_score': 0.3}
        guardrails = Guardrails(config)
        
        assert guardrails.confidence_threshold == 0.5
        assert guardrails.max_hallucination_score == 0.3


class TestAPI:
    """Tests for API endpoints."""
    
    def test_query_request(self):
        """Test query request model."""
        from src.api.models import QueryRequest
        
        request = QueryRequest(
            query="What is your return policy?",
            top_k=5,
            include_citations=True
        )
        
        assert request.query == "What is your return policy?"
        assert request.top_k == 5
    
    def test_query_response(self):
        """Test query response model."""
        from src.api.models import QueryResponse
        
        response = QueryResponse(
            query="What is your return policy?",
            answer="We offer a 30-day return policy.",
            citations=[],
            context=[],
            metadata={}
        )
        
        assert response.answer == "We offer a 30-day return policy."
