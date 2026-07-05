# Sample test for chunking module.

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_recursive_chunker():
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
    
    print("✓ Recursive chunker test passed")


def test_structure_chunker():
    """Test structure-aware chunker."""
    from src.chunking.structure_chunker import StructureAwareChunker
    
    config = {'chunk_size': 100, 'chunk_overlap': 10}
    chunker = StructureAwareChunker(config)
    
    text = "# Heading 1\n\nContent here.\n\n# Heading 2\n\nMore content."
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    print("✓ Structure chunker test passed")


def test_faq_chunker():
    """Test FAQ chunker."""
    from src.chunking.faq_chunker import FAQChunker
    
    config = {'chunk_size': 512}
    chunker = FAQChunker(config)
    
    text = "Q: What is your return policy?\nA: 30 days."
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    print("✓ FAQ chunker test passed")


def test_text_preprocessor():
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
    print("✓ Text preprocessor test passed")


def test_dense_retriever():
    """Test dense retriever initialization."""
    from src.retrieval.dense_retriever import DenseRetriever
    
    config = {'top_k': 5, 'score_threshold': 0.7}
    retriever = DenseRetriever(config)
    
    assert retriever.top_k == 5
    assert retriever.score_threshold == 0.7
    print("✓ Dense retriever test passed")


def test_hybrid_retriever():
    """Test hybrid retriever initialization."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    
    config = {'top_k': 5, 'dense_weight': 0.7}
    retriever = HybridRetriever(config)
    
    assert retriever.top_k == 5
    assert 0 <= retriever.dense_weight <= 1
    print("✓ Hybrid retriever test passed")


def test_guardrails():
    """Test guardrails."""
    from src.generation.guardrails import Guardrails
    
    config = {'confidence_threshold': 0.5, 'max_hallucination_score': 0.3}
    guardrails = Guardrails(config)
    
    assert guardrails.confidence_threshold == 0.5
    assert guardrails.max_hallucination_score == 0.3
    print("✓ Guardrails test passed")


def test_query_request():
    """Test query request model."""
    from src.api.models import QueryRequest
    
    request = QueryRequest(
        query="What is your return policy?",
        top_k=5,
        include_citations=True
    )
    
    assert request.query == "What is your return policy?"
    assert request.top_k == 5
    print("✓ Query request test passed")


def test_query_response():
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
    print("✓ Query response test passed")


if __name__ == "__main__":
    print("Running Hybrid RAG tests...\n")
    
    test_recursive_chunker()
    test_structure_chunker()
    test_faq_chunker()
    test_text_preprocessor()
    test_dense_retriever()
    test_hybrid_retriever()
    test_guardrails()
    test_query_request()
    test_query_response()
    
    print("\n✅ All tests passed!")
