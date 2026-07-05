"""
Tests for Hybrid RAG evaluation
"""

import pytest
from src.generation.base import GenerationResult
from src.retrieval.base import RetrievedChunk
from src.evaluation.evaluators import (
    FaithfulnessEvaluator,
    RelevanceEvaluator,
    HallucinationEvaluator,
)


@pytest.mark.asyncio
async def test_faithfulness_evaluator_with_citations():
    """Test faithfulness evaluation with citations."""
    chunks = [
        RetrievedChunk(
            content="Test content",
            metadata={"chunk_id": "chunk_1"},
            score=0.9,
            source="test.txt",
        )
    ]
    
    result = GenerationResult(
        answer="This is a test answer.",
        citations=[{"id": 1, "source": "test.txt"}],
        confidence=0.8,
    )
    
    evaluator = FaithfulnessEvaluator()
    evaluation = await evaluator.evaluate("test query", chunks, result)
    
    assert evaluation.metric == "faithfulness"
    assert evaluation.score > 0


@pytest.mark.asyncio
async def test_faithfulness_evaluator_no_citations():
    """Test faithfulness evaluation without citations."""
    chunks = [
        RetrievedChunk(
            content="Test content",
            metadata={"chunk_id": "chunk_1"},
            score=0.9,
            source="test.txt",
        )
    ]
    
    result = GenerationResult(
        answer="This is a test answer.",
        citations=[],
        confidence=0.8,
    )
    
    evaluator = FaithfulnessEvaluator()
    evaluation = await evaluator.evaluate("test query", chunks, result)
    
    assert evaluation.score == 0.0


@pytest.mark.asyncio
async def test_relevance_evaluator():
    """Test relevance evaluation."""
    chunks = [
        RetrievedChunk(
            content="Test content",
            metadata={"chunk_id": "chunk_1"},
            score=0.9,
            source="test.txt",
        ),
        RetrievedChunk(
            content="More content",
            metadata={"chunk_id": "chunk_2"},
            score=0.7,
            source="test2.txt",
        ),
    ]
    
    result = GenerationResult(
        answer="Test answer",
        citations=[],
        confidence=0.8,
    )
    
    evaluator = RelevanceEvaluator()
    evaluation = await evaluator.evaluate("test query", chunks, result)
    
    assert evaluation.metric == "relevance"
    assert evaluation.score == 0.8  # (0.9 + 0.7) / 2


@pytest.mark.asyncio
async def test_hallucination_evaluator_low_confidence():
    """Test hallucination evaluation with low confidence."""
    chunks = [
        RetrievedChunk(
            content="Test content",
            metadata={"chunk_id": "chunk_1"},
            score=0.9,
            source="test.txt",
        )
    ]
    
    result = GenerationResult(
        answer="Test answer",
        citations=[],
        confidence=0.3,
    )
    
    evaluator = HallucinationEvaluator()
    evaluation = await evaluator.evaluate("test query", chunks, result)
    
    assert evaluation.metric == "hallucination_risk"
    assert evaluation.score > 0.5  # High risk


@pytest.mark.asyncio
async def test_hallucination_evaluator_high_confidence():
    """Test hallucination evaluation with high confidence."""
    chunks = [
        RetrievedChunk(
            content="Test content",
            metadata={"chunk_id": "chunk_1"},
            score=0.9,
            source="test.txt",
        )
    ]
    
    result = GenerationResult(
        answer="Test answer",
        citations=[],
        confidence=0.9,
    )
    
    evaluator = HallucinationEvaluator()
    evaluation = await evaluator.evaluate("test query", chunks, result)
    
    assert evaluation.score == 0.0  # Low risk
