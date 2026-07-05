"""
Hybrid RAG - Evaluation module
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.generation.base import GenerationResult
from src.retrieval.base import RetrievedChunk


class EvaluationResult:
    """Represents an evaluation result."""

    def __init__(
        self,
        metric: str,
        score: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.metric = metric
        self.score = score
        self.details = details or {}


class BaseEvaluator(ABC):
    """Base class for evaluators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        result: GenerationResult,
    ) -> EvaluationResult:
        """Evaluate a query result."""
        pass


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluator for faithfulness (groundedness)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    async def evaluate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        result: GenerationResult,
    ) -> EvaluationResult:
        """Evaluate faithfulness based on citations and confidence."""
        if not result.citations:
            return EvaluationResult(
                metric="faithfulness",
                score=0.0,
                details={"reason": "No citations found"},
            )

        # Simple heuristic: score based on citation coverage and confidence
        citation_score = min(len(result.citations) / 3, 1.0)  # Max score with 3+ citations
        confidence_score = result.confidence

        final_score = (citation_score + confidence_score) / 2

        return EvaluationResult(
            metric="faithfulness",
            score=final_score,
            details={
                "num_citations": len(result.citations),
                "confidence": result.confidence,
                "citation_score": citation_score,
            },
        )


class RelevanceEvaluator(BaseEvaluator):
    """Evaluator for relevance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    async def evaluate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        result: GenerationResult,
    ) -> EvaluationResult:
        """Evaluate relevance based on retrieval scores."""
        if not retrieved_chunks:
            return EvaluationResult(
                metric="relevance",
                score=0.0,
                details={"reason": "No chunks retrieved"},
            )

        avg_score = sum(chunk.score for chunk in retrieved_chunks) / len(retrieved_chunks)

        return EvaluationResult(
            metric="relevance",
            score=avg_score,
            details={
                "num_chunks": len(retrieved_chunks),
                "avg_score": avg_score,
            },
        )


class HallucinationEvaluator(BaseEvaluator):
    """Evaluator for hallucinations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)

    async def evaluate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        result: GenerationResult,
    ) -> EvaluationResult:
        """Evaluate hallucination risk based on confidence score."""
        if result.confidence < self.confidence_threshold:
            return EvaluationResult(
                metric="hallucination_risk",
                score=1.0 - result.confidence,  # Higher score = higher risk
                details={
                    "confidence": result.confidence,
                    "risk": "high",
                },
            )

        return EvaluationResult(
            metric="hallucination_risk",
            score=0.0,
            details={
                "confidence": result.confidence,
                "risk": "low",
            },
        )
