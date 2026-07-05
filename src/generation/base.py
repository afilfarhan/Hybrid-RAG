"""
Hybrid RAG - Generation module
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.retrieval.base import RetrievedChunk


class GenerationResult:
    """Represents a generation result."""

    def __init__(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.answer = answer
        self.citations = citations
        self.confidence = confidence
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"GenerationResult(answer_length={len(self.answer)}, confidence={self.confidence:.2f})"


class BaseGenerator(ABC):
    """Base class for generators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> GenerationResult:
        """Generate an answer from retrieved chunks."""
        pass
