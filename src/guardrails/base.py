"""
Hybrid RAG - Guardrails module
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.generation.base import GenerationResult


class GuardrailResult:
    """Represents a guardrail check result."""

    def __init__(
        self,
        passed: bool,
        reason: str,
        action: str = "allow",  # allow, modify, reject
        modified_result: Optional[GenerationResult] = None,
    ):
        self.passed = passed
        self.reason = reason
        self.action = action
        self.modified_result = modified_result


class BaseGuardrail(ABC):
    """Base class for guardrails."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def check(self, result: GenerationResult, query: str) -> GuardrailResult:
        """Check a generation result against guardrail rules."""
        pass


class HallucinationGuardrail(BaseGuardrail):
    """Guardrail to detect hallucinations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)

    async def check(self, result: GenerationResult, query: str) -> GuardrailResult:
        """Check for hallucinations based on confidence score."""
        if result.confidence < self.confidence_threshold:
            return GuardrailResult(
                passed=False,
                reason=f"Confidence score {result.confidence:.2f} below threshold {self.confidence_threshold}",
                action="reject",
            )
        return GuardrailResult(passed=True, reason="Confidence score acceptable", action="allow")


class LengthGuardrail(BaseGuardrail):
    """Guardrail to limit answer length."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_length = config.get("max_length", 2000)

    async def check(self, result: GenerationResult, query: str) -> GuardrailResult:
        """Check answer length."""
        if len(result.answer) > self.max_length:
            # Truncate answer
            truncated_answer = result.answer[: self.max_length] + "\n\n[Answer truncated]"
            modified = GenerationResult(
                answer=truncated_answer,
                citations=result.citations,
                confidence=result.confidence,
                metadata=result.metadata,
            )
            return GuardrailResult(
                passed=True,
                reason="Answer truncated to max length",
                action="modify",
                modified_result=modified,
            )
        return GuardrailResult(passed=True, reason="Answer length acceptable", action="allow")


class GuardrailPipeline:
    """Pipeline of guardrails."""

    def __init__(self, guardrails: List[BaseGuardrail]):
        self.guardrails = guardrails

    async def check(self, result: GenerationResult, query: str) -> GuardrailResult:
        """Run all guardrails in sequence."""
        current_result = result

        for guardrail in self.guardrails:
            check_result = await guardrail.check(current_result, query)

            if check_result.action == "reject":
                return check_result

            if check_result.action == "modify" and check_result.modified_result:
                current_result = check_result.modified_result

        return GuardrailResult(
            passed=True,
            reason="All guardrails passed",
            action="allow",
            modified_result=current_result,
        )
