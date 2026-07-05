"""Guardrails for ensuring answer quality and safety."""

from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class Guardrails:
    """Guardrails for ensuring answer quality and safety."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize guardrails.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.max_hallucination_score = config.get('max_hallucination_score', 0.3)
        self.enable_fallback = config.get('enable_fallback', True)
        
    async def check_confidence(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Check retrieval confidence.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Returns:
            Confidence score
        """
        if not context:
            return 0.0
        
        scores = [doc.get('score', 0) for doc in context]
        avg_score = sum(scores) / len(scores)
        
        return avg_score
    
    async def check_hallucination(
        self,
        answer: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Check for hallucination in answer.
        
        Args:
            answer: Generated answer
            context: Retrieved context documents
            
        Returns:
            Hallucination score (0 = no hallucination, 1 = full hallucination)
        """
        if not context:
            return 1.0
        
        context_text = " ".join(doc.get('text', '') for doc in context)
        context_words = set(context_text.lower().split())
        answer_words = set(answer.lower().split())
        
        answer_only_words = answer_words - context_words
        
        if not answer_words:
            return 0.0
        
        hallucination_ratio = len(answer_only_words) / len(answer_words)
        
        return hallucination_ratio
    
    async def check_safety(
        self,
        answer: str,
        query: str
    ) -> Dict[str, Any]:
        """Check answer for safety issues.
        
        Args:
            answer: Generated answer
            query: User query
            
        Returns:
            Safety check results
        """
        safety_issues = []
        
        sensitive_patterns = [
            r'\b(pii|personal information)\b',
            r'\b(password|secret)\b',
            r'\b(credit card|card number)\b',
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                safety_issues.append('potential_sensitive_data')
                break
        
        return {
            'is_safe': len(safety_issues) == 0,
            'issues': safety_issues
        }
    
    async def apply_guardrails(
        self,
        query: str,
        context: List[Dict[str, Any]],
        answer: str
    ) -> Dict[str, Any]:
        """Apply all guardrails to answer.
        
        Args:
            query: User query
            context: Retrieved context documents
            answer: Generated answer
            
        Returns:
            Guardrails result
        """
        confidence = await self.check_confidence(query, context)
        hallucination = await self.check_hallucination(answer, context)
        safety = await self.check_safety(answer, query)
        
        should_fallback = (
            confidence < self.confidence_threshold or
            hallucination > self.max_hallucination_score
        )
        
        if should_fallback and self.enable_fallback:
            answer = self._get_fallback_answer()
        
        return {
            'answer': answer,
            'confidence': confidence,
            'hallucination_score': hallucination,
            'safety': safety,
            'should_fallback': should_fallback,
            'applied_guardrails': True
        }
    
    def _get_fallback_answer(self) -> str:
        """Get fallback answer when guardrails fail.
        
        Returns:
            Fallback answer
        """
        return "I don't have sufficient information to answer this question accurately. Please try rephrasing or ask about something else."
