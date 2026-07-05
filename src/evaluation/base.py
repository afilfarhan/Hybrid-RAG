"""Base class for evaluators."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get('name', 'evaluator')
        
    @abstractmethod
    async def evaluate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        answer: str
    ) -> Dict[str, Any]:
        """Evaluate a RAG response.
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Evaluation results
        """
        pass
    
    @abstractmethod
    async def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate multiple test cases.
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            Batch evaluation results
        """
        pass
