"""Golden test set for evaluation."""

from typing import Dict, Any, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class GoldenTestSet:
    """Golden test set for evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize golden test set.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.test_set_path = Path(config.get('test_set_path', './data/eval/golden_test_set.json'))
        self.test_cases = []
        
        self._load_test_set()
    
    def _load_test_set(self):
        """Load test set from file."""
        if self.test_set_path.exists():
            with open(self.test_set_path, 'r') as f:
                self.test_cases = json.load(f)
            logger.info(f"Loaded {len(self.test_cases)} test cases")
        else:
            logger.warning(f"Test set file not found: {self.test_set_path}")
    
    def add_test_case(
        self,
        query: str,
        expected_answer: str,
        expected_context_sources: List[str] = None
    ):
        """Add a test case.
        
        Args:
            query: User query
            expected_answer: Expected answer
            expected_context_sources: Expected context sources
        """
        test_case = {
            'query': query,
            'expected_answer': expected_answer,
            'expected_context_sources': expected_context_sources or [],
            'created_at': str(__import__('datetime').datetime.now())
        }
        
        self.test_cases.append(test_case)
        self._save_test_set()
    
    def remove_test_case(self, index: int) -> bool:
        """Remove a test case.
        
        Args:
            index: Test case index
            
        Returns:
            True if successful
        """
        if 0 <= index < len(self.test_cases):
            self.test_cases.pop(index)
            self._save_test_set()
            return True
        return False
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get all test cases.
        
        Returns:
            List of test cases
        """
        return self.test_cases
    
    def _save_test_set(self):
        """Save test set to file."""
        self.test_set_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.test_set_path, 'w') as f:
            json.dump(self.test_cases, f, indent=2)
