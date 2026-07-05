"""Feedback handler for user feedback."""

from typing import Dict, Any, List
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackHandler:
    """Feedback handler for user feedback."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feedback handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feedback_dir = Path(config.get('feedback_dir', './data/feedback'))
        self.feedback_file = self.feedback_dir / 'feedback.json'
        
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.feedback_file.exists():
            self._save_feedback([])
    
    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Load feedback from file.
        
        Returns:
            List of feedback entries
        """
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback(self, feedback: List[Dict[str, Any]]):
        """Save feedback to file.
        
        Args:
            feedback: List of feedback entries
        """
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)
    
    async def submit_feedback(
        self,
        query: str,
        answer: str,
        score: int,
        comment: str = '',
        trace_id: str = ''
    ) -> Dict[str, Any]:
        """Submit user feedback.
        
        Args:
            query: User query
            answer: Generated answer
            score: Feedback score (1-5)
            comment: Optional comment
            trace_id: Optional trace ID
            
        Returns:
            Feedback confirmation
        """
        feedback_entry = {
            'query': query,
            'answer': answer,
            'score': score,
            'comment': comment,
            'trace_id': trace_id,
            'timestamp': str(__import__('datetime').datetime.now())
        }
        
        feedback = self._load_feedback()
        feedback.append(feedback_entry)
        self._save_feedback(feedback)
        
        logger.info(f"Received feedback: score={score}, query={query[:50]}...")
        
        return {
            'status': 'success',
            'feedback_id': len(feedback) - 1
        }
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics.
        
        Returns:
            Feedback statistics
        """
        feedback = self._load_feedback()
        
        if not feedback:
            return {
                'total': 0,
                'avg_score': 0,
                'positive': 0,
                'negative': 0
            }
        
        scores = [f['score'] for f in feedback]
        avg_score = sum(scores) / len(scores)
        
        positive = sum(1 for s in scores if s >= 4)
        negative = sum(1 for s in scores if s <= 2)
        
        return {
            'total': len(feedback),
            'avg_score': round(avg_score, 2),
            'positive': positive,
            'negative': negative,
            'positive_rate': round(positive / len(feedback) * 100, 1)
        }
    
    async def get_feedback_list(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get feedback list.
        
        Args:
            limit: Maximum number of entries
            offset: Offset for pagination
            
        Returns:
            List of feedback entries
        """
        feedback = self._load_feedback()
        
        return feedback[offset:offset + limit]
