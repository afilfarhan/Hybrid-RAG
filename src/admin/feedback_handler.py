"""
Hybrid RAG - Feedback handler
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


class Feedback:
    """Represents user feedback."""

    def __init__(
        self,
        query_id: str,
        rating: int,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.query_id = query_id
        self.rating = rating  # 1-5
        self.comment = comment
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat() + "Z"


class FeedbackHandler:
    """Handler for user feedback."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feedback_store: List[Feedback] = []

    async def submit_feedback(
        self,
        query_id: str,
        rating: int,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Submit feedback for a query."""
        if rating < 1 or rating > 5:
            return False

        feedback = Feedback(
            query_id=query_id,
            rating=rating,
            comment=comment,
            metadata=metadata or {},
        )

        self.feedback_store.append(feedback)
        return True

    async def get_feedback(
        self, query_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get feedback, optionally filtered by query ID."""
        if query_id:
            return [
                f.__dict__
                for f in self.feedback_store
                if f.query_id == query_id
            ]
        return [f.__dict__ for f in self.feedback_store]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        if not self.feedback_store:
            return {"total": 0, "avg_rating": 0.0}

        ratings = [f.rating for f in self.feedback_store]
        avg_rating = sum(ratings) / len(ratings)

        return {
            "total": len(self.feedback_store),
            "avg_rating": avg_rating,
            "rating_distribution": {
                str(i): ratings.count(i) for i in range(1, 6)
            },
        }
