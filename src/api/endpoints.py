"""
Hybrid RAG - API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@router.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "Hybrid RAG",
        "version": "0.1.0",
        "description": "Hybrid RAG system with Arabic/English support",
    }


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    top_k: int = 5
    filter: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    query_id: str
    rating: int  # 1-5
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response model for feedback endpoint."""
    status: str


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query and return an answer."""
    return QueryResponse(
        answer="System not fully initialized - please configure embedding service, vector store, and generator.",
        citations=[],
        confidence=0.0,
        metadata={"status": "not_initialized"},
    )


@router.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Submit feedback on a query result."""
    return FeedbackResponse(status="received")


@router.get("/status")
def status_endpoint():
    """Get system status."""
    return {
        "status": "healthy",
        "embedding_service": False,
        "vector_store": False,
        "retriever": False,
        "generator": False,
        "guardrails": False,
    }
