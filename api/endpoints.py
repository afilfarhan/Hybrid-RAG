"""
Hybrid RAG - API Layer

REST API endpoints for the RAG system.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from services.base import RAGService

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    query_id: str
    rating: int
    comment: Optional[str] = None


# Global RAG service instance
rag_service: Optional[RAGService] = None


def set_rag_service(service: RAGService):
    """Set the RAG service instance."""
    global rag_service
    rag_service = service


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query through the RAG system."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    try:
        result = rag_service.query(request.query, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Submit feedback for a query."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    # In a real system, this would store feedback in a database
    return {"status": "feedback received", "query_id": request.query_id}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if rag_service else "initializing",
        "service": "Hybrid RAG API"
    }


@router.get("/docs")
async def documentation():
    """API documentation."""
    return {
        "endpoints": {
            "/query": "POST - Process a query",
            "/feedback": "POST - Submit feedback",
            "/health": "GET - Health check"
        }
    }
