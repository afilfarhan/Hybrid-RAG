"""
Hybrid RAG - API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from src.core.config import settings

router = APIRouter()

# Global services (initialized in main.py)
embedding_service = None
vector_store = None
retriever = None
generator = None


def set_services(e, v, r, g):
    """Set global services from main.py."""
    global embedding_service, vector_store, retriever, generator
    embedding_service = e
    vector_store = v
    retriever = r
    generator = g


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
    rating: int
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response model for feedback endpoint."""
    status: str


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


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query and return an answer."""
    global embedding_service, vector_store, retriever, generator
    
    if not all([embedding_service, vector_store, retriever, generator]):
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please wait for startup to complete."
        )
    
    try:
        # Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(
            query=request.query,
            filter=request.filter
        )
        
        # Generate answer
        generation_result = generator.generate(
            query=request.query,
            contexts=retrieved_chunks,
            max_tokens=500,
        )
        
        return QueryResponse(
            answer=generation_result["answer"],
            citations=generation_result["citations"],
            confidence=generation_result["confidence"],
            metadata=generation_result["metadata"],
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Submit feedback on a query result."""
    return FeedbackResponse(status="received")


@router.get("/status")
def status_endpoint():
    """Get system status."""
    return {
        "status": "healthy",
        "embedding_service": embedding_service is not None,
        "vector_store": vector_store is not None,
        "retriever": retriever is not None,
        "generator": generator is not None,
    }
