"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str
    metadata_filter: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = 5
    include_citations: Optional[bool] = True
    stream: Optional[bool] = False


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    context: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    trace_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    
    query: str
    answer: str
    score: int
    comment: Optional[str] = ''
    trace_id: Optional[str] = ''


class IngestionRequest(BaseModel):
    """Request model for ingestion endpoint."""
    
    source_type: str
    source_config: Dict[str, Any]
    schedule: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint."""
    
    status: str
    source_id: str
    documents_ingested: int
    failed_documents: int


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str
    components: Dict[str, Dict[str, Any]]
