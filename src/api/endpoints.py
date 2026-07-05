"""FastAPI endpoints for RAG service."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid RAG API",
    description="Retrieval-Augmented Generation API with hybrid search",
    version="1.0.0"
)

rag_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup."""
    global rag_service
    from src.api.rag_service import RAGService
    
    config = {
        'openai_api_key': 'OPENAI_API_KEY_PLACEHOLDER',
        'openai_model': 'gpt-4o',
        'openai_embedding_model': 'text-embedding-3-large',
        'vector_store_type': 'chroma',
        'chroma_persist_dir': './data/chroma',
        'chroma_collection_name': 'hybrid_rag_docs',
        'embedding_dimension': 3072,
        'chunk_size': 512,
        'chunk_overlap': 50,
        'retrieval_top_k': 5,
        'retrieval_score_threshold': 0.7,
        'hybrid_retrieval_weight': 0.7,
        'temperature': 0.2,
        'max_tokens': 1000,
        'cache_enabled': True,
        'cache_ttl': 3600,
        'redis_url': 'redis://localhost:6379',
        'log_level': 'INFO',
        'log_file': './logs/app.log',
        'security_enabled': True,
        'pii_redaction': True,
        'access_control_enabled': True,
        'evaluation_enabled': True,
        'evaluation_metrics': ['faithfulness', 'relevance', 'hallucination_rate']
    }
    
    rag_service = RAGService(config)
    await rag_service.initialize()
    
    logger.info("RAG service started")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await rag_service.health_check()


@app.post("/query")
async def query(request: Request, data: Dict[str, Any]):
    """Query endpoint."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        query = data.get('query', '')
        metadata_filter = data.get('metadata_filter')
        top_k = data.get('top_k', 5)
        
        result = await rag_service.query(query, metadata_filter, top_k)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(data: Dict[str, Any]):
    """Feedback endpoint."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        query = data.get('query', '')
        answer = data.get('answer', '')
        score = data.get('score', 0)
        comment = data.get('comment', '')
        trace_id = data.get('trace_id', '')
        
        result = await rag_service.submit_feedback(query, answer, score, comment, trace_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """Get service status."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await rag_service.health_check()


@app.get("/metrics")
async def metrics():
    """Get service metrics."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        'vector_store_count': await rag_service.vector_store.get_count(),
        'cache_enabled': rag_service.cache is not None,
        'tracer_enabled': rag_service.tracer.enabled
    }
