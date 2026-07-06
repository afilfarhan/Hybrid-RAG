"""
Hybrid RAG - Main entry point with real services
"""

import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.endpoints import router as api_router
from src.core.config import settings
from src.embedding.openai_service import OpenAIEmbeddingService
from src.embedding.chroma_store import ChromaDBVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.openai_generator import OpenAIGenerator
from src.services.embedding_inmemory import InMemoryEmbeddingService
from src.services.vector_store_inmemory import InMemoryVectorStore
from src.services.retriever_inmemory import InMemoryRetriever
from src.services.generator_inmemory import MockGeneratorService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Global service instances
embedding_service: OpenAIEmbeddingService = None
vector_store: ChromaDBVectorStore = None
retriever: HybridRetriever = None
generator: OpenAIGenerator = None

using_real_services = False


async def startup_event():
    """Initialize resources on startup."""
    global embedding_service, vector_store, retriever, generator, using_real_services
    
    logger.info("Starting Hybrid RAG system...")
    logger.info(f"Configuration loaded: {settings.model_dump_json()}")
    
    # Try to initialize real services first
    if settings.openai_api_key:
        try:
            # Initialize embedding service
            embedding_service = OpenAIEmbeddingService({
                "api_key": settings.openai_api_key,
                "model_name": settings.openai_embedding_model or "text-embedding-3-small",
            })
            await embedding_service.get_dimension()
            logger.info(f"Embedding service initialized with model: {settings.openai_embedding_model}")
            
            # Initialize vector store
            vector_store = ChromaDBVectorStore({
                "collection_name": "hybrid_rag",
                "persist_path": "./data/vector_store",
            })
            await vector_store.connect()
            logger.info("Vector store connected")
            
            # Initialize hybrid retriever
            retriever = HybridRetriever({
                "embedding_service": embedding_service,
                "vector_store": vector_store,
                "top_k": settings.retrieval_top_k,
                "dense_weight": settings.retrieval_dense_weight,
                "sparse_weight": settings.retrieval_sparse_weight,
                "score_threshold": 0.5,
            })
            logger.info("Hybrid retriever initialized")
            
            # Initialize generator
            generator = OpenAIGenerator({
                "api_key": settings.openai_api_key,
                "model_name": settings.openai_model or "gpt-4o-mini",
            })
            logger.info(f"Generator initialized with model: {settings.openai_model}")
            
            using_real_services = True
            logger.info("Real services initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize real services: {e}. Falling back to in-memory services.")
            using_real_services = False
    
    # Fall back to in-memory services
    if not using_real_services:
        logger.info("Initializing in-memory services (fallback mode)")
        embedding_service = InMemoryEmbeddingService(dimension=384)
        vector_store = InMemoryVectorStore(dimension=384)
        retriever = InMemoryRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            top_k=settings.retrieval_top_k,
            dense_weight=settings.retrieval_dense_weight,
            sparse_weight=settings.retrieval_sparse_weight,
        )
        generator = MockGeneratorService()
        logger.info("In-memory services initialized successfully")
    
    # Set services in endpoints module
    from src.api.endpoints import set_services
    set_services(embedding_service, vector_store, retriever, generator)
    
    # Add sample data for testing (only in in-memory mode)
    if not using_real_services:
        try:
            from add_sample_data import add_sample_data_to_retriever
            add_sample_data_to_retriever(retriever)
            logger.info("Sample data added to vector store")
        except Exception as e:
            logger.warning(f"Could not add sample data: {e}")


async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down Hybrid RAG system...")
    
    if vector_store and using_real_services:
        await vector_store.disconnect()
        logger.info("Vector store disconnected")


async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    await startup_event()
    yield
    await shutdown_event()


app = FastAPI(
    title="Hybrid RAG API",
    description="A retrieval-augmented generation system with citations and low hallucination",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "mode": "real" if using_real_services else "in-memory"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Hybrid RAG",
        "version": "0.1.0",
        "description": "A retrieval-augmented generation system with citations and low hallucination",
        "mode": "real" if using_real_services else "in-memory",
    }


app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
