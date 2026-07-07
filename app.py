import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from services.chromadb_store import ChromaDBVectorStore
from services.inmemory import (
    InMemoryEmbeddingService,
    InMemoryRetrievalService,
    InMemoryGenerationService
)
from services.base import RAGService
from api.endpoints import router as api_router, set_rag_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_services():
    """Initialize all RAG services."""
    logger.info("Initializing RAG services...")
    
    # Create services
    embedding_service = InMemoryEmbeddingService()
    vector_store = ChromaDBVectorStore(
        persist_path="./data/vector_store",
        collection_name="documents",
        dimension=384
    )
    retrieval_service = InMemoryRetrievalService(vector_store, top_k=5)
    generation_service = InMemoryGenerationService()
    
    # Create RAG service
    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        retrieval_service=retrieval_service,
        generation_service=generation_service
    )
    
    # Set in API layer
    set_rag_service(rag_service)
    
    logger.info("RAG services initialized successfully")
    return rag_service


def load_sample_documents(rag_service):
    """Load sample documents into the vector store."""
    # Skip loading sample docs if vector store already has documents
    stats = rag_service.vector_store.get_stats()
    if stats['count'] > 0:
        logger.info(f"Vector store already has {stats['count']} documents. Skipping sample document loading.")
        return
    
    sample_docs = [
        {
            "text": "We offer a 30-day return policy for all products. Items must be in their original condition with all packaging and accessories included.",
            "metadata": {"source": "returns_policy", "type": "faq"}
        },
        {
            "text": "To return an item, contact our support team within 30 days of purchase to request a return authorization. We will provide you with a return label and instructions.",
            "metadata": {"source": "returns_policy", "type": "faq"}
        },
        {
            "text": "Refunds are processed within 5-7 business days after we receive your returned item.",
            "metadata": {"source": "returns_policy", "type": "faq"}
        },
        {
            "text": "Standard shipping takes 3-5 business days. Free for orders over $50, $5.99 for orders under $50.",
            "metadata": {"source": "shipping_info", "type": "faq"}
        },
        {
            "text": "Express shipping takes 1-2 business days and costs $14.99 with tracking available online.",
            "metadata": {"source": "shipping_info", "type": "faq"}
        },
        {
            "text": "We ship internationally to over 50 countries with delivery time of 7-14 business days.",
            "metadata": {"source": "shipping_info", "type": "faq"}
        },
        {
            "text": "Premium Wireless Headphones cost $299.99 with active noise cancellation and 30-hour battery life.",
            "metadata": {"source": "product_catalog", "type": "product"}
        },
        {
            "text": "Smart Watch Pro costs $399.99 with heart rate monitoring, GPS tracking, and 7-day battery life.",
            "metadata": {"source": "product_catalog", "type": "product"}
        },
        {
            "text": "Wireless Earbuds cost $149.99 with active noise cancellation and 24-hour battery life.",
            "metadata": {"source": "product_catalog", "type": "product"}
        },
        {
            "text": "We accept credit cards, PayPal, Apple Pay, and Google Pay for payments.",
            "metadata": {"source": "faq", "type": "payment"}
        },
        {
            "text": "All our products come with a 1-year manufacturer warranty covering defects in materials and workmanship.",
            "metadata": {"source": "faq", "type": "warranty"}
        },
    ]
    
    logger.info(f"Loading {len(sample_docs)} sample documents...")
    for doc in sample_docs:
        rag_service.ingest_document(doc["text"], doc["metadata"])
    
    logger.info(f"Loaded {len(sample_docs)} sample documents")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    # Startup
    rag_service = initialize_services()
    load_sample_documents(rag_service)
    yield
    # Shutdown
    logger.info("Shutting down Hybrid RAG system...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Hybrid RAG API",
    description="A retrieval-augmented generation system with citations and low hallucination",
    version="1.0.0",
    lifespan=lifespan,
)


# Include API router
app.include_router(api_router, prefix="/api/v1")


# Serve frontend UI
@app.get("/")
async def serve_frontend():
    """Serve the frontend UI."""
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"message": "Hybrid RAG API is running. Visit /ui for the interface."}


@app.get("/ui")
async def serve_ui_index():
    """Serve the frontend UI index."""
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"message": "UI not found"}


# Mount UI directory
ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")


def get_app():
    """Get the FastAPI app for testing."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
