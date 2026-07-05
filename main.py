"""Main entry point for Hybrid RAG system."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import settings
from src.api.rag_service import RAGService


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Hybrid RAG System")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"Vector Store: {settings.vector_store_type}")
    logger.info(f"Chunk Size: {settings.chunk_size}")
    
    config = {
        'openai_api_key': settings.openai_api_key,
        'openai_model': settings.openai_model,
        'openai_embedding_model': settings.openai_embedding_model,
        'vector_store_type': settings.vector_store_type,
        'chroma_persist_dir': settings.chroma_persist_dir,
        'chroma_collection_name': settings.chroma_collection_name,
        'embedding_dimension': settings.embedding_dimension,
        'chunk_size': settings.chunk_size,
        'chunk_overlap': settings.chunk_overlap,
        'retrieval_top_k': settings.retrieval_top_k,
        'retrieval_score_threshold': settings.retrieval_score_threshold,
        'hybrid_retrieval_weight': settings.hybrid_retrieval_weight,
        'temperature': settings.temperature,
        'max_tokens': settings.max_tokens,
        'system_prompt': settings.system_prompt,
        'cache_enabled': settings.cache_enabled,
        'cache_ttl': settings.cache_ttl,
        'redis_url': settings.redis_url,
        'log_level': settings.log_level,
        'log_file': settings.log_file,
        'security_enabled': settings.security_enabled,
        'pii_redaction': settings.pii_redaction,
        'access_control_enabled': settings.access_control_enabled,
        'evaluation_enabled': settings.evaluation_enabled,
        'evaluation_metrics': settings.evaluation_metrics,
        'data_sources_dir': settings.data_sources_dir,
        'sample_docs_dir': settings.sample_docs_dir
    }
    
    try:
        rag_service = RAGService(config)
        await rag_service.initialize()
        
        logger.info("Hybrid RAG System initialized successfully")
        
        return rag_service
        
    except Exception as e:
        logger.error(f"Failed to initialize Hybrid RAG System: {e}")
        raise


if __name__ == "__main__":
    try:
        rag_service = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
