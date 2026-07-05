"""RAG service for API endpoints."""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for API endpoints."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RAG service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.embedding_service = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self.guardrails = None
        self.cache = None
        self.tracer = None
        self.feedback_handler = None
        
    async def initialize(self):
        """Initialize all components."""
        from src.embedding.openai_service import OpenAIEmbeddingService
        from src.embedding.vector_store import VectorStore
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.generation.openai_generator import OpenAIGenerator
        from src.generation.guardrails import Guardrails
        from src.cache.redis_cache import RedisCache
        from src.tracing.langsmith_tracer import LangSmithTracer
        from src.admin.feedback_handler import FeedbackHandler
        
        self.embedding_service = OpenAIEmbeddingService(self.config)
        self.vector_store = VectorStore(self.config)
        await self.vector_store.connect()
        
        self.retriever = HybridRetriever({
            **self.config,
            'embedding_service': self.embedding_service,
            'vector_store': self.vector_store
        })
        
        self.generator = OpenAIGenerator(self.config)
        self.guardrails = Guardrails(self.config)
        self.tracer = LangSmithTracer(self.config)
        await self.tracer.connect()
        self.feedback_handler = FeedbackHandler(self.config)
        
        if self.config.get('cache_enabled', True):
            self.cache = RedisCache(self.config)
            await self.cache.connect()
        
        logger.info("RAG service initialized")
    
    async def query(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Process a query through the RAG pipeline.
        
        Args:
            query: User query
            metadata_filter: Optional metadata filter
            top_k: Number of documents to retrieve
            
        Returns:
            Query results
        """
        cache_key = f"query:{query}:{str(metadata_filter)}:{top_k}"
        
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info("Cache hit for query")
                return cached
        
        steps = {}
        
        steps['retrieval'] = {'input': {'query': query}}
        if metadata_filter:
            context = await self.retriever.retrieve_with_filter(query, metadata_filter)
        else:
            context = await self.retriever.retrieve(query)
        steps['retrieval']['output'] = {'context_count': len(context)}
        
        steps['generation'] = {'input': {'query': query, 'context_count': len(context)}}
        result = await self.generator.generate(query, context)
        steps['generation']['output'] = {'answer_length': len(result['answer'])}
        
        steps['guardrails'] = {'input': {'answer': result['answer']}}
        guarded = await self.guardrails.apply_guardrails(query, context, result['answer'])
        steps['guardrails']['output'] = {'should_fallback': guarded['should_fallback']}
        
        trace_id = await self.tracer.trace_query(query, steps)
        
        response = {
            'query': query,
            'answer': guarded['answer'],
            'citations': result['citations'],
            'context': context,
            'metadata': {
                'trace_id': trace_id,
                'confidence': guarded['confidence'],
                'hallucination_score': guarded['hallucination_score'],
                'usage': result['usage']
            }
        }
        
        if self.cache:
            await self.cache.set(cache_key, response)
        
        return response
    
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
        result = await self.feedback_handler.submit_feedback(
            query, answer, score, comment, trace_id
        )
        
        if self.tracer.enabled and trace_id:
            await self.tracer.trace_feedback(trace_id, {
                'type': 'user',
                'score': score,
                'comment': comment
            })
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check results
        """
        components = {}
        
        components['vector_store'] = await self.vector_store.get_status()
        components['embedding_service'] = {'status': 'healthy'}
        components['cache'] = {'status': 'healthy' if self.cache else 'disabled'}
        components['tracer'] = {'status': 'healthy' if self.tracer.enabled else 'disabled'}
        
        return {
            'status': 'healthy' if all(
                c.get('status') == 'healthy' or c.get('status') == 'disabled'
                for c in components.values()
            ) else 'unhealthy',
            'components': components
        }
