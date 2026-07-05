"""LangSmith tracing implementation."""

from typing import Dict, Any, Optional
import logging
import uuid

logger = logging.getLogger(__name__)


class LangSmithTracer:
    """LangSmith tracing implementation."""
    
    def __init__(self, config: dict):
        """Initialize LangSmith tracer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get('api_key', '')
        self.project_name = config.get('project_name', 'HybridRAG')
        self.enabled = config.get('enabled', True)
        self.client = None
        
    async def connect(self):
        """Connect to LangSmith."""
        if self.enabled:
            try:
                from langsmith import Client
                self.client = Client(api_key=self.api_key)
                logger.info("Connected to LangSmith")
            except ImportError:
                logger.warning("LangSmith not installed, using local tracing")
                self.enabled = False
        else:
            logger.info("LangSmith tracing disabled")
    
    async def trace_query(
        self,
        query: str,
        steps: Dict[str, Any]
    ) -> str:
        """Trace a query through the RAG pipeline.
        
        Args:
            query: User query
            steps: Pipeline steps with results
            
        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())
        
        if not self.enabled or not self.client:
            logger.debug(f"Tracing query: {query[:50]}... (ID: {trace_id})")
            return trace_id
        
        try:
            run = self.client.create_run(
                run_type="chain",
                name="HybridRAG Query",
                inputs={"query": query},
                trace_id=trace_id,
                project_name=self.project_name
            )
            
            for step_name, step_data in steps.items():
                self.client.create_run(
                    run_type="chain",
                    name=step_name,
                    inputs=step_data.get('input', {}),
                    outputs=step_data.get('output', {}),
                    parent_run_id=run.id,
                    trace_id=trace_id
                )
            
            self.client.update_run(
                run.id,
                outputs=steps.get('final_output', {}),
                status="success"
            )
            
            logger.info(f"Traced query: {query[:50]}... (ID: {trace_id})")
            
        except Exception as e:
            logger.error(f"Error tracing query: {e}")
        
        return trace_id
    
    async def trace_feedback(
        self,
        trace_id: str,
        feedback: Dict[str, Any]
    ) -> bool:
        """Record feedback for a traced query.
        
        Args:
            trace_id: Trace ID
            feedback: Feedback data
            
        Returns:
            True if successful
        """
        if not self.enabled or not self.client:
            logger.debug(f"Recording feedback for trace: {trace_id}")
            return True
        
        try:
            self.client.create_feedback(
                trace_id=trace_id,
                key=feedback.get('type', 'user'),
                score=feedback.get('score', 1),
                comment=feedback.get('comment', '')
            )
            
            logger.info(f"Recorded feedback for trace: {trace_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a traced query by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace data or None
        """
        if not self.enabled or not self.client:
            return {'trace_id': trace_id, 'local': True}
        
        try:
            runs = self.client.list_runs(trace_id=trace_id)
            runs_list = list(runs)
            
            return {
                'trace_id': trace_id,
                'runs': runs_list
            }
            
        except Exception as e:
            logger.error(f"Error getting trace: {e}")
            return None
