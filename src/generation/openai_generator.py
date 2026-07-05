"""OpenAI generator implementation."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OpenAIGenerator:
    """OpenAI generator implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model_name', 'gpt-4o')
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 1000)
        self.client = None
        
    async def _get_client(self):
        """Get OpenAI client.
        
        Returns:
            OpenAI client instance
        """
        if self.client is None:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        return self.client
    
    async def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate answer from query and context.
        
        Args:
            query: User query
            context: Retrieved context documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer with metadata
        """
        client = await self._get_client()
        
        context_text = self._format_context(context)
        
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant that answers questions based solely on the provided context. "
                "Always cite your sources using [Source X] format where X is the document rank. "
                "If the answer is not in the context, say 'I don't have information on that'."
            )
        
        user_prompt = f"""Question: {query}

Context:
{context_text}

Please answer the question using only the information from the context above. Include citations in your answer."""

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'model': self.model_name,
                'context': context,
                'citations': self._extract_citations(answer),
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    async def generate_stream(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ):
        """Generate answer with streaming.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Yields:
            Streaming response chunks
        """
        client = await self._get_client()
        
        context_text = self._format_context(context)
        
        system_prompt = (
            "You are a helpful assistant that answers questions based solely on the provided context. "
            "Always cite your sources using [Source X] format where X is the document rank. "
            "If the answer is not in the context, say 'I don't have information on that'."
        )
        
        user_prompt = f"""Question: {query}

Context:
{context_text}

Please answer the question using only the information from the context above. Include citations in your answer."""

        try:
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            raise
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for prompt.
        
        Args:
            context: List of context documents
            
        Returns:
            Formatted context string
        """
        formatted = []
        
        for i, doc in enumerate(context, 1):
            text = doc.get('text', '')
            source = doc.get('metadata', {}).get('source', 'unknown')
            metadata = doc.get('metadata', {})
            
            doc_info = f"[Source {i}] (source: {source})"
            if metadata.get('file_name'):
                doc_info += f" - {metadata['file_name']}"
            if metadata.get('page'):
                doc_info += f" - Page {metadata['page']}"
            
            formatted.append(f"{doc_info}\n{text}\n")
        
        return "\n\n".join(formatted)
    
    def _extract_citations(self, answer: str) -> List[Dict[str, Any]]:
        """Extract citations from answer.
        
        Args:
            answer: Generated answer
            
        Returns:
            List of citations
        """
        import re
        
        citations = []
        pattern = r'\[Source\s+(\d+)\]'
        matches = re.findall(pattern, answer)
        
        for match in set(matches):
            citations.append({
                'source_id': int(match),
                'reference': f'[Source {match}]'
            })
        
        return citations
