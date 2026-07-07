"""
LiteLLM Services for Production

These services use LiteLLM to support multiple LLM providers (OpenAI, NVIDIA NIM, Anthropic, Google, etc.)
"""

import os
import logging
from typing import List, Dict, Any, Optional

from .base import EmbeddingService, GenerationService

logger = logging.getLogger(__name__)


class LiteLMEmbeddingService(EmbeddingService):
    """Embedding service using LiteLLM with support for multiple providers."""
    
    def __init__(self, model_name: str = None, provider: str = None, api_key: str = None, base_url: str = None):
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm not installed. "
                "Install with: pip install litellm"
            )
        
        self.provider = provider or os.getenv("PROVIDER", "openai")
        
        # Handle model name - LiteLLM requires provider prefix like "nvidia_nim/meta/llama3-70b-instruct"
        raw_model = model_name or os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
        # Check if provider prefix is already present
        if "/" not in raw_model:
            # Add provider prefix
            self.model_name = f"{self.provider}/{raw_model}"
        else:
            # Model already has path, but may need provider prefix
            parts = raw_model.split("/")
            if len(parts) == 2 and parts[0] != self.provider:
                # Model has org/model format, add provider prefix
                self.model_name = f"{self.provider}/{raw_model}"
            else:
                self.model_name = raw_model
        
        # Set API key - use NVIDIA_NIM_API_KEY for NVIDIA NIM
        if api_key:
            self.litellm.api_key = api_key
        elif os.getenv("NVIDIA_NIM_API_KEY"):
            self.litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
        elif os.getenv("NVIDIA_API_KEY"):
            self.litellm.api_key = os.getenv("NVIDIA_API_KEY")
        elif os.getenv("OPENAI_API_KEY"):
            self.litellm.api_key = os.getenv("OPENAI_API_KEY")
        
        # Set base URL for NVIDIA NIM
        if self.provider == "nvidia_nim":
            self.litellm.api_base = os.getenv("NVIDIA_NIM_API_BASE", os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
        elif base_url:
            self.litellm.api_base = base_url
        
        self.dimension: Optional[int] = None
    
    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        try:
            response = self.litellm.embedding(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        try:
            response = self.litellm.embedding(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.dimension is None:
            test_text = "test"
            embedding = self.embed(test_text)
            self.dimension = len(embedding)
        return self.dimension


class LiteLMGenerationService(GenerationService):
    """Generation service using LiteLLM with support for multiple providers."""
    
    def __init__(self, model_name: str = None, provider: str = None, api_key: str = None, base_url: str = None):
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm not installed. "
                "Install with: pip install litellm"
            )
        
        self.provider = provider or os.getenv("PROVIDER", "openai")
        
        # Handle model name - LiteLLM requires provider prefix like "nvidia_nim/meta/llama3-70b-instruct"
        raw_model = model_name or os.getenv("NVIDIA_LLM_MODEL", "meta/llama3-70b-instruct")
        # Check if provider prefix is already present
        if "/" not in raw_model:
            # Add provider prefix
            self.model_name = f"{self.provider}/{raw_model}"
        else:
            # Model already has path, but may need provider prefix
            parts = raw_model.split("/")
            if len(parts) == 2 and parts[0] != self.provider:
                # Model has org/model format, add provider prefix
                self.model_name = f"{self.provider}/{raw_model}"
            else:
                self.model_name = raw_model
        
        # Set API key - use NVIDIA_NIM_API_KEY for NVIDIA NIM
        if api_key:
            self.litellm.api_key = api_key
        elif os.getenv("NVIDIA_NIM_API_KEY"):
            self.litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
        elif os.getenv("NVIDIA_API_KEY"):
            self.litellm.api_key = os.getenv("NVIDIA_API_KEY")
        elif os.getenv("OPENAI_API_KEY"):
            self.litellm.api_key = os.getenv("OPENAI_API_KEY")
        
        # Set base URL for NVIDIA NIM
        if self.provider == "nvidia_nim":
            self.litellm.api_base = os.getenv("NVIDIA_NIM_API_BASE", os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
        elif base_url:
            self.litellm.api_base = base_url
        
        self.system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Your answers must be grounded in the provided context and include citations to the source documents.

Instructions:
1. Answer the question using ONLY the information from the provided context
2. Include citations in the format [Source: doc_id] at the end of each claim
3. If the answer is not in the context, say "I don't have enough information to answer that question"
4. Be concise and focused on the question
5. Include the most relevant citations for each part of your answer"""
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate a response based on query and retrieved contexts."""
        if not contexts:
            return {
                "text": "I don't have enough information to answer this question.",
                "confidence": 0.0,
                "model": self.model_name,
                "citations": []
            }
        
        # Build context with metadata
        context_text = ""
        for i, ctx in enumerate(contexts):
            source_id = ctx.get("id", f"doc_{i}")
            content = ctx.get("text", "")
            score = ctx.get("similarity", 0)
            context_text += f"[Source: {source_id}] (score: {score:.3f})\n{content}\n\n"
        
        prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            # Build completion kwargs
            completion_kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            }
            
            # Add api_base if set (for custom endpoints like NVIDIA NIM)
            if hasattr(self, 'litellm') and hasattr(self.litellm, 'api_base') and self.litellm.api_base:
                completion_kwargs["api_base"] = self.litellm.api_base
            
            response = self.litellm.completion(**completion_kwargs)
            
            answer = response.choices[0].message.content
            
            # Extract citations from contexts
            citations = []
            for ctx in contexts:
                source_id = ctx.get("id", "unknown")
                score = ctx.get("similarity", 0)
                if score > 0.3:
                    citations.append({
                        "id": source_id,
                        "score": score,
                        "text": ctx.get("text", "")[:200] + "..."
                    })
            
            # Calculate confidence based on average similarity
            scores = [c.get("similarity", 0) for c in contexts]
            confidence = sum(scores) / len(scores) if scores else 0.0
            
            return {
                "text": answer,
                "confidence": confidence,
                "model": self.model_name,
                "citations": citations
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "text": f"Error generating response: {str(e)}",
                "confidence": 0.0,
                "model": self.model_name,
                "citations": []
            }
