"""
Hybrid RAG - LiteLLM generator with citations
"""

from typing import Any, Dict, List, Optional

import litellm

from src.core.provider_config import get_provider_model
from src.generation.base import BaseGenerator, GenerationResult
from src.retrieval.base import RetrievedChunk


class LiteLMGenerator(BaseGenerator):
    """Generator using LiteLLM with support for multiple providers and citation support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Handle provider-based model naming
        model_name = config.get("model_name", "gpt-4o-mini")
        provider = config.get("provider", None)
        
        if provider:
            model_name = get_provider_model(provider, "llm", model_name)
        
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", None)

        self.model_name = model_name
        self.system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Your answers must be grounded in the provided context and include citations to the source documents.

Instructions:
1. Answer the question using ONLY the information from the provided context
2. Include citations in the format [Source: doc_id] at the end of each claim
3. If the answer is not in the context, say "I don't have enough information to answer that question"
4. Be concise and focused on the question
5. Include the most relevant citations for each part of your answer"""

        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url

    async def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> GenerationResult:
        """Generate an answer from retrieved chunks."""
        context = self._build_context(retrieved_chunks)

        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            answer = response.choices[0].message.content

            citations = self._extract_citations(retrieved_chunks)
            confidence = self._calculate_confidence(retrieved_chunks)

            return GenerationResult(
                answer=answer,
                citations=citations,
                confidence=confidence,
                metadata={
                    "model": self.model_name,
                    "num_chunks_used": len(retrieved_chunks),
                },
            )

        except Exception as e:
            print(f"Error generating answer: {e}")
            return GenerationResult(
                answer="I encountered an error while generating the answer.",
                citations=[],
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata
            source = metadata.get("source", "unknown")
            chunk_id = metadata.get("chunk_id", f"chunk_{i}")

            context_parts.append(
                f"[Chunk {i} | Source: {source} | ID: {chunk_id}]\n"
                f"{chunk.content}\n"
            )

        return "\n\n".join(context_parts)

    def _extract_citations(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Extract citations from retrieved chunks."""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            citations.append(
                {
                    "id": i,
                    "chunk_id": chunk.metadata.get("chunk_id", ""),
                    "source": chunk.metadata.get("source", "unknown"),
                    "score": chunk.score,
                    "content_preview": (
                        chunk.content[:200] + "..."
                        if len(chunk.content) > 200
                        else chunk.content
                    ),
                }
            )
        return citations

    def _calculate_confidence(self, chunks: List[RetrievedChunk]) -> float:
        """Calculate confidence score based on retrieval scores."""
        if not chunks:
            return 0.0

        avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
        return min(max(avg_score, 0.0), 1.0)
