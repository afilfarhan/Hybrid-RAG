"""
Hybrid RAG - OpenAI generator with citations
"""

from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI

from src.generation.base import BaseGenerator, GenerationResult
from src.retrieval.base import RetrievedChunk


class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI models with citation support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        api_key = config.get("api_key", "")
        model_name = config.get("model_name", "gpt-4o-mini")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name

        # System prompt for grounded generation
        self.system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Your answers must be grounded in the provided context and include citations to the source documents.

Instructions:
1. Answer the question using ONLY the information from the provided context
2. Include citations in the format [Source: doc_id] at the end of each claim
3. If the answer is not in the context, say "I don't have enough information to answer that question"
4. Be concise and focused on the question
5. Include the most relevant citations for each part of your answer"""

    async def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> GenerationResult:
        """Generate an answer from retrieved chunks."""
        # Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)

        # Build prompt
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Low temperature for grounded responses
                max_tokens=1000,
            )

            answer = response.choices[0].message.content

            # Extract citations from retrieved chunks
            citations = self._extract_citations(retrieved_chunks)

            # Calculate confidence based on retrieval scores
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
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                }
            )
        return citations

    def _calculate_confidence(self, chunks: List[RetrievedChunk]) -> float:
        """Calculate confidence score based on retrieval scores."""
        if not chunks:
            return 0.0

        avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
        return min(max(avg_score, 0.0), 1.0)
