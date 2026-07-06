"""
Mock generator service for testing.
"""

from typing import List, Dict, Any


class MockGeneratorService:
    """Mock LLM generator that constructs answers from retrieved chunks."""
    
    def __init__(self):
        self._model_name = "mock-gpt-4"
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """Generate an answer from retrieved contexts."""
        if not contexts:
            return {
                "answer": "I don't have enough information to answer this question.",
                "confidence": 0.0,
                "citations": [],
                "metadata": {
                    "model": self._model_name,
                    "used_contexts": 0,
                },
            }
        
        context_texts = [ctx["text"] for ctx in contexts[:3]]
        combined_context = "\n\n".join(context_texts)
        
        answer = f"Based on the available information:\n\n{combined_context}\n\nThis is a mock answer generated from the retrieved contexts."
        
        citations = [
            {
                "chunk_id": ctx["id"],
                "text": ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
                "score": ctx.get("score", 0),
            }
            for ctx in contexts[:3]
        ]
        
        confidence = sum(ctx.get("score", 0) for ctx in contexts[:3]) / min(len(contexts), 3)
        
        return {
            "answer": answer,
            "confidence": round(confidence, 2),
            "citations": citations,
            "metadata": {
                "model": self._model_name,
                "used_contexts": len(contexts),
                "max_tokens": max_tokens,
            },
        }
    
    def get_model_name(self) -> str:
        """Return model name."""
        return self._model_name
