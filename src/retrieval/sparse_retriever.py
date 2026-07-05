"""
Hybrid RAG - Sparse retriever using BM25
"""

from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever, RetrievedChunk


class SparseRetriever(BaseRetriever):
    """Sparse retriever using BM25."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []

    async def retrieve(
        self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks using BM25 sparse search."""
        if self.bm25 is None:
            return []

        # Tokenize query
        query_tokens = query.lower().split()

        # Get scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Format results
        chunks = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                chunks.append(
                    RetrievedChunk(
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                        score=float(scores[idx]),
                        source=doc.get("metadata", {}).get("source", "unknown"),
                    )
                )

        return chunks

    async def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve using hybrid search (dense + sparse)."""
        # TODO: Implement hybrid search combining dense and sparse results
        return await self.retrieve(query, top_k, filter)

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents for sparse search."""
        self.documents = documents

        # Tokenize documents
        tokenized_docs = []
        for doc in documents:
            content = doc.get("content", "")
            tokens = content.lower().split()
            tokenized_docs.append(tokens)

        self.bm25 = BM25Okapi(tokenized_docs)
