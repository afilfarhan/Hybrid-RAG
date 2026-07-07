"""
In-Memory Services for Development/Testing

These services provide fallback functionality when real services are unavailable.
"""

import numpy as np
from typing import List, Dict, Any
from .base import EmbeddingService, VectorStore, RetrievalService, GenerationService


class InMemoryEmbeddingService(EmbeddingService):
    """Simple in-memory embedding service using a pre-trained model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return [e.tolist() for e in embeddings]


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using numpy for similarity search."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
    
    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a document to the store."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode([text])[0].tolist()
        
        doc_id = f"doc_{len(self.documents)}"
        self.documents.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata
        })
        self.embeddings.append(embedding)
        
        return doc_id
    
    def add_batch(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to the store."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)
        
        doc_ids = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            doc_id = f"doc_{len(self.documents)}"
            self.documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata
            })
            self.embeddings.append(embedding.tolist())
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode([query])[0]
        
        if not self.embeddings:
            return []
        
        # Calculate cosine similarity
        embeddings_array = np.array(self.embeddings)
        query_array = np.array(query_embedding)
        
        similarities = np.dot(embeddings_array, query_array) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_array) + 1e-8
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                **self.documents[idx],
                "similarity": float(similarities[idx])
            })
        
        return results


class InMemoryRetrievalService(RetrievalService):
    """Simple retrieval service using in-memory vector store."""
    
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        k = top_k or self.top_k
        return self.vector_store.search(query, k)


class InMemoryGenerationService(GenerationService):
    """Simple in-memory generation service using a template-based approach."""
    
    def __init__(self, model_name: str = "in-memory"):
        self.model_name = model_name
    
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
                "model": self.model_name
            }
        
        # Extract context texts and scores
        context_items = [(c["text"], c.get("similarity", 0)) for c in contexts]
        context_items.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out low-quality chunks (very low similarity or very short)
        filtered_contexts = [
            (text, score) for text, score in context_items 
            if score > 0.3 and len(text.strip()) > 20
        ]
        
        if not filtered_contexts:
            return {
                "text": "I don't have enough relevant information to answer this question.",
                "confidence": 0.0,
                "model": self.model_name
            }
        
        # Extract just the relevant parts - take first 3 high-quality chunks
        top_chunks = filtered_contexts[:3]
        
        # Clean up text - remove page numbers, headers, footers, copyright notices
        cleaned_chunks = []
        for text, score in top_chunks:
            import re
            cleaned = text.strip()
            # Remove page markers with special characters (dashes, commas, backticks, etc.)
            cleaned = re.sub(r'[`\-,\.]{2,}', ' ', cleaned)
            # Remove excessive dashes and special characters
            cleaned = re.sub(r'[\u2014\u2013]{2,}', ' ', cleaned)
            # Remove multiple spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            # Remove leading/trailing special characters
            cleaned = cleaned.strip('`-,.\u2014\u2013 ')
            
            # Additional cleanup: remove common noise words/phrases
            # Remove "Control" when used as a noise word (common in scanned PDFs)
            cleaned = re.sub(r'\bControl\b', '', cleaned)
            # Remove ISO copyright markers
            cleaned = re.sub(r'© ISO/IEC \d+ – All rights reserved.*?(?=\bISO\b|$)', '', cleaned, flags=re.IGNORECASE)
            # Remove page numbers like "19 ISO/IEC 27001:2022(E)"
            cleaned = re.sub(r'\b\d+\s+ISO/IEC.*?\(E\)\b', '', cleaned, flags=re.IGNORECASE)
            # Clean up multiple spaces again
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            cleaned_chunks.append(cleaned)
        
        # Generate a summary based on the query
        query_lower = query.lower()
        answer_intro = ""
        
        if "cybersecurity" in query_lower or "security" in query_lower:
            answer_intro = "According to ISO/IEC 27001, cybersecurity involves:\n\n"
        elif "leadership" in query_lower:
            answer_intro = "Leadership and commitment according to ISO/IEC 27001 require:\n\n"
        elif "planning" in query_lower:
            answer_intro = "When planning the information security management system, consider:\n\n"
        else:
            answer_intro = "Here is the relevant information:\n\n"
        
        # Build a concise response
        response_parts = []
        for i, chunk in enumerate(cleaned_chunks, 1):
            # Extract just the relevant sentence or two
            sentences = chunk.split('. ')
            # Take first 2-3 sentences that seem relevant
            relevant_sentences = sentences[:3]
            # Filter out very short sentences
            meaningful = [s.strip() for s in relevant_sentences if len(s.strip()) > 10]
            if meaningful:
                response_parts.append(f"• {' '.join(meaningful)}.")
        
        # Build the final response
        response = answer_intro + "\n".join(response_parts)
        response += f"\n\nAnswer: This information is derived from {len(cleaned_chunks)} relevant document chunks."
        
        # Calculate confidence based on similarity scores
        avg_similarity = np.mean([score for _, score in filtered_contexts]) if filtered_contexts else 0
        confidence = min(avg_similarity, 0.95)
        
        return {
            "text": response,
            "confidence": float(confidence),
            "model": self.model_name
        }
