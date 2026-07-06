"""
Add sample documents to the vector store for testing.
"""

from src.services.embedding_inmemory import InMemoryEmbeddingService
from src.services.vector_store_inmemory import InMemoryVectorStore, VectorChunk
from src.services.retriever_inmemory import InMemoryRetriever


def add_sample_data_to_retriever(retriever: InMemoryRetriever) -> InMemoryRetriever:
    """Add sample data to an existing retriever."""
    sample_texts = [
        "We offer a 30-day return policy for all products. Items must be in their original condition with all packaging and accessories included.",
        "To return an item, contact our support team within 30 days of purchase to request a return authorization. We will provide you with a return label and instructions.",
        "Refunds are processed within 5-7 business days after we receive your returned item.",
        "Standard shipping takes 3-5 business days. Free for orders over $50, $5.99 for orders under $50.",
        "Express shipping takes 1-2 business days and costs $14.99 with tracking available online.",
        "We ship internationally to over 50 countries with delivery time of 7-14 business days.",
        "Premium Wireless Headphones cost $299.99 with active noise cancellation and 30-hour battery life.",
        "Smart Watch Pro costs $399.99 with heart rate monitoring, GPS tracking, and 7-day battery life.",
        "Wireless Earbuds cost $149.99 with active noise cancellation and 24-hour battery life.",
        "We accept credit cards, PayPal, Apple Pay, and Google Pay for payments.",
        "All our products come with a 1-year manufacturer warranty covering defects in materials and workmanship.",
    ]
    
    chunks = []
    for i, text in enumerate(sample_texts):
        chunks.append({
            "id": f"chunk_{i}",
            "text": text,
            "metadata": {
                "source": "sample_data",
                "chunk_id": f"chunk_{i}",
                "doc_type": "faq"
            }
        })
    
    retriever.add_chunks(chunks)
    print(f"Added {len(chunks)} sample chunks to vector store")
    return retriever


if __name__ == "__main__":
    # Test with fresh services
    embedding_service = InMemoryEmbeddingService(dimension=384)
    vector_store = InMemoryVectorStore(dimension=384)
    retriever = InMemoryRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=5,
        dense_weight=0.7,
        sparse_weight=0.3,
    )
    
    add_sample_data_to_retriever(retriever)
    
    # Test query
    results = retriever.retrieve("What is your return policy?")
    print(f"\nRetrieved {len(results)} results:")
    for r in results:
        print(f"  - {r.get('text', '')[:100]}...")
