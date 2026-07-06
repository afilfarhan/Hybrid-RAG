"""
Sample script to ingest PDF files into ChromaDB.

Usage:
    python ingest_documents.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from services.chromadb_store import ChromaDBVectorStore
from document_ingestion import DocumentIngestor, HAS_UNSTRUCTURED


def main():
    print("=" * 70)
    print("📄 Document Ingestion to ChromaDB")
    print("=" * 70)
    
    # Check dependencies
    if not HAS_UNSTRUCTURED:
        print("\n⚠️  WARNING: unstructured not installed")
        print("Install with: pip install unstructured[pdf]")
        print("This script will still work but PDF support may be limited\n")
    
    # Initialize vector store
    print("\n🔧 Initializing ChromaDB vector store...")
    vector_store = ChromaDBVectorStore(
        persist_path="./data/vector_store",
        collection_name="documents",
        dimension=384
    )
    print(f"✓ Collection created: {vector_store.collection_name}")
    print(f"✓ Current count: {vector_store.get_stats()['count']} documents")
    
    # Initialize ingester
    print("\n🔧 Initializing document ingester...")
    ingester = DocumentIngestor(
        chunk_size=512,
        chunk_overlap=51
    )
    print("✓ Document ingester ready")
    
    # Ingest sample files
    print("\n" + "=" * 70)
    print("📚 Ingesting Documents")
    print("=" * 70)
    
    # Sample documents to ingest
    sample_files = [
        "./data/sample_docs/returns_policy.md",
        "./data/sample_docs/shipping_info.md",
        "./data/sample_docs/product_catalog.md",
        "./data/sample_docs/faq.md",
    ]
    
    total_chunks = 0
    
    for file_path in sample_files:
        if Path(file_path).exists():
            try:
                print(f"\n📄 Processing: {file_path}")
                chunks = ingester.ingest_file(
                    file_path,
                    metadata={"doc_type": "faq", "source": file_path}
                )
                print(f"   ✓ Created {len(chunks)} chunks")
                
                # Add to vector store
                texts = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                vector_store.add_batch(texts, metadatas)
                print(f"   ✓ Added to vector store: {len(chunks)} chunks")
                
                total_chunks += len(chunks)
            except Exception as e:
                print(f"   ✗ Error: {e}")
        else:
            print(f"\n⚠️  File not found (skipping): {file_path}")
    
    # Ingest directory if available
    sample_docs_dir = Path("./data/sample_docs")
    if sample_docs_dir.exists():
        print(f"\n📁 Ingesting entire directory: {sample_docs_dir}")
        all_chunks = ingester.ingest_directory(str(sample_docs_dir), recursive=True)
        print(f"   ✓ Total chunks from directory: {len(all_chunks)}")
        
        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            metadatas = [c["metadata"] for c in all_chunks]
            vector_store.add_batch(texts, metadatas)
            print(f"   ✓ Added to vector store: {len(all_chunks)} chunks")
            total_chunks += len(all_chunks)
    
    # Final stats
    print("\n" + "=" * 70)
    print("📊 Final Statistics")
    print("=" * 70)
    stats = vector_store.get_stats()
    print(f"Collection: {stats['name']}")
    print(f"Document count: {stats['count']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Total chunks ingested: {total_chunks}")
    
    # Persist
    print("\n💾 Persisting database...")
    vector_store.persist()
    print("✓ Database persisted to disk")
    
    # Test search
    print("\n" + "=" * 70)
    print("🔍 Testing Search")
    print("=" * 70)
    
    test_queries = [
        "What is your return policy?",
        "How long does shipping take?",
        "Do you accept credit cards?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vector_store.search(query, top_k=3)
        for i, result in enumerate(results):
            print(f"  {i+1}. [{result['similarity']:.3f}] {result['text'][:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ Ingestion Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start the API server: python -m uvicorn app:app --reload")
    print("2. Visit: http://localhost:8000")
    print("3. Query your documents using the chat interface")
    print()


if __name__ == "__main__":
    main()
