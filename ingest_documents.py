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
    print("[DOC] Document Ingestion to ChromaDB")
    print("=" * 70)
    
    # Check dependencies
    if not HAS_UNSTRUCTURED:
        print("\n[WARN] unstructured not installed")
        print("Install with: pip install unstructured[pdf]")
        print("PDF support will be limited, but text files will work\n")
    
    # Initialize vector store
    print("\n[INIT] Initializing ChromaDB vector store...")
    vector_store = ChromaDBVectorStore(
        persist_path="./data/vector_store",
        collection_name="documents",
        dimension=384
    )
    print(f"[OK] Collection created: {vector_store.collection_name}")
    print(f"[OK] Current count: {vector_store.get_stats()['count']} documents")
    
    # Initialize ingester
    print("\n[INIT] Initializing document ingester...")
    ingester = DocumentIngestor(
        chunk_size=512,
        chunk_overlap=51
    )
    print("[OK] Document ingester ready")
    
    # Ingest sample files
    print("\n" + "=" * 70)
    print("[DOC] Ingesting Documents")
    print("=" * 70)
    
    # Sample documents to ingest
    sample_files = [
        "./data/sample_test_docs/returns_policy.md",
        "./data/sample_test_docs/shipping_info.md",
        "./data/sample_test_docs/product_catalog.md",
        "./data/sample_test_docs/faq.md",
    ]
    
    total_chunks = 0
    
    for file_path in sample_files:
        if Path(file_path).exists():
            try:
                print(f"\n[FILE] Processing: {file_path}")
                chunks = ingester.ingest_file(
                    file_path,
                    metadata={"doc_type": "faq", "source": file_path}
                )
                print(f"   [OK] Created {len(chunks)} chunks")
                
                # Add to vector store
                texts = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                vector_store.add_batch(texts, metadatas)
                print(f"   [OK] Added to vector store: {len(chunks)} chunks")
                
                total_chunks += len(chunks)
            except Exception as e:
                print(f"   [ERROR] {e}")
        else:
            print(f"\n[SKIP] File not found: {file_path}")
    
    # Ingest directory if available
    sample_docs_dir = Path("./data/sample_docs")
    if sample_docs_dir.exists():
        print(f"\n[DIR] Ingesting entire directory: {sample_docs_dir}")
        all_chunks = ingester.ingest_directory(str(sample_docs_dir), recursive=True)
        print(f"   [OK] Total chunks from directory: {len(all_chunks)}")
        
        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            metadatas = [c["metadata"] for c in all_chunks]
            vector_store.add_batch(texts, metadatas)
            print(f"   [OK] Added to vector store: {len(all_chunks)} chunks")
            total_chunks += len(all_chunks)
    
    # Final stats
    print("\n" + "=" * 70)
    print("[STATS] Final Statistics")
    print("=" * 70)
    stats = vector_store.get_stats()
    print(f"Collection: {stats['name']}")
    print(f"Document count: {stats['count']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Total chunks ingested: {total_chunks}")
    
    # Persist
    print("\n[SAVE] Persisting database...")
    vector_store.persist()
    print("[OK] Database persisted to disk")
    
    # Test search
    print("\n" + "=" * 70)
    print("[TEST] Testing Search")
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
    print("[SUCCESS] Ingestion Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start the API server: python -m uvicorn app:app --reload")
    print("2. Visit: http://localhost:8000")
    print("3. Query your documents using the chat interface")
    print()


if __name__ == "__main__":
    main()
