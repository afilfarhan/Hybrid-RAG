"""
Quick test script to verify Hybrid RAG setup.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    try:
        from services.chromadb_store import ChromaDBVectorStore
        print("[OK] ChromaDBVectorStore imported")
    except Exception as e:
        print(f"[FAIL] ChromaDBVectorStore import failed: {e}")
        return False
    
    try:
        from document_ingestion import DocumentIngestor
        print("[OK] DocumentIngestor imported")
    except Exception as e:
        print(f"[FAIL] DocumentIngestor import failed: {e}")
        return False
    
    try:
        from services.inmemory import InMemoryEmbeddingService
        print("[OK] InMemoryEmbeddingService imported")
    except Exception as e:
        print(f"[FAIL] InMemoryEmbeddingService import failed: {e}")
        return False
    
    try:
        from services.inmemory import InMemoryRetrievalService
        print("[OK] InMemoryRetrievalService imported")
    except Exception as e:
        print(f"[FAIL] InMemoryRetrievalService import failed: {e}")
        return False
    
    return True


def test_vector_store():
    """Test vector store operations."""
    print("\n" + "=" * 60)
    print("Testing Vector Store")
    print("=" * 60)
    
    try:
        from services.chromadb_store import ChromaDBVectorStore
        
        # Create test collection
        vector_store = ChromaDBVectorStore(
            persist_path="./data/test_vector_store",
            collection_name="test_collection",
            dimension=384
        )
        print("[OK] Vector store initialized")
        
        # Test adding documents
        texts = [
            "The return policy allows returns within 30 days.",
            "Shipping takes 5-7 business days for standard delivery.",
            "We accept all major credit cards and PayPal."
        ]
        metadatas = [
            {"doc_type": "policy", "section": "returns"},
            {"doc_type": "policy", "section": "shipping"},
            {"doc_type": "policy", "section": "payments"}
        ]
        
        vector_store.add_batch(texts, metadatas)
        print("[OK] Documents added to vector store")
        
        # Test search
        results = vector_store.search("What is your return policy?", top_k=2)
        print(f"[OK] Search returned {len(results)} results")
        
        if results:
            print(f"  Top result: {results[0]['text'][:80]}...")
            print(f"  Similarity: {results[0]['similarity']:.3f}")
        
        # Test stats
        stats = vector_store.get_stats()
        print(f"[OK] Collection stats: {stats['count']} documents")
        
        # Clean up
        import chromadb
        client = chromadb.PersistentClient(path="./data/test_vector_store")
        client.delete_collection("test_collection")
        print("[OK] Test collection cleaned up")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ingestion():
    """Test document ingestion."""
    print("\n" + "=" * 60)
    print("Testing Document Ingestion")
    print("=" * 60)
    
    try:
        from document_ingestion import DocumentIngestor
        
        ingester = DocumentIngestor(chunk_size=256, chunk_overlap=25)
        
        # Test with existing sample file
        sample_file = "./data/sample_docs/returns_policy.md"
        if Path(sample_file).exists():
            chunks = ingester.ingest_file(sample_file, metadata={"doc_type": "policy"})
            print(f"[OK] Document ingested: {len(chunks)} chunks created")
            
            if chunks:
                print(f"  First chunk: {chunks[0]['text'][:80]}...")
            
            return True
        else:
            print(f"[WARN] Sample file not found: {sample_file}")
            return True
        
    except Exception as e:
        print(f"[FAIL] Document ingestion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Hybrid RAG - Quick Test")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Vector Store
    results.append(("Vector Store", test_vector_store()))
    
    # Test 3: Ingestion
    results.append(("Ingestion", test_ingestion()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! Your Hybrid RAG system is ready.")
        print("\nNext steps:")
        print("1. Add your PDF files to data/sample_docs/")
        print("2. Run: python ingest_documents.py")
        print("3. Start server: python -m uvicorn app:app --reload")
        print("4. Visit: http://localhost:8000")
    else:
        print("\n[WARN] Some tests failed. Check the errors above.")
    
    print("\n")


if __name__ == "__main__":
    main()
