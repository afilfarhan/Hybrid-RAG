# Hybrid RAG - Complete Setup Guide

This guide will help you set up ChromaDB and ingest PDF files into your Hybrid RAG system.

## 📋 Prerequisites

- Python 3.9+
- pip package manager
- 4GB+ RAM (8GB recommended for local embeddings)

## 🚀 Quick Setup (Windows)

### 1. Run the Setup Script

```bash
# Double-click setup.bat or run in PowerShell
.\setup.bat
```

### 2. Add Your PDF Files

Place your PDF files in `data/sample_docs/` directory:

```
Hybrid RAG/
└── data/
    └── sample_docs/
        ├── returns_policy.pdf
        ├── shipping_info.pdf
        └── product_catalog.pdf
```

### 3. Run Ingestion Script

```bash
python ingest_documents.py
```

This will:
- Create ChromaDB vector store at `data/vector_store/`
- Chunk and embed all documents
- Store embeddings for semantic search

### 4. Start the Server

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Application

- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

---

## 📖 Manual Setup

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# PDF processing (recommended)
pip install unstructured[pdf]
pip install pdfminer.six
pip install pdfplumber
```

### Step 2: Verify Installation

```bash
python -c "import chromadb; print('ChromaDB OK')"
python -c "from unstructured.partition.pdf import partition_pdf; print('Unstructured OK')"
```

### Step 3: Test the System

```bash
python test_setup.py
```

### Step 4: Ingest Documents

```bash
# Run the ingestion script
python ingest_documents.py
```

Or manually:

```python
from services.chromadb_store import ChromaDBVectorStore
from document_ingestion import DocumentIngestor

# Initialize
vector_store = ChromaDBVectorStore(
    persist_path="./data/vector_store",
    collection_name="documents"
)

ingester = DocumentIngestor(chunk_size=512, chunk_overlap=51)

# Ingest a PDF
chunks = ingester.ingest_file(
    "./data/sample_docs/your_document.pdf",
    metadata={"doc_type": "policy"}
)

# Add to vector store
texts = [c["text"] for c in chunks]
metadatas = [c["metadata"] for c in chunks]
vector_store.add_batch(texts, metadatas)

# Persist
vector_store.persist()

print(f"✓ Ingested {len(chunks)} chunks")
```

### Step 5: Test Search

```python
from services.chromadb_store import ChromaDBVectorStore

vector_store = ChromaDBVectorStore(
    persist_path="./data/vector_store",
    collection_name="documents"
)

results = vector_store.search("What is your return policy?", top_k=3)
for result in results:
    print(f"[{result['similarity']:.3f}] {result['text'][:100]}...")
```

---

## 📂 Project Structure

```
Hybrid RAG/
├── services/
│   ├── chromadb_store.py      # ChromaDB vector store
│   ├── inmemory.py            # In-memory fallback services
│   └── rag_service.py         # RAG orchestration
├── data/
│   ├── vector_store/          # ChromaDB persists here
│   │   ├── chroma.sqlite3
│   │   └── embeddings/
│   └── sample_docs/           # Place your PDFs here
├── ingest_documents.py        # Document ingestion script
├── test_setup.py              # Test script
├── app.py                     # Main entry point
└── requirements.txt           # Dependencies
```

---

## 🎯 Supported File Formats

| Format | Extension | Required Package |
|--------|-----------|------------------|
| PDF | .pdf | `unstructured[pdf]` |
| Markdown | .md, .markdown | `unstructured` |
| Text | .txt | Built-in |
| HTML | .html | `unstructured` |

---

## ⚙️ Configuration

Edit `.env` to customize:

```env
# Vector Store
CHROMA_PERSIST_PATH=./data/vector_store
CHROMA_COLLECTION_NAME=documents

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=51

# OpenAI (optional)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4
```

---

## 🧪 Troubleshooting

### Issue: "unstructured not installed"

**Solution:**
```bash
pip install unstructured[pdf]
pip install pdfminer.six
pip install pdfplumber
```

### Issue: "ChromaDB collection already exists"

**Solution:**
```python
import chromadb
client = chromadb.PersistentClient(path="./data/vector_store")
client.delete_collection("documents")
# Then recreate collection
```

### Issue: PDF parsing errors

**Solutions:**
1. Try a different PDF (some scanned PDFs need OCR)
2. Convert PDF to text first
3. Use `pdfplumber` directly for complex PDFs

### Issue: Out of memory with large PDFs

**Solutions:**
1. Reduce `chunk_size` parameter
2. Process PDFs in batches
3. Use a machine with more RAM

### Issue: Slow embedding

**Solutions:**
1. Use a GPU-enabled machine
2. Reduce batch size
3. Use a smaller embedding model

---

## 📚 Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Unstructured Documentation](https://unstructured-io.github.io/unstructured/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 🎓 Next Steps

1. **Add more documents** - Place PDFs in `data/sample_docs/` and re-run ingestion
2. **Fine-tune chunking** - Adjust `chunk_size` and `chunk_overlap` in `ingest_documents.py`
3. **Customize embeddings** - Switch to OpenAI embeddings in `.env`
4. **Deploy** - Use Docker for production deployment

---

## 💡 Tips

1. **Chunk Size**: Smaller chunks (256-512) work better for retrieval
2. **Overlap**: 10-20% overlap ensures context isn't lost
3. **Metadata**: Always include source and document type
4. **Persistence**: ChromaDB automatically persists to disk
5. **Batch Ingestion**: Process documents in batches for large datasets

---

## ✅ Verification

Run the test script to verify everything works:

```bash
python test_setup.py
```

Expected output:
```
============================================================
Hybrid RAG - Quick Test
============================================================

============================================================
Testing Imports
============================================================
[OK] ChromaDBVectorStore imported
[OK] DocumentIngestor imported
[OK] InMemoryEmbeddingService imported
[OK] InMemoryRetrievalService imported

============================================================
Testing Vector Store
============================================================
[OK] Vector store initialized
[OK] Documents added to vector store
[OK] Search returned 2 results
[OK] Collection stats: 3 documents
[OK] Test collection cleaned up

============================================================
Testing Document Ingestion
============================================================
[OK] Document ingested: 7 chunks created

============================================================
Test Summary
============================================================
Imports: [PASS]
Vector Store: [PASS]
Ingestion: [PASS]

[SUCCESS] All tests passed! Your Hybrid RAG system is ready.
```
