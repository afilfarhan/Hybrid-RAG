# Setup ChromaDB with PDF Ingestion

This guide will help you set up ChromaDB and ingest PDF files into your Hybrid RAG system.

## 📋 Prerequisites

- Python 3.9+
- pip package manager

## 🚀 Quick Setup

### Step 1: Install Required Dependencies

```bash
# Core dependencies (if not already installed)
pip install -r requirements.txt

# Additional dependencies for PDF support
pip install unstructured[pdf]
pip install pdfminer.six
pip install pdfplumber
```

### Step 2: Verify Installation

```bash
python -c "import chromadb; print('ChromaDB:', chromadb.__version__)"
python -c "from unstructured.partition.pdf import partition_pdf; print('Unstructured OK')"
```

### Step 3: Create Sample PDF Files (Optional)

Create a directory for your PDF files:

```bash
mkdir -p data/sample_docs
```

You can download sample PDFs or create your own. For testing, you can convert Markdown to PDF:

```bash
# Install pypandoc for Markdown to PDF conversion
pip install pypandoc

# Or use online tools to convert your documents to PDF
```

## 📂 Ingest Documents

### Option 1: Using the Ingestion Script

```bash
# Create an ingestion script
python document_ingestion.py > ingest.py

# Run the ingestion script
python ingest.py
```

### Option 2: Manual Ingestion

```python
from services.chromadb_store import ChromaDBVectorStore
from document_ingestion import DocumentIngestor

# Initialize vector store
vector_store = ChromaDBVectorStore(
    persist_path="./data/vector_store",
    collection_name="documents",
    dimension=384
)

# Initialize ingester
ingester = DocumentIngestor(chunk_size=512, chunk_overlap=51)

# Ingest a single PDF
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

### Option 3: Batch Ingest Directory

```python
from services.chromadb_store import ChromaDBVectorStore
from document_ingestion import DocumentIngestor

# Initialize
vector_store = ChromaDBVectorStore(
    persist_path="./data/vector_store",
    collection_name="documents"
)

ingester = DocumentIngestor()

# Ingest entire directory
all_chunks = ingester.ingest_directory("./data/sample_docs", recursive=True)

# Add to vector store
texts = [c["text"] for c in all_chunks]
metadatas = [c["metadata"] for c in all_chunks]
vector_store.add_batch(texts, metadatas)

# Persist
vector_store.persist()

print(f"✓ Ingested {len(all_chunks)} chunks from directory")
```

## 🔧 Supported File Formats

The `DocumentIngestor` supports:

- **PDF** - Requires `unstructured[pdf]`
- **Markdown** (.md) - Requires `unstructured`
- **Text** (.txt) - Built-in
- **HTML** - Requires `unstructured`

## 📊 Verify Ingestion

```python
from services.chromadb_store import ChromaDBVectorStore

# Load existing collection
vector_store = ChromaDBVectorStore(
    persist_path="./data/vector_store",
    collection_name="documents"
)

# Check stats
stats = vector_store.get_stats()
print(f"Documents in collection: {stats['count']}")
print(f"Dimension: {stats['dimension']}")

# Test search
results = vector_store.search("What is your return policy?", top_k=3)
for i, result in enumerate(results):
    print(f"\n{i+1}. {result['text'][:100]}...")
    print(f"   Source: {result['metadata'].get('source', 'N/A')}")
    print(f"   Similarity: {result['similarity']:.3f}")
```

## 🔄 Update Documents

To update or replace documents:

```python
from services.chromadb_store import ChromaDBVectorStore

vector_store = ChromaDBVectorStore(
    persist_path="./data/vector_store",
    collection_name="documents"
)

# Clear collection (optional)
# vector_store.clear()

# Re-ingest
# ... (use ingestion code above)

# Persist
vector_store.persist()
```

## 📁 Directory Structure

```
Hybrid RAG/
├── data/
│   └── vector_store/     # ChromaDB persists here
│       ├── chroma.sqlite3
│       └── embeddings/
└── sample_docs/          # Your documents
    ├── returns_policy.pdf
    ├── shipping_info.pdf
    └── product_catalog.pdf
```

## 🐛 Troubleshooting

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
# Delete and recreate
import chromadb
client = chromadb.PersistentClient(path="./data/vector_store")
client.delete_collection("documents")
# Then recreate collection
```

### Issue: PDF parsing errors

**Solutions:**
1. Try a different PDF (some scanned PDFs need OCR)
2. Use `pdfplumber` directly for complex PDFs
3. Convert PDF to text first using Adobe Acrobat or similar

### Issue: Out of memory with large PDFs

**Solutions:**
1. Reduce `chunk_size` parameter
2. Process PDFs in batches
3. Use a machine with more RAM

## 🚀 Next Steps

1. **Test your setup:**
   ```bash
   python -c "
   from services.chromadb_store import ChromaDBVectorStore
   vs = ChromaDBVectorStore()
   print('Stats:', vs.get_stats())
   results = vs.search('test query', top_k=2)
   print('Results:', len(results))
   "
   ```

2. **Start the API server:**
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Query your documents:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is your return policy?"}'
   ```

## 📚 Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Unstructured Documentation](https://unstructured-io.github.io/unstructured/)
- [Sentence Transformers](https://www.sbert.net/)

## 💡 Tips

1. **Chunk Size**: Smaller chunks (256-512) work better for retrieval
2. **Overlap**: 10-20% overlap ensures context isn't lost
3. **Metadata**: Always include source and document type
4. **Persistence**: ChromaDB automatically persists to disk
5. **Batch Ingestion**: Process documents in batches for large datasets
