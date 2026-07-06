# Hybrid RAG - Setup Guide

Complete setup instructions for the Hybrid RAG system with ChromaDB and PDF ingestion.

## 📋 Prerequisites

- Python 3.9+
- pip package manager
- 4GB+ RAM (8GB recommended for local embeddings)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install PDF processing (recommended)
pip install unstructured[pdf]
pip install pdfminer.six
pip install pdfplumber
```

### 2. Start ChromaDB & Ingest Documents

```bash
# Run the ingestion script (creates vector store automatically)
python ingest_documents.py
```

### 3. Start the API Server

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Application

- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## 📂 Project Structure

```
Hybrid RAG/
├── app.py                  # Main entry point
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── ingest_documents.py    # Document ingestion script
├── SETUP_CHROMADB.md      # ChromaDB setup guide
│
├── services/
│   ├── chromadb_store.py  # ChromaDB vector store
│   ├── embedding_inmemory.py
│   ├── retriever_inmemory.py
│   └── generator_inmemory.py
│
├── data/
│   └── vector_store/      # ChromaDB persists here
│       ├── chroma.sqlite3
│       └── embeddings/
│
└── sample_docs/           # Place your PDFs here
    ├── returns_policy.pdf
    ├── shipping_info.pdf
    └── product_catalog.pdf
```

## 🎯 Ingesting PDFs

### Method 1: Using the Script (Recommended)

```bash
# Place your PDFs in data/sample_docs/
# Then run:
python ingest_documents.py
```

### Method 2: Manual Ingestion

```python
from services.chromadb_store import ChromaDBVectorStore
from document_ingestion import DocumentIngestor

# Initialize
vector_store = ChromaDBVectorStore(
    persist_path="./data/vector_store",
    collection_name="documents"
)

ingester = DocumentIngestor()

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
```

### Method 3: Ingest Directory

```python
# Ingest all documents in a directory
all_chunks = ingester.ingest_directory(
    "./data/sample_docs",
    recursive=True
)

# Add to vector store
texts = [c["text"] for c in all_chunks]
metadatas = [c["metadata"] for c in all_chunks]
vector_store.add_batch(texts, metadatas)
```

## 🔧 Supported Formats

- **PDF** - Requires `unstructured[pdf]`
- **Markdown** (.md) - Built-in
- **Text** (.txt) - Built-in
- **HTML** - Requires `unstructured`

## ⚙️ Configuration

Edit `.env` to configure:

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

## 🧪 Testing

```bash
# Test vector store
python -c "
from services.chromadb_store import ChromaDBVectorStore
vs = ChromaDBVectorStore()
print('Stats:', vs.get_stats())
results = vs.search('test query', top_k=2)
print('Results:', len(results))
"

# Test API
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your return policy?"}'
```

## 🐛 Troubleshooting

### "unstructured not installed"
```bash
pip install unstructured[pdf]
pip install pdfminer.six
```

### "ChromaDB collection already exists"
```python
import chromadb
client = chromadb.PersistentClient(path="./data/vector_store")
client.delete_collection("documents")
```

### Out of memory with large PDFs
- Reduce `chunk_size` in `DocumentIngestor`
- Process PDFs in smaller batches
- Use a machine with more RAM

## 📚 Additional Resources

- [ChromaDB Docs](https://docs.trychroma.com/)
- [Unstructured Docs](https://unstructured-io.github.io/unstructured/)
- [Sentence Transformers](https://www.sbert.net/)

## 🎓 Next Steps

1. **Add more documents** - Place PDFs in `data/sample_docs/` and re-run ingestion
2. **Fine-tune chunking** - Adjust `chunk_size` and `chunk_overlap` in `ingest_documents.py`
3. **Customize embeddings** - Switch to OpenAI embeddings in `.env`
4. **Deploy** - Use Docker for production deployment
