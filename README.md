# Hybrid RAG - Clean Architecture

A retrieval-augmented generation system with citations and low hallucination, built with a clean, modular architecture.

## 📁 Project Structure

```
Hybrid RAG/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── start.bat              # Windows quick start script
├── setup.bat              # Windows setup script
├── ingest_documents.py    # PDF/document ingestion script
├── test_setup.py          # Setup verification script
├── README.md              # This file
├── SETUP_GUIDE.md         # Complete setup documentation
├── SETUP_CHROMADB.md      # ChromaDB-specific guide
├── PROVIDER_GUIDE.md      # LLM provider configuration guide
├── ui/                    # Frontend UI (static files)
│   └── index.html         # Chatbot interface
├── services/              # Core RAG services
│   ├── base.py            # Service interfaces
│   ├── inmemory.py        # In-memory fallback services
│   ├── litellm.py         # LiteLLM provider services (OpenAI, NVIDIA NIM, etc.)
│   └── chromadb_store.py  # ChromaDB vector store
├── api/                   # REST API endpoints
│   └── endpoints.py       # API routes
├── data/                  # Sample documents and data
│   ├── sample_docs/       # Place your PDFs here
│   └── vector_store/      # ChromaDB persists here
└── tests/                 # Test suite
```

## 🚀 Quick Start

### Option 1: Using the Setup Script (Windows)

```bash
# First-time setup
.\setup.bat

# Then start the server
.\start.bat

# Or manually start
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Manual Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Install PDF support
pip install unstructured[pdf]

# Start server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Direct Python

```bash
python app.py
```

## 🌐 Access Points

Once the server is running:

- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health
- **Query Endpoint**: http://localhost:8000/api/v1/query

## 🏗️ Architecture

### Core Services

The system is built on a clean architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                  │
│              REST endpoints and request handling        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   RAG Service                           │
│              Orchestrates all components                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Services Layer (Abstract)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Embedding   │  │  Vector      │  │  Retrieval   │ │
│  │  Service     │  │  Store       │  │  Service     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐                                       │
│  │  Generation  │                                       │
│  │  Service     │                                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Implementation Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ In-Memory       │  │ OpenAI/ChromaDB │              │
│  │ (Development)   │  │ (Production)    │              │
│  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### Service Interfaces

All services are defined as abstract base classes:

- **EmbeddingService**: Converts text to vector embeddings
- **VectorStore**: Stores and retrieves vectors by similarity
- **RetrievalService**: Combines dense and sparse retrieval
- **GenerationService**: Generates responses from contexts

### Fallback Mechanism

The system automatically falls back to in-memory services when production services are unavailable:

1. Try to initialize OpenAI embeddings
2. If failed, use in-memory sentence-transformers
3. Try to initialize ChromaDB
4. If failed, use in-memory numpy-based vector store

## 🌍 LLM Provider Configuration

The system supports multiple LLM providers through LiteLLM:

| Provider | API Key | Provider Name |
|----------|---------|---------------|
| OpenAI | `OPENAI_API_KEY` | `openai` |
| NVIDIA NIM | `NVIDIA_API_KEY` | `nvidia_nim` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` |
| Google Gemini | `GOOGLE_API_KEY` | `google` |
| AWS Bedrock | (auto-detected) | `bedrock` |

### Quick Start: NVIDIA NIM

1. Get API key from [https://build.nvidia.com/](https://build.nvidia.com/)
2. Update `.env`:
   ```env
   NVIDIA_API_KEY=nvapi-xxxx
   PROVIDER=nvidia_nim
   NVIDIA_LLM_MODEL=meta/llama3-70b-instruct
   NVIDIA_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
   ```
3. Restart the server

For detailed provider configuration, see [PROVIDER_GUIDE.md](PROVIDER_GUIDE.md).

## 📚 Sample Documents

The system comes with sample documents for testing:

- **Returns Policy** - 30-day window, refund process
- **Shipping Info** - Domestic & international shipping
- **FAQ** - Payment methods, warranties, account questions

### Adding Your Own Documents

1. Place PDF, Markdown, or text files in `data/sample_docs/`
2. Run: `python ingest_documents.py`
3. Start querying your documents!

## 📄 PDF & Document Ingestion

### Supported Formats

| Format | Extension | Required Package |
|--------|-----------|------------------|
| PDF | .pdf | `unstructured[pdf]`, `markdown`, `beautifulsoup4` |
| Markdown | .md, .markdown | `unstructured`, `markdown` |
| Text | .txt | Built-in |
| HTML | .html | `unstructured`, `beautifulsoup4` |

### Quick Ingestion

```bash
# Install all required dependencies
pip install unstructured[pdf] unstructured[md] markdown beautifulsoup4 pdfminer.six pdfplumber

# Ingest all documents in data/sample_test_docs/
python ingest_documents.py
```

### Important: Understanding the Vector Store

The vector store **accumulates** documents - it does not replace them. When you run `ingest_documents.py`:

1. **Existing documents are kept** - Your new PDFs are added to the existing collection
2. **Sample docs are also loaded** - The API server loads sample documents on startup
3. **Search returns the best matches** - Results depend on query similarity

### To Start Fresh

If you want to clear all documents and start over:

```python
# Run this in Python
from services.chromadb_store import ChromaDBVectorStore
vs = ChromaDBVectorStore()
vs.clear()
print("Vector store cleared!")
```

Or delete the vector store directory:

```bash
rm -rf data/vector_store
```

Then run ingestion again:

```bash
python ingest_documents.py
```

### Response Quality

The system automatically cleans PDF output by:
- Removing page markers and special characters
- Filtering low-quality chunks
- Generating concise, focused answers
- Showing only the most relevant 3 chunks per query

### Manual Ingestion

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
```

## 🌐 Frontend UI

The chatbot interface features:

- **Modern Design**: Gradient background, clean typography
- **Real-time Chat**: Message bubbles with animations
- **Citations Display**: Shows source documents for each answer
- **Metadata**: Shows model and context count
- **Responsive**: Works on desktop and mobile

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `OPENAI_API_KEY` | OpenAI API key (optional) | - |
| `CHROMA_PERSIST_PATH` | ChromaDB persist path | `./data/vector_store` |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection name | `documents` |
| `CHUNK_SIZE` | Document chunk size | `512` |
| `CHUNK_OVERLAP` | Chunk overlap | `51` |

### Supported Vector Stores

| Type | Service | Use Case |
|------|---------|----------|
| `inmemory` | Sentence Transformers + NumPy | Development, testing |
| `chroma` | ChromaDB + Sentence Transformers | Production, persistent storage |

## 🧪 Testing

```bash
# Run tests
python test_setup.py

# Run with pytest
pytest tests/

# Run with coverage
pytest --cov=services --cov=api tests/
```

## 📖 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### POST `/api/v1/query`
Process a query through the RAG system.

**Request:**
```json
{
  "query": "What is your return policy?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "We offer a 30-day return policy...",
  "citations": [...],
  "confidence": 0.92,
  "metadata": {...}
}
```

## 🚢 Deployment

### Docker

```bash
docker build -t hybrid-rag .
docker run -p 8000:8000 hybrid-rag
```

### Production

1. Set `OPENAI_API_KEY` in `.env`
2. Set `VECTOR_STORE_TYPE=chroma`
3. Configure Redis for caching
4. Run with: `uvicorn app:app --host 0.0.0.0 --port 8000`

### Cloud Platforms

- **AWS**: Use EC2 or Lambda with EFS for vector store
- **Azure**: Use App Service with File Storage
- **GCP**: Use Cloud Run with Cloud Storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_setup.py`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with FastAPI and sentence-transformers
- Inspired by RAG best practices and research papers
- Uses ChromaDB for production vector storage

## 📚 Additional Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions
- **[SETUP_CHROMADB.md](SETUP_CHROMADB.md)** - ChromaDB-specific guide
- **[README_SETUP.md](README_SETUP.md)** - Quick start guide

## 🐛 Troubleshooting

### "unstructured not installed"
```bash
pip install unstructured[pdf]
```

### "ChromaDB collection already exists"
```python
import chromadb
client = chromadb.PersistentClient(path="./data/vector_store")
client.delete_collection("documents")
```

### Slow embedding
- Use a GPU-enabled machine
- Reduce batch size
- Use a smaller embedding model

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
