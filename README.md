# Hybrid RAG

A Retrieval-Augmented Generation (RAG) system that grounds LLM answers in a curated knowledge base with hybrid search, citations, and comprehensive quality control.

## Overview

Hybrid RAG combines dense and sparse retrieval methods to deliver accurate, source-cited answers from your own documents. Built with modularity and production-readiness in mind, it includes:

- **Hybrid Retrieval**: Combines dense vector search with BM25 keyword search
- **Structure-Aware Chunking**: Respects document structure for better context
- **Guardrails**: Ensures answer quality and prevents hallucinations
- **Caching**: Reduces latency and cost with Redis caching
- **Observability**: Full query tracing with LangSmith
- **Evaluation**: Comprehensive metrics for faithfulness, relevance, and hallucination detection
- **Admin Tools**: Document management and feedback collection

## Features

✅ **Ingestion Pipeline**
- File connectors (PDF, DOCX, TXT, MD, HTML)
- Web scraping connector
- API connector (REST and GraphQL)
- Scheduled ingestion

✅ **Advanced Chunking**
- Recursive character chunking
- Structure-aware chunking (respects headings)
- Semantic chunking (groups related content)
- Specialized chunkers (FAQ, Product records)

✅ **Hybrid Retrieval**
- Dense vector search
- Sparse BM25 search
- Hybrid fusion with configurable weighting
- Metadata filtering
- Optional reranking

✅ **Generation with Guardrails**
- Context-grounded responses
- Source citations
- Confidence scoring
- Fallback responses for low-confidence queries
- Safety checks

✅ **Production-Ready**
- Caching layer (Redis)
- Full observability (LangSmith)
- Feedback collection
- Comprehensive evaluation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp config/.env.example .env
```

Edit `.env` with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### 3. Create Sample Documents

```bash
python create_sample_docs.py
```

### 4. Run the System

```bash
python main.py
```

### 5. Start the API Server

```bash
uvicorn src.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your return policy?"}'
```

#### Submit Feedback
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your return policy?",
    "answer": "We offer a 30-day return policy...",
    "score": 5,
    "comment": "Excellent answer!"
  }'
```

## Project Structure

```
Hybrid RAG/
├── src/
│   ├── core/           # Core utilities and configuration
│   ├── ingestion/      # Ingestion pipeline (file, web, API)
│   ├── chunking/       # Preprocessing and chunking
│   ├── embedding/      # Embedding service and vector store
│   ├── retrieval/      # Retrieval system (dense, hybrid, reranker)
│   ├── generation/     # Generation and guardrails
│   ├── cache/          # Caching layer (Redis)
│   ├── tracing/        # Tracing and observability (LangSmith)
│   ├── admin/          # Admin tools (document manager, feedback)
│   ├── evaluation/     # Evaluation harness
│   └── api/            # API endpoints
├── config/
│   └── .env.example    # Configuration template
├── data/
│   └── sample_docs/    # Sample documents
├── tests/              # Tests
├── logs/               # Log files
├── requirements.txt    # Dependencies
├── main.py             # Entry point
└── README.md           # This file
```

## Configuration

Key configuration options (see `.env.example` for full list):

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `OPENAI_MODEL` | gpt-4o | LLM model to use |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-large | Embedding model |
| `VECTOR_STORE_TYPE` | chroma | Vector store type |
| `CHUNK_SIZE` | 512 | Chunk size in tokens |
| `CHUNK_OVERLAP` | 50 | Chunk overlap |
| `RETRIEVAL_TOP_K` | 5 | Number of chunks to retrieve |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.7 | Minimum similarity score |
| `HYBRID_RETRIEVAL_WEIGHT` | 0.7 | Weight for hybrid retrieval |
| `CACHE_ENABLED` | true | Enable caching |
| `LOG_LEVEL` | INFO | Logging level |

## Running Tests

```bash
pytest tests/ -v
```

## Logging

Logs are written to `logs/app.log` and console (based on `LOG_LEVEL`).

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with LangChain, OpenAI, ChromaDB, and FastAPI
- Inspired by industry best practices for production RAG systems
- Designed for scalability and maintainability
