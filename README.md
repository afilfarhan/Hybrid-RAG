# Hybrid RAG

A production-ready, hybrid Retrieval-Augmented Generation (RAG) system that grounds LLM answers in a curated knowledge base with hybrid search, citations, comprehensive quality control, and multi-provider LLM support.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-79%20passed-brightgreen.svg)](https://pytest.org)

## Overview

Hybrid RAG combines dense and sparse retrieval methods to deliver accurate, source-cited answers from your own documents. Built with modularity and production-readiness in mind, it includes:

- **Hybrid Retrieval**: Combines dense vector search with BM25 keyword search
- **Structure-Aware Chunking**: Respects document structure for better context
- **Guardrails**: Ensures answer quality and prevents hallucinations
- **Multi-Provider LLM Support**: Switch between OpenAI, Anthropic, Google, NVIDIA NIM, and more
- **Caching**: Reduces latency and cost with Redis caching
- **Observability**: Full query tracing with LangSmith
- **Evaluation**: Comprehensive metrics for faithfulness, relevance, and hallucination detection
- **Admin Tools**: Document management and feedback collection

## Features

### 📥 Ingestion Pipeline
- File connectors (PDF, DOCX, TXT, MD, HTML)
- Web scraping connector
- API connector (REST and GraphQL)
- Scheduled ingestion
- Bilingual support (Arabic/English)

### 📝 Advanced Chunking
- Recursive character chunking
- Structure-aware chunking (respects headings)
- Semantic chunking (groups related content)
- Specialized chunkers (FAQ, Product records, tables)
- 300-800 token chunks with 10-20% overlap
- Preserve Q&A pairs and structured data intact

### 🔍 Hybrid Retrieval
- Dense vector search
- Sparse BM25 search
- Hybrid fusion with configurable weighting
- Metadata filtering (language, access level, doc type)
- Optional reranking for improved relevance

### 🤖 Generation with Guardrails
- Context-grounded responses with citations
- Confidence scoring
- Fallback responses for low-confidence queries
- Safety checks and compliance filtering
- Audit trail for all responses

### 🚀 Production-Ready
- Multi-provider LLM support (OpenAI, Anthropic, Google, NVIDIA NIM, Bedrock, Azure)
- Caching layer (Redis)
- Full observability (LangSmith)
- Feedback collection and analysis
- Comprehensive evaluation harness
- Async support for high throughput

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp config/.env.example .env
```

Edit `.env` with your API keys:

```env
# LLM Provider Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# NVIDIA NIM Example
# NVIDIA_NIM_API_KEY=nvapi-...

# Embedding Configuration
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Vector Store
VECTOR_STORE_TYPE=chroma

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Retrieval
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.7
HYBRID_RETRIEVAL_WEIGHT=0.7

# Caching
CACHE_ENABLED=true
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
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

## Multi-Provider LLM Support

Switch between different LLM providers with a single configuration change:

### OpenAI
```python
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Anthropic (Claude)
```python
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### Google Gemini
```python
LLM_PROVIDER=google
GOOGLE_API_KEY=...
```

### NVIDIA NIM
```python
LLM_PROVIDER=nvidia_nim
NVIDIA_NIM_API_KEY=nvapi-...
```

### AWS Bedrock
```python
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### Google Vertex AI
```python
LLM_PROVIDER=vertex_ai
VERTEX_PROJECT=your-gcp-project-id
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

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

**Response:**
```json
{
  "answer": "We offer a 30-day return policy...",
  "citations": [
    {
      "id": 1,
      "source": "products.pdf",
      "chunk_id": "chunk_123",
      "score": 0.95
    }
  ],
  "confidence": 0.95
}
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

#### Get Feedback Statistics
```bash
curl http://localhost:8000/feedback/stats
```

## Use Cases

### 1. Customer Support Knowledge Base
- Ingest product documentation and FAQs
- Provide accurate, cited answers to customer queries
- Collect feedback to improve responses over time

### 2. Internal Document Search
- Search through company policies, procedures, and manuals
- Support bilingual content (Arabic/English)
- Ensure compliance with data residency requirements

### 3. Legal & Compliance Assistant
- Query legal documents and regulations
- Include citations for audit trails
- Apply guardrails to prevent hallucinations

### 4. Product Recommendation Engine
- Search product catalog with structured data
- Provide detailed product information with sources
- Support multilingual customers

### 5. Research Assistant
- Query academic papers and technical documentation
- Extract structured information
- Track confidence scores for reliability assessment

## Project Structure

```
Hybrid RAG/
├── src/
│   ├── core/           # Core utilities, provider config, ID generation
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
├── tests/              # Tests (79 passing)
├── logs/               # Log files
├── requirements.txt    # Dependencies
├── main.py             # Entry point
└── README.md           # This file
```

## Configuration

Key configuration options (see `.env.example` for full list):

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | openai | LLM provider (openai, anthropic, google, nvidia_nim, bedrock, vertex_ai) |
| `OPENAI_API_KEY` | required | OpenAI API key |
| `OPENAI_MODEL` | gpt-4o-mini | LLM model to use |
| `EMBEDDING_PROVIDER` | openai | Embedding provider |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |
| `NVIDIA_NIM_API_KEY` | required | NVIDIA NIM API key (if using) |
| `ANTHROPIC_API_KEY` | required | Anthropic API key (if using) |
| `VECTOR_STORE_TYPE` | chroma | Vector store type |
| `CHUNK_SIZE` | 512 | Chunk size in tokens |
| `CHUNK_OVERLAP` | 50 | Chunk overlap |
| `RETRIEVAL_TOP_K` | 5 | Number of chunks to retrieve |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.7 | Minimum similarity score |
| `HYBRID_RETRIEVAL_WEIGHT` | 0.7 | Weight for hybrid retrieval |
| `CACHE_ENABLED` | true | Enable caching |
| `REDIS_URL` | redis://localhost:6379 | Redis connection URL |
| `LOG_LEVEL` | INFO | Logging level |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_provider_config.py -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Results:** 79 passed, 4 skipped

## Logging

Logs are written to `logs/app.log` and console (based on `LOG_LEVEL`).

## Provider Configuration Examples

See [PROVIDER_CONFIG.md](PROVIDER_CONFIG.md) for detailed examples of:
- Switching between providers
- Configuring API keys
- Using NVIDIA NIM models
- Embedding configuration
- Best practices

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with LangChain, LiteLLM, ChromaDB, and FastAPI
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM provider support
- Supports all models from [NVIDIA NIM](https://build.nvidia.com)
- Inspired by industry best practices for production RAG systems
- Designed for scalability and maintainability

## Getting Help

- Check the [PROVIDER_CONFIG.md](PROVIDER_CONFIG.md) for provider-specific setup
- Review the [RAG_Project_PRD.md](RAG_Project_PRD.md) for detailed requirements
- Run `pytest tests/ -v` to verify your setup
- Check logs for debugging information
