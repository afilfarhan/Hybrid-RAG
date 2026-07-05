# Hybrid RAG - System Architecture

## Overview

This document describes the architecture of the Hybrid RAG (Retrieval-Augmented Generation) system.

## System Components

### 1. Ingestion Pipeline
- **File Connector**: Supports PDF, DOCX, TXT, MD, HTML files
- **Web Connector**: Scrapes and ingests web content
- **API Connector**: Connects to REST and GraphQL APIs
- **Scheduler**: Automated ingestion on schedules

### 2. Preprocessing & Chunking
- **Recursive Chunker**: Splits text by characters with overlap
- **Structure-Aware Chunker**: Respects document structure (headings, sections)
- **Semantic Chunker**: Groups semantically related content
- **FAQ Chunker**: Keeps Q&A pairs together
- **Product Chunker**: Keeps product records atomic
- **Text Preprocessor**: Cleans and normalizes text

### 3. Embedding Service
- **OpenAI Embedding Service**: Uses OpenAI's embedding models
- **Vector Store**: ChromaDB for storing and retrieving embeddings

### 4. Retrieval System
- **Dense Retriever**: Uses vector embeddings for similarity search
- **Hybrid Retriever**: Combines dense and sparse (BM25) search
- **Reranker**: Improves ranking using cross-encoder scoring

### 5. Generation
- **OpenAI Generator**: Uses OpenAI's language models
- **Guardrails**: Ensures answer quality and safety

### 6. Caching
- **Redis Cache**: Caches frequent queries to reduce latency and cost

### 7. Tracing & Observability
- **LangSmith Tracer**: Traces queries through the pipeline for debugging and evaluation

### 8. Admin Tools
- **Document Manager**: Add, update, delete documents
- **Feedback Handler**: Collect and analyze user feedback

### 9. Evaluation
- **Faithfulness Evaluator**: Checks if answers are grounded in context
- **Relevance Evaluator**: Checks if answers address the question
- **Hallucination Evaluator**: Detects hallucinated information
- **Golden Test Set**: Predefined test cases for evaluation

## Data Flow

```
User Query
    ↓
Ingestion Pipeline (if new documents)
    ↓
Preprocessing & Chunking
    ↓
Embedding Service → Vector Store
    ↓
Retrieval (Dense/Hybrid + Reranker)
    ↓
Generation (with Guardrails)
    ↓
Response to User
```

## Configuration

Configuration is loaded from environment variables. See `.env.example` for available options.

## API Endpoints

- `GET /health`: Health check
- `POST /query`: Submit a query
- `POST /feedback`: Submit feedback
- `GET /status`: Get service status
- `GET /metrics`: Get service metrics

## Running the System

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your configuration:
```bash
cp config/.env.example .env
```

3. Run the system:
```bash
python main.py
```

4. Start the API server:
```bash
uvicorn src.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

## Testing

Run the sample document creation script:
```bash
python create_sample_docs.py
```

## Project Structure

```
Hybrid RAG/
├── src/
│   ├── core/           # Core utilities and configuration
│   ├── ingestion/      # Ingestion pipeline
│   ├── chunking/       # Preprocessing and chunking
│   ├── embedding/      # Embedding service and vector store
│   ├── retrieval/      # Retrieval system
│   ├── generation/     # Generation and guardrails
│   ├── cache/          # Caching layer
│   ├── tracing/        # Tracing and observability
│   ├── admin/          # Admin tools
│   ├── evaluation/     # Evaluation harness
│   ├── api/            # API endpoints
│   └── sample_data.py  # Sample data
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

## License

MIT License
