# Hybrid RAG - API Documentation

## Overview

The Hybrid RAG API provides endpoints for querying the RAG system, submitting feedback, and checking system status.

## Quick Start

### Start the API Server

```bash
uvicorn src.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### Health Check

Check if the service is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "vector_store": {"status": "healthy"},
    "embedding_service": {"status": "healthy"},
    "cache": {"status": "healthy"},
    "tracer": {"status": "healthy"}
  }
}
```

### Query

Submit a query to the RAG system.

**Endpoint:** `POST /query`

**Request Body:**
```json
{
  "query": "What is your return policy?",
  "metadata_filter": {"source_type": "file"},
  "top_k": 5,
  "include_citations": true,
  "stream": false
}
```

**Response:**
```json
{
  "query": "What is your return policy?",
  "answer": "We offer a 30-day return policy for all products...",
  "citations": [
    {
      "source_id": 1,
      "reference": "[Source 1]"
    }
  ],
  "context": [
    {
      "id": "doc_abc123",
      "text": "We offer a 30-day return policy...",
      "metadata": {"source_type": "file", "source": "returns_policy.md"},
      "score": 0.95,
      "rank": 1
    }
  ],
  "metadata": {
    "trace_id": "uuid-here",
    "confidence": 0.92,
    "hallucination_score": 0.05,
    "usage": {
      "prompt_tokens": 150,
      "completion_tokens": 50,
      "total_tokens": 200
    }
  }
}
```

### Submit Feedback

Submit feedback for a query.

**Endpoint:** `POST /feedback`

**Request Body:**
```json
{
  "query": "What is your return policy?",
  "answer": "We offer a 30-day return policy...",
  "score": 5,
  "comment": "Excellent answer!",
  "trace_id": "uuid-here"
}
```

**Response:**
```json
{
  "status": "success",
  "feedback_id": 0
}
```

### Get Status

Get system status.

**Endpoint:** `GET /status`

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "vector_store": {"status": "healthy"},
    "embedding_service": {"status": "healthy"},
    "cache": {"status": "healthy"},
    "tracer": {"status": "healthy"}
  }
}
```

### Get Metrics

Get system metrics.

**Endpoint:** `GET /metrics`

**Response:**
```json
{
  "vector_store_count": 100,
  "cache_enabled": true,
  "tracer_enabled": true
}
```

## Python Client

You can use the RAG service directly in Python:

```python
import asyncio
from src.api.rag_service import RAGService

async def main():
    config = {
        'openai_api_key': 'your_api_key',
        # ... other config
    }
    
    rag_service = RAGService(config)
    await rag_service.initialize()
    
    result = await rag_service.query(
        query="What is your return policy?",
        top_k=5
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Citations: {result['citations']}")

asyncio.run(main())
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error
- `503`: Service not initialized

**Error Response:**
```json
{
  "detail": "Error message"
}
```

## Authentication

The API does not include authentication by default. Add authentication middleware as needed:

```python
from fastapi import Depends, HTTPException, Header

async def verify_token(x_api_key: str = Header(None)):
    if x_api_key != "your_secret_key":
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/query", dependencies=[Depends(verify_token)])
async def query(data: Dict[str, Any]):
    # ... query logic
```

## Rate Limiting

Add rate limiting middleware:

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/query")
@limiter.limit("100/minute")
async def query(data: Dict[str, Any]):
    # ... query logic
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

See `kubernetes/` directory for deployment manifests (not included in this repo).

## Troubleshooting

### Service Not Initialized

If you see "Service not initialized", ensure the API server started correctly and the RAG service initialized:

```bash
curl http://localhost:8000/health
```

### OpenAI API Errors

Check your OpenAI API key and ensure you have credits:

```bash
echo $OPENAI_API_KEY
```

### Vector Store Errors

Ensure the vector store directory exists and is writable:

```bash
mkdir -p ./data/chroma
chmod 755 ./data/chroma
```
