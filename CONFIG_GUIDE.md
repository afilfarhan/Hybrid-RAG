# Hybrid RAG - Configuration Guide

## Environment Variables

Create a `.env` file in the project root with the following variables:

### OpenAI Configuration
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### Vector Store Configuration
```env
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=hybrid_rag_docs
```

### Embedding Settings
```env
EMBEDDING_DIMENSION=3072
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### Retrieval Settings
```env
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.7
HYBRID_RETRIEVAL_WEIGHT=0.7
```

### Generation Settings
```env
TEMPERATURE=0.2
MAX_TOKENS=1000
SYSTEM_PROMPT=You are a helpful assistant...
```

### Caching
```env
CACHE_ENABLED=true
CACHE_TTL=3600
REDIS_URL=redis://localhost:6379
```

### Logging
```env
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Security
```env
SECURITY_ENABLED=true
PII_REDACTION=true
ACCESS_CONTROL_ENABLED=true
```

### Evaluation
```env
EVALUATION_ENABLED=true
EVALUATION_METRICS=["faithfulness", "relevance", "hallucination_rate"]
```

## Configuration File

The system loads configuration from environment variables using Pydantic Settings. See `src/core/config.py` for all available options.

## Example .env File

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=hybrid_rag_docs

# Embedding Settings
EMBEDDING_DIMENSION=3072
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Retrieval Settings
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.7
HYBRID_RETRIEVAL_WEIGHT=0.7

# Generation Settings
TEMPERATURE=0.2
MAX_TOKENS=1000
SYSTEM_PROMPT=You are a helpful assistant that answers questions based solely on the provided context. Always cite your sources and say 'I don't have information on that' if the answer isn't in the context.

# Caching
CACHE_ENABLED=true
CACHE_TTL=3600
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Security
SECURITY_ENABLED=true
PII_REDACTION=true
ACCESS_CONTROL_ENABLED=true

# Evaluation
EVALUATION_ENABLED=true
EVALUATION_METRICS=["faithfulness", "relevance", "hallucination_rate"]
```

## Dynamic Configuration

You can also pass configuration dynamically when initializing components:

```python
from src.embedding.openai_service import OpenAIEmbeddingService

config = {
    'api_key': 'your_api_key',
    'model_name': 'text-embedding-3-large',
    'dimension': 3072
}

service = OpenAIEmbeddingService(config)
```

## Configuration Validation

The system validates configuration on startup. Invalid configurations will raise errors:

- `OPENAI_API_KEY` must be set
- `vector_store_type` must be one of: chroma, pinecone, qdrant, weaviate
- `log_level` must be a valid Python logging level
