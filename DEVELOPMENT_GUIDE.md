# Hybrid RAG - Development Guide

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Redis (optional, for caching)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Hybrid\ RAG
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp config/.env.example .env
# Edit .env with your OpenAI API key
```

## Project Structure

```
Hybrid RAG/
├── src/
│   ├── core/           # Core utilities
│   ├── ingestion/      # Ingestion pipeline
│   ├── chunking/       # Chunking system
│   ├── embedding/      # Embedding service
│   ├── retrieval/      # Retrieval system
│   ├── generation/     # Generation module
│   ├── cache/          # Caching layer
│   ├── tracing/        # Tracing and observability
│   ├── admin/          # Admin tools
│   ├── evaluation/     # Evaluation harness
│   └── api/            # API endpoints
├── config/
│   └── .env.example    # Configuration template
├── data/
│   └── sample_docs/    # Sample documents
├── tests/              # Tests
├── logs/               # Log files
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## Running the System

### Start the API Server
```bash
uvicorn src.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
python run_tests.py
# or
pytest tests/ -v
```

### Create Sample Documents
```bash
python create_sample_docs.py
```

## Development Workflow

### 1. Add a New Feature

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Implement the feature following the existing patterns

3. Add tests for the new functionality

4. Update documentation as needed

5. Run tests:
```bash
python run_tests.py
```

6. Commit and push:
```bash
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

### 2. Fix a Bug

1. Create a new branch:
```bash
git checkout -b fix/bug-description
```

2. Identify and fix the bug

3. Add a test that reproduces the bug

4. Run tests to verify the fix

5. Commit and push

### 3. Add a New Chunker

1. Create a new file in `src/chunking/`
2. Implement the `BaseChunker` interface
3. Add tests in `tests/test_chunking.py`
4. Update `src/chunking/__init__.py`

### 4. Add a New Ingestion Connector

1. Create a new file in `src/ingestion/`
2. Implement the `BaseIngestionPipeline` interface
3. Add tests
4. Update `src/ingestion/__init__.py`

## Code Style

### Python Style
- Follow PEP 8
- Use type hints
- Keep functions focused and small
- Use meaningful variable names

### Documentation
- Add docstrings to all public classes and functions
- Update README.md for new features
- Add examples where helpful

### Testing
- Write unit tests for new functionality
- Test edge cases
- Use fixtures for common test data

## Debugging

### Enable Debug Logging
```env
LOG_LEVEL=DEBUG
```

### Check Vector Store
```bash
python -c "from src.embedding.vector_store import VectorStore; import asyncio; asyncio.run(VectorStore({'store_type': 'chroma'}).connect())"
```

### Test Embedding
```python
from src.embedding.openai_service import OpenAIEmbeddingService
import asyncio

async def test():
    service = OpenAIEmbeddingService({'api_key': 'your_key'})
    embedding = await service.embed("test")
    print(f"Embedding dimension: {len(embedding)}")

asyncio.run(test())
```

## Performance Optimization

### Caching
- Enable Redis caching
- Set appropriate TTL values
- Cache frequent queries

### Chunking
- Adjust chunk size based on your use case
- Use structure-aware chunking for documents with clear structure
- Consider semantic chunking for natural language content

### Retrieval
- Adjust `RETRIEVAL_TOP_K` based on accuracy needs
- Use metadata filtering to reduce search space
- Consider reranking for improved quality

## Deployment

### Docker
```bash
docker build -t hybrid-rag .
docker run -p 8000:8000 hybrid-rag
```

### Production
- Use a production-grade WSGI server (Gunicorn, Uvicorn with workers)
- Enable HTTPS
- Configure rate limiting
- Set up monitoring and alerting
- Use environment variables for secrets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Questions?

Check the documentation or open an issue.
