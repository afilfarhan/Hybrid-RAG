# Hybrid RAG - Changelog

## Version 1.0.0 (Current)

### Features
- **Ingestion Pipeline**
  - File connector (PDF, DOCX, TXT, MD, HTML)
  - Web connector for scraping
  - API connector (REST and GraphQL)
  - Scheduled ingestion

- **Chunking System**
  - Recursive character chunking
  - Structure-aware chunking (respects headings)
  - Semantic chunking (groups related content)
  - FAQ chunker (keeps Q&A pairs together)
  - Product chunker (keeps product records atomic)
  - Text preprocessor (cleans and normalizes text)

- **Embedding Service**
  - OpenAI embedding service
  - ChromaDB vector store integration

- **Retrieval System**
  - Dense vector search
  - Sparse BM25 search
  - Hybrid retrieval with configurable weighting
  - Metadata filtering
  - Optional reranking

- **Generation**
  - OpenAI generator with citations
  - Guardrails for quality control
  - Confidence scoring
  - Fallback responses

- **Caching**
  - Redis caching layer
  - Configurable TTL

- **Observability**
  - LangSmith tracing
  - Full query pipeline tracing

- **Admin Tools**
  - Document manager
  - Feedback handler

- **Evaluation**
  - Faithfulness evaluator
  - Relevance evaluator
  - Hallucination evaluator
  - Golden test set support

### API
- `/health` - Health check
- `/query` - Submit queries
- `/feedback` - Submit feedback
- `/status` - Get system status
- `/metrics` - Get service metrics

### Documentation
- `README.md` - Project overview
- `ARCHITECTURE.md` - System architecture
- `CONFIG_GUIDE.md` - Configuration guide
- `API_DOCUMENTATION.md` - API documentation
- `RAG_Project_PRD.md` - Original PRD

### Testing
- Unit tests
- Integration tests
- Sample document creation script

### Configuration
- Environment-based configuration
- Pydantic Settings for type safety
- Validation on startup

## Future Versions

### Version 1.1.0 (Planned)
- Multi-language support
- Advanced reranking models
- Query expansion
- Dynamic chunk size adjustment

### Version 1.2.0 (Planned)
- Multiple vector store support (Pinecone, Qdrant, Weaviate)
- Advanced metadata filtering
- Hybrid search optimization
- Distributed caching

### Version 2.0.0 (Planned)
- Custom embedding models
- Fine-tuning support
- Advanced guardrails
- Multi-tenant support
