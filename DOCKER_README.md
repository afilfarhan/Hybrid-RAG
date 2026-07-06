# Hybrid RAG - Docker Quick Start Guide

## Prerequisites
- Docker Desktop installed and running
- OpenAI API key (or alternative provider)

## Quick Start

### 1. Clone and Setup
```powershell
cd "D:\Hybrid RAG"
```

### 2. Create Environment File
```powershell
Copy-Item .env.example .env
```

### 3. Edit `.env` and add your API key:
```env
OPENAI_API_KEY=sk-your_actual_openai_api_key_here
```

### 4. Start Services
```powershell
docker-compose up -d
```

### 5. Verify Services
```powershell
# Check all services are healthy
docker-compose ps

# View logs
docker-compose logs -f rag-api
```

### 6. Test the API
```powershell
# Health check
curl http://localhost:8000/health

# Query the RAG system
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query" -Method POST -ContentType "application/json" -Body '{"query":"What is PDPL?","top_k":5}'
```

## Available Services

| Service | Port | Description |
|---------|------|-------------|
| RAG API | 8000 | Main API server |
| Redis | 6379 | Cache server |
| ChromaDB | 8001 | Vector database |

## Stop Services
```powershell
docker-compose down
```

## Stop and Remove Volumes
```powershell
docker-compose down -v
```

## Troubleshooting

### Service not starting
```powershell
docker-compose logs rag-api
```

### Port already in use
Edit `docker-compose.yml` and change the port mappings:
```yaml
ports:
  - "8002:8000"  # Host:Container
```

### Clear cache
```powershell
docker-compose down -v
docker-compose up -d
```

## Using Alternative LLM Providers

Edit `.env` and replace `OPENAI_API_KEY` with your provider:

### Anthropic (Claude)
```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_MODEL=claude-3-5-sonnet-20241022
OPENAI_EMBEDDING_MODEL=claude-3-5-sonnet-20241022
```

### Google Gemini
```env
GOOGLE_API_KEY=your_google_key
OPENAI_MODEL=gemini-1.5-flash
OPENAI_EMBEDDING_MODEL=text-embedding-004
```

### NVIDIA NIM
```env
NVIDIA_API_KEY=your_nvidia_key
OPENAI_MODEL=nvidia_nim/meta/llama3-70b-instruct
OPENAI_EMBEDDING_MODEL=nvidia_nim/nvidia/nv-embed-v1
```

## Custom Configuration

Edit `.env` to customize:
- `CHUNK_SIZE`: Token size for document chunks (default: 512)
- `RETRIEVAL_TOP_K`: Number of chunks to retrieve (default: 5)
- `CORS_ORIGINS`: Allowed frontend origins
- `DATA_RESIDENCY`: Data residency for compliance (us, eu, sa)
