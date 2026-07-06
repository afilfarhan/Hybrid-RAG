# Hybrid RAG - Clean Architecture

A retrieval-augmented generation system with citations and low hallucination, built with a clean, modular architecture.

## 📁 Project Structure

```
Hybrid RAG/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create from .env.example)
├── ui/                   # Frontend UI (static files)
│   └── index.html       # Chatbot interface
├── services/             # Core RAG services
│   ├── base.py          # Service interfaces
│   └── inmemory.py      # In-memory fallback services
├── api/                  # REST API endpoints
│   └── endpoints.py     # API routes
├── data/                 # Sample documents and data
├── tests/                # Test suite
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Optional: OpenAI API (for production)
OPENAI_API_KEY=your_api_key_here

# Service configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

### 3. Run the Application

```bash
python app.py
```

Or with uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Chatbot

Open your browser and navigate to:
- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

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

## 📚 Sample Documents

The system comes with sample documents for testing:

- Product catalog (headphones, watches, earbuds)
- Return policy (30-day window, refund process)
- Shipping information (domestic & international)
- FAQ (payment methods, warranties, etc.)

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
| `VECTOR_STORE_TYPE` | `inmemory` or `chroma` | `inmemory` |

## 🧪 Testing

```bash
# Run tests
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with FastAPI and sentence-transformers
- Inspired by RAG best practices and research papers
