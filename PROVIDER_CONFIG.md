# LiteLLM Provider Configuration

This document explains how to configure and switch between different LLM providers in the Hybrid RAG system using LiteLLM.

## Supported Providers

| Provider | LLM Models | Embedding Models |
|----------|-----------|------------------|
| OpenAI | gpt-4o-mini, gpt-4o, gpt-4-turbo | text-embedding-3-small, text-embedding-3-large |
| Anthropic | claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus | N/A |
| Google Gemini | gemini-1.5-flash, gemini-1.5-pro | text-embedding-004 |
| Google Vertex AI | claude-sonnet-4, gemini-1.5-pro | text-embedding-004 |
| AWS Bedrock | anthropic.claude-3-5-sonnet, amazon.nova-micro | titan-embed-text-v1, cohere.embed-english-v3 |
| NVIDIA NIM | All models on build.nvidia.com | All embedding models on build.nvidia.com |
| Azure OpenAI | Custom deployment names | Custom deployment names |

## Quick Start

### Using Provider Configuration Module

```python
from src.core.provider_config import create_provider_config
from src.embedding.litellm_service import LiteLMEmbeddingService
from src.generation.litellm_generator import LiteLMGenerator

# Configure OpenAI
openai_config = create_provider_config(
    provider="openai",
    model_type="llm",
    model_name="gpt-4o-mini",
    api_key="your-api-key",
)

# Configure Anthropic
anthropic_config = create_provider_config(
    provider="anthropic",
    model_type="llm",
    model_name="claude-3-5-sonnet-20241022",
    api_key="your-api-key",
)

# Configure NVIDIA NIM
nvidia_nim_config = create_provider_config(
    provider="nvidia_nim",
    model_type="llm",
    model_name="meta/llama3-70b-instruct",
    api_key="nvapi-...",
)

# Use in services
generator = LiteLMGenerator(openai_config)
embedding_service = LiteLMEmbeddingService(openai_config)
```

### Using Environment Variables

```python
import os
from src.core.provider_config import create_provider_config

config = create_provider_config(
    provider="openai",
    model_type="llm",
    model_name="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)
```

## Switching Providers

### Method 1: Direct Configuration

```python
from src.core.provider_config import create_provider_config

# Start with OpenAI
openai_config = create_provider_config(
    provider="openai",
    model_type="llm",
    model_name="gpt-4o-mini",
    api_key="sk-...",
)

# Switch to Anthropic
anthropic_config = create_provider_config(
    provider="anthropic",
    model_type="llm",
    model_name="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",
)
```

### Method 2: Using switch_provider()

```python
from src.core.provider_config import switch_provider

# Start with OpenAI
config = create_provider_config("openai", "llm", "gpt-4o-mini", "sk-...")

# Switch to Anthropic (preserves other settings)
config = switch_provider(config, "anthropic", "llm", "claude-3-5-sonnet", "sk-ant-...")
```

## Available Models

### Get Available Providers
```python
from src.core.provider_config import get_available_providers

providers = get_available_providers()
# ['openai', 'anthropic', 'google', 'vertex_ai', 'bedrock']
```

### Get Available Models for a Provider
```python
from src.core.provider_config import get_available_models

openai_llm_models = get_available_models("openai", "llm")
# ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo']

openai_embedding_models = get_available_models("openai", "embedding")
# ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
```

## Model Naming Convention

LiteLLM uses the format: `<provider>/<model-name>`

| Provider | Model Prefix | Example |
|----------|-------------|---------|
| OpenAI | `openai/` | `openai/gpt-4o-mini` |
| Anthropic | `anthropic/` | `anthropic/claude-3-5-sonnet-20241022` |
| Google | (no prefix) | `gemini-1.5-flash` |
| Vertex AI | `vertex_ai/` | `vertex_ai/claude-sonnet-4` |
| Bedrock | `bedrock/` | `bedrock/anthropic.claude-3-5-sonnet-20241022-v1:0` |
| NVIDIA NIM | `nvidia_nim/` | `nvidia_nim/meta/llama3-70b-instruct` |
| Azure | `azure/` | `azure/your-deployment` |

When using the `provider_config` module, you only specify the provider and model name - the prefix is added automatically.

**Note for NVIDIA NIM:** All models on [build.nvidia.com](https://build.nvidia.com) are supported. Just specify the model name (e.g., `meta/llama3-70b-instruct`) and the `nvidia_nim/` prefix will be added automatically.

## Configuration Examples

### Complete RAG Setup with OpenAI
```python
from src.core.provider_config import create_provider_config
from src.embedding.litellm_service import LiteLMEmbeddingService
from src.generation.litellm_generator import LiteLMGenerator
from src.retrieval.dense import DenseRetriever
from src.chunking.structure import StructureAwareChunker

# Configure providers
embedding_config = create_provider_config(
    provider="openai",
    model_type="embedding",
    model_name="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

llm_config = create_provider_config(
    provider="openai",
    model_type="llm",
    model_name="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Initialize services
embedding_service = LiteLMEmbeddingService(embedding_config)
generator = LiteLMGenerator(llm_config)

# Initialize chunker (uses default embedding service if none provided)
chunker = StructureAwareChunker()
```

### Complete RAG Setup with Anthropic
```python
from src.core.provider_config import create_provider_config
from src.embedding.litellm_service import LiteLMEmbeddingService
from src.generation.litellm_generator import LiteLMGenerator

# Configure providers
embedding_config = create_provider_config(
    provider="openai",  # Use OpenAI for embeddings (Anthropic doesn't provide embedding API)
    model_type="embedding",
    model_name="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

llm_config = create_provider_config(
    provider="anthropic",
    model_type="llm",
    model_name="claude-3-5-sonnet-20241022",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Initialize services
embedding_service = LiteLMEmbeddingService(embedding_config)
generator = LiteLMGenerator(llm_config)
```

### AWS Bedrock Configuration
```python
from src.core.provider_config import create_provider_config

bedrock_config = create_provider_config(
    provider="bedrock",
    model_type="llm",
    model_name="anthropic.claude-3-5-sonnet-20241022-v1:0",
    extra_kwargs={
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "aws_region_name": "us-east-1",
    },
)
```

### Google Vertex AI Configuration
```python
from src.core.provider_config import create_provider_config

vertex_config = create_provider_config(
    provider="vertex_ai",
    model_type="llm",
    model_name="claude-sonnet-4",
    extra_kwargs={
        "vertex_project": "your-gcp-project-id",
        "vertex_location": "us-east5",
    },
)
```

## Provider Configuration Examples

### Complete RAG Setup with NVIDIA NIM
```python
from src.core.provider_config import create_provider_config
from src.embedding.litellm_service import LiteLMEmbeddingService
from src.generation.litellm_generator import LiteLMGenerator

# Configure providers
embedding_config = create_provider_config(
    provider="nvidia_nim",
    model_type="embedding",
    model_name="nvidia/nv-embedqa-e5-v5",
    api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
)

llm_config = create_provider_config(
    provider="nvidia_nim",
    model_type="llm",
    model_name="meta/llama3-70b-instruct",
    api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
)

# Initialize services
embedding_service = LiteLMEmbeddingService(embedding_config)
generator = LiteLMGenerator(llm_config)
```

## API Reference

### `create_provider_config(provider, model_type, model_name, api_key, extra_kwargs)`

Create a configuration dictionary for a specific provider.

**Parameters:**
- `provider` (str): Provider name ('openai', 'anthropic', 'google', 'vertex_ai', 'bedrock')
- `model_type` (str): Model type ('llm' or 'embedding')
- `model_name` (str, optional): Specific model name. Uses default if None.
- `api_key` (str, optional): API key for the provider
- `extra_kwargs` (dict, optional): Additional provider-specific kwargs

**Returns:**
- Dict: Configuration dictionary for LiteLLM

### `get_provider_model(provider, model_type, model_name)`

Get the full model name with provider prefix.

**Parameters:**
- `provider` (str): Provider name
- `model_type` (str): Model type ('llm' or 'embedding')
- `model_name` (str, optional): Specific model name

**Returns:**
- str: Full model name with provider prefix (e.g., 'openai/gpt-4o-mini')

### `switch_provider(current_config, new_provider, model_type, model_name, api_key)`

Switch provider while preserving other configuration.

**Parameters:**
- `current_config` (dict): Current configuration dictionary
- `new_provider` (str): New provider name
- `model_type` (str): Model type ('llm' or 'embedding')
- `model_name` (str, optional): Specific model name
- `api_key` (str, optional): New API key

**Returns:**
- Dict: Updated configuration dictionary

### `get_available_providers()`

Get list of supported providers.

**Returns:**
- List[str]: List of provider names

### `get_available_models(provider, model_type)`

Get list of available models for a provider.

**Parameters:**
- `provider` (str): Provider name
- `model_type` (str): Model type ('llm' or 'embedding')

**Returns:**
- List[str]: List of available model names

## Best Practices

1. **Use Environment Variables**: Never hardcode API keys in your code
2. **Provider Abstraction**: Use the `provider_config` module to abstract provider-specific details
3. **Test Switching**: Test your application with different providers to ensure compatibility
4. **Model Selection**: Choose models based on your needs:
   - Speed: gpt-4o-mini, claude-3-5-haiku, gemini-1.5-flash
   - Quality: gpt-4o, claude-3-opus, gemini-1.5-pro
   - Cost: Smaller models are cheaper for high-volume use
5. **NVIDIA NIM**: 
   - Get your API key from [build.nvidia.com](https://build.nvidia.com)
   - All models on the platform are supported
   - Use `nvidia_nim/` prefix for embeddings and LLMs
6. **Embedding Considerations**: Anthropic doesn't provide embedding API - use OpenAI, Google, or NVIDIA NIM for embeddings when using Claude

## Troubleshooting

### Error: "Model 'xxx' not available for provider 'yyy'"

Check the available models for your provider using `get_available_models(provider, model_type)`.

### Error: "Invalid provider: xxx"

Ensure you're using a supported provider: 'openai', 'anthropic', 'google', 'vertex_ai', 'bedrock'.

### API Key Issues

Make sure your API key is set correctly:
```python
import os
os.environ.get("YOUR_API_KEY")  # Returns None if not set
```

Or set it directly:
```python
config = create_provider_config(
    provider="openai",
    model_type="llm",
    model_name="gpt-4o-mini",
    api_key="sk-...",  # Direct API key (not recommended for production)
)
```

## Examples Directory

See `src/core/provider_examples.py` for more detailed examples including:
- All provider configurations
- Switching between providers
- Environment variable setup
- Model availability checks
