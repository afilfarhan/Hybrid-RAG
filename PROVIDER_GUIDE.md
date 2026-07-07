# Hybrid RAG - Provider Configuration Guide

This guide shows how to switch between different LLM providers (OpenAI, NVIDIA NIM, Anthropic, Google, etc.)

## Supported Providers

| Provider | API Key Env Var | Provider Name |
|----------|-----------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `openai` |
| NVIDIA NIM | `NVIDIA_API_KEY` | `nvidia_nim` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` |
| Google Gemini | `GOOGLE_API_KEY` | `google` |
| AWS Bedrock | (auto-detected) | `bedrock` |
| Azure OpenAI | `AZURE_API_KEY` | `azure` |

## Quick Start: NVIDIA NIM

### Step 1: Get NVIDIA API Key

1. Sign up at [https://build.nvidia.com/](https://build.nvidia.com/)
2. Get your API key from the dashboard
3. NVIDIA NIM offers free tier with 10K tokens/day

### Step 2: Update `.env`

```env
# Comment out OpenAI key
# OPENAI_API_KEY=your_openai_key

# Add NVIDIA NIM key
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx

# Set provider
PROVIDER=nvidia_nim

# Optional: Set specific models (model names without provider prefix)
NVIDIA_LLM_MODEL=llama3-70b-instruct
NVIDIA_EMBEDDING_MODEL=nv-embedqa-e5-v5
```

### Step 3: Restart Server

```bash
python app.py
```

## Provider-Specific Configuration

### NVIDIA NIM

```env
PROVIDER=nvidia_nim
NVIDIA_API_KEY=your_api_key

# Available LLM models:
# - meta/llama3-70b-instruct
# - meta/llama3-8b-instruct
# - mistralai/mistral-7b-instruct
# - mistralai/mixtral-8x7b-instruct
# - google/gemma-7b
# - microsoft/phi-3-mini-4k-instruct

# Available Embedding models:
# - nvidia/nv-embedqa-e5-v5
# - nvidia/nv-embedqa-mistral-7b-v2
# - nvidia/llama-3-embedding
```

### Anthropic

```env
PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key

# Available models:
# - claude-3-5-sonnet-20241022
# - claude-3-5-haiku-20241022
# - claude-3-opus-20240229
```

### Google Gemini

```env
PROVIDER=google
GOOGLE_API_KEY=your_api_key

# Available models:
# - gemini-1.5-flash
# - gemini-1.5-pro
# - gemini-2.0-flash-exp
```

### AWS Bedrock

```env
PROVIDER=bedrock
# AWS credentials are auto-detected from environment
# (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)

# Available models:
# - anthropic.claude-3-5-sonnet-20241022-v1:0
# - amazon.nova-micro-v1:0
```

## Advanced: Custom Base URL

For self-hosted LLMs or custom endpoints:

```env
PROVIDER=openai
OPENAI_API_KEY=your_api_key
BASE_URL=https://your-custom-endpoint.com/v1
```

## Testing Your Configuration

After updating `.env`, restart the server and check the logs:

```bash
python app.py
```

You should see:
```
INFO - Generator initialized with model: nvidia_nim/meta/llama3-70b-instruct
INFO - Embedding service initialized with model: nvidia/nv-embedqa-e5-v5
```

## Troubleshooting

### Error: "Invalid provider"
- Make sure `PROVIDER` is set to one of the supported providers
- Check for typos in the provider name

### Error: "API key not found"
- Make sure the correct API key environment variable is set
- Restart the server after updating `.env`

### Error: "Model not found"
- Check that the model name is correct for your provider
- Use the model names from the provider config table above

## LiteLLM Support

This system uses [LiteLLM](https://docs.litellm.ai/) which supports 100+ LLM providers. For other providers, check the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).
