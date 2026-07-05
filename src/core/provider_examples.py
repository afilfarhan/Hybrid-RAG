"""
Hybrid RAG - Provider Configuration Examples

This file demonstrates how to configure different LLM providers using LiteLLM.
Each provider has its own configuration block that can be used directly.
"""

from src.core.provider_config import (
    create_provider_config,
    switch_provider,
    get_available_providers,
    get_available_models,
)

# =============================================================================
# Provider Configuration Examples
# =============================================================================

# 1. OpenAI Configuration
# -----------------------
OPENAI_CONFIG = create_provider_config(
    provider="openai",
    model_type="llm",
    model_name="gpt-4o-mini",  # or gpt-4o, gpt-4-turbo
    api_key="your-openai-api-key",
)

OPENAI_EMBEDDING_CONFIG = create_provider_config(
    provider="openai",
    model_type="embedding",
    model_name="text-embedding-3-small",  # or text-embedding-3-large
    api_key="your-openai-api-key",
)

# 2. Anthropic Configuration
# --------------------------
ANTHROPIC_CONFIG = create_provider_config(
    provider="anthropic",
    model_type="llm",
    model_name="claude-3-5-sonnet-20241022",  # or claude-3-5-haiku-20241022, claude-3-opus-20240229
    api_key="your-anthropic-api-key",
)

# 3. Google Gemini Configuration
# ------------------------------
GOOGLE_CONFIG = create_provider_config(
    provider="google",
    model_type="llm",
    model_name="gemini-1.5-flash",  # or gemini-1.5-pro
    api_key="your-google-api-key",
)

GOOGLE_EMBEDDING_CONFIG = create_provider_config(
    provider="google",
    model_type="embedding",
    model_name="models/text-embedding-004",
    api_key="your-google-api-key",
)

# 4. Google Vertex AI Configuration
# ---------------------------------
VERTEX_AI_CONFIG = create_provider_config(
    provider="vertex_ai",
    model_type="llm",
    model_name="claude-sonnet-4",  # or gemini-1.5-pro
    extra_kwargs={
        "vertex_project": "your-gcp-project-id",
        "vertex_location": "us-east5",
    },
)

# 5. AWS Bedrock Configuration
# ----------------------------
BEDROCK_CONFIG = create_provider_config(
    provider="bedrock",
    model_type="llm",
    model_name="anthropic.claude-3-5-sonnet-20241022-v1:0",
    extra_kwargs={
        "aws_access_key_id": "your-aws-access-key",
        "aws_secret_access_key": "your-aws-secret-key",
        "aws_region_name": "us-east-1",
    },
)

# 6. NVIDIA NIM Configuration
# ---------------------------
NVIDIA_NIM_CONFIG = create_provider_config(
    provider="nvidia_nim",
    model_type="llm",
    model_name="meta/llama3-70b-instruct",  # or any model from build.nvidia.com
    api_key="nvapi-...",  # Get from https://build.nvidia.com/
)

NVIDIA_NIM_EMBEDDING_CONFIG = create_provider_config(
    provider="nvidia_nim",
    model_type="embedding",
    model_name="nvidia/nv-embedqa-e5-v5",
    api_key="nvapi-...",
)

# 7. Azure OpenAI Configuration
# -----------------------------
AZURE_CONFIG = create_provider_config(
    provider="azure",
    model_type="llm",
    model_name="your-deployment-name",
    base_url="https://your-resource.openai.azure.com/",
    extra_kwargs={
        "api_version": "2024-02-15-preview",
    },
)

# =============================================================================
# Usage Examples
# =============================================================================

# Example 1: Using OpenAI
# -----------------------
"""
from src.embedding.litellm_service import LiteLMEmbeddingService
from src.generation.litellm_generator import LiteLMGenerator

# Configure services
embedding_service = LiteLMEmbeddingService(OPENAI_EMBEDDING_CONFIG)
generator = LiteLMGenerator(OPENAI_CONFIG)
"""

# Example 2: Using NVIDIA NIM
# ----------------------------
"""
from src.embedding.litellm_service import LiteLMEmbeddingService
from src.generation.litellm_generator import LiteLMGenerator

# Configure services with NVIDIA NIM
embedding_service = LiteLMEmbeddingService(NVIDIA_NIM_EMBEDDING_CONFIG)
generator = LiteLMGenerator(NVIDIA_NIM_CONFIG)
"""

# Example 2: Switching from OpenAI to Anthropic
# ----------------------------------------------
"""
# Start with OpenAI
config = create_provider_config("openai", "llm", "gpt-4o-mini", "sk-...")

# Switch to Anthropic
config = switch_provider(config, "anthropic", "llm", "claude-3-5-sonnet", "sk-ant-...")
"""

# Example 3: Check available providers and models
# ------------------------------------------------
"""
providers = get_available_providers()
print("Available providers:", providers)

openai_models = get_available_models("openai", "llm")
print("OpenAI LLM models:", openai_models)

anthropic_models = get_available_models("anthropic", "llm")
print("Anthropic LLM models:", anthropic_models)
"""

# Example 4: Using with different providers
# ------------------------------------------
"""
# OpenAI
openai_config = create_provider_config("openai", "llm", "gpt-4o-mini", "sk-...")
generator_openai = LiteLMGenerator(openai_config)

#Anthropic
anthropic_config = create_provider_config("anthropic", "llm", "claude-3-5-sonnet", "sk-ant-...")
generator_anthropic = LiteLMGenerator(anthropic_config)

# Google
google_config = create_provider_config("google", "llm", "gemini-1.5-flash", "your-key...")
generator_google = LiteLMGenerator(google_config)
"""

# =============================================================================
# LiteLLM Model Naming Convention
# =============================================================================

"""
LiteLLM uses the format: <provider>/<model-name>

Provider prefixes:
- openai: openai/gpt-4o-mini
- anthropic: anthropic/claude-3-5-sonnet-20241022
- google: gemini-1.5-flash (no prefix for some models)
- vertex_ai: vertex_ai/claude-sonnet-4
- bedrock: bedrock/anthropic.claude-3-5-sonnet-20241022-v1:0
- azure: azure/<deployment-name>

When using the provider_config module, you only need to specify the provider
and model name, and the module handles the prefix automatically.
"""

# =============================================================================
# Environment Variable Configuration
# =============================================================================

"""
For production, use environment variables instead of hardcoding API keys:

import os
from src.core.provider_config import create_provider_config

OPENAI_CONFIG = create_provider_config(
    provider="openai",
    model_type="llm",
    model_name="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

ANTHROPIC_CONFIG = create_provider_config(
    provider="anthropic",
    model_type="llm",
    model_name="claude-3-5-sonnet-20241022",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
"""
