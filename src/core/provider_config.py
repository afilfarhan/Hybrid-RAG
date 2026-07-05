"""
Hybrid RAG - LiteLLM provider configuration
Supports easy switching between LLM providers (OpenAI, Anthropic, Google, Bedrock, NVIDIA NIM, etc.)
"""

from typing import Any, Dict, List, Optional


# Provider model naming prefixes
PROVIDER_PREFIXES = {
    "openai": "openai/",
    "anthropic": "anthropic/",
    "google": "gemini/",  # Google Gemini
    "vertex_ai": "vertex_ai/",  # Google Vertex AI
    "bedrock": "bedrock/",  # AWS Bedrock
    "azure": "azure/",  # Azure OpenAI
    "nvidia_nim": "nvidia_nim/",  # NVIDIA NIM
}

# Provider-specific model examples
PROVIDER_MODELS = {
    "openai": {
        "embedding": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        "llm": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "o3-mini"],
    },
    "anthropic": {
        "embedding": [],  # Anthropic doesn't provide embedding API
        "llm": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
    },
    "google": {
        "embedding": ["models/text-embedding-004"],
        "llm": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
    },
    "vertex_ai": {
        "embedding": ["text-embedding-004"],
        "llm": ["claude-sonnet-4", "claude-opus-4-0", "gemini-1.5-pro"],
    },
    "bedrock": {
        "embedding": ["amazon.titan-embed-text-v1", "cohere.embed-english-v3"],
        "llm": ["anthropic.claude-3-5-sonnet-20241022-v1:0", "amazon.nova-micro-v1:0"],
    },
    "nvidia_nim": {
        "embedding": [
            "nvidia/nv-embedqa-e5-v5",
            "nvidia/nv-embedqa-mistral-7b-v2",
            "nvidia/llama-3-embedding",
            "nvidia/nemo-2-0-instruct",
        ],
        "llm": [
            "nvidia/nemotron-4-340b-reward",
            "meta/llama3-70b-instruct",
            "meta/llama3-8b-instruct",
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "google/gemma-7b",
            "microsoft/phi-3-mini-4k-instruct",
        ],
    },
}

# Default models for each provider
DEFAULT_MODELS = {
    "openai": {"embedding": "text-embedding-3-small", "llm": "gpt-4o-mini"},
    "anthropic": {"embedding": None, "llm": "claude-3-5-sonnet-20241022"},
    "google": {"embedding": "models/text-embedding-004", "llm": "gemini-1.5-flash"},
    "vertex_ai": {"embedding": "text-embedding-004", "llm": "claude-sonnet-4"},
    "bedrock": {"embedding": "amazon.titan-embed-text-v1", "llm": "anthropic.claude-3-5-sonnet-20241022-v1:0"},
    "nvidia_nim": {"embedding": "nvidia/nv-embedqa-e5-v5", "llm": "meta/llama3-70b-instruct"},
}


def get_provider_model(provider: str, model_type: str = "llm", model_name: Optional[str] = None) -> str:
    """
    Get the full model name with provider prefix.
    
    Args:
        provider: Provider name (openai, anthropic, google, vertex_ai, bedrock, nvidia_nim)
        model_type: Type of model (llm or embedding)
        model_name: Specific model name (uses default if None)
    
    Returns:
        Full model name with provider prefix (e.g., "openai/gpt-4o-mini" or "nvidia_nim/meta/llama3-70b-instruct")
    
    Raises:
        ValueError: If provider or model_type is invalid
    
    Note:
        For NVIDIA NIM, all models on build.nvidia.com are supported.
        Just specify the model name without the nvidia_nim/ prefix - it will be added automatically.
    """
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Invalid provider: {provider}. Available: {list(PROVIDER_MODELS.keys())}")
    
    if model_type not in ["llm", "embedding"]:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'llm' or 'embedding'")
    
    available_models = PROVIDER_MODELS[provider][model_type]
    
    if not available_models:
        raise ValueError(f"Provider '{provider}' does not support {model_type} models")
    
    if model_name is None:
        model_name = DEFAULT_MODELS[provider][model_type]
    
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not available for provider '{provider}'. Available: {available_models}")
    
    prefix = PROVIDER_PREFIXES.get(provider, "")
    return f"{prefix}{model_name}"


def get_available_providers() -> List[str]:
    """Get list of supported providers."""
    return list(PROVIDER_MODELS.keys())


def get_available_models(provider: str, model_type: str = "llm") -> List[str]:
    """Get list of available models for a provider."""
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Invalid provider: {provider}")
    return PROVIDER_MODELS[provider][model_type]


def create_provider_config(
    provider: str,
    model_type: str = "llm",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for a specific provider.
    
    Args:
        provider: Provider name (openai, anthropic, google, vertex_ai, bedrock, nvidia_nim)
        model_type: Model type (llm or embedding)
        model_name: Specific model name (uses default if None)
        api_key: API key for the provider
        extra_kwargs: Additional provider-specific kwargs
    
    Returns:
        Configuration dictionary for LiteLLM
    
    Example:
        >>> config = create_provider_config(
        ...     provider="nvidia_nim",
        ...     model_type="llm",
        ...     model_name="meta/llama3-70b-instruct",
        ...     api_key="nvapi-..."
        ... )
        >>> # Use with LiteLMGenerator(config)
    """
    full_model_name = get_provider_model(provider, model_type, model_name)
    
    config = {
        "model_name": full_model_name,
    }
    
    if api_key:
        config["api_key"] = api_key
    
    if extra_kwargs:
        config.update(extra_kwargs)
    
    return config


def switch_provider(
    current_config: Dict[str, Any],
    new_provider: str,
    model_type: str = "llm",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Switch provider while preserving other configuration.
    
    Args:
        current_config: Current configuration dictionary
        new_provider: New provider name
        model_type: Model type (llm or embedding)
        model_name: Specific model name (uses default if None)
        api_key: New API key
    
    Returns:
        Updated configuration dictionary
    
    Example:
        >>> # Start with OpenAI
        >>> config = create_provider_config("openai", "llm", "gpt-4o-mini", "sk-...")
        >>> # Switch to Anthropic
        >>> config = switch_provider(config, "anthropic", "llm", "claude-3-5-sonnet", "sk-ant-...")
    """
    new_config = create_provider_config(
        provider=new_provider,
        model_type=model_type,
        model_name=model_name,
        api_key=api_key or current_config.get("api_key"),
    )
    
    # Preserve non-provider-specific settings
    preserved_keys = ["base_url", "timeout", "max_retries", "temperature", "max_tokens"]
    for key in preserved_keys:
        if key in current_config:
            new_config[key] = current_config[key]
    
    return new_config
