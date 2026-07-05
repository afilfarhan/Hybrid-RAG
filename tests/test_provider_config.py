"""
Tests for provider configuration module
"""

import pytest

from src.core.provider_config import (
    get_provider_model,
    get_available_providers,
    get_available_models,
    create_provider_config,
    switch_provider,
    PROVIDER_MODELS,
    DEFAULT_MODELS,
)


class TestProviderConfig:
    """Test provider configuration functions."""

    def test_get_provider_model_openai(self):
        """Test OpenAI model naming."""
        model = get_provider_model("openai", "llm", "gpt-4o-mini")
        assert model == "openai/gpt-4o-mini"

    def test_get_provider_model_anthropic(self):
        """Test Anthropic model naming."""
        model = get_provider_model("anthropic", "llm", "claude-3-5-sonnet-20241022")
        assert model == "anthropic/claude-3-5-sonnet-20241022"

    def test_get_provider_model_google(self):
        """Test Google model naming."""
        model = get_provider_model("google", "llm", "gemini-1.5-flash")
        assert model == "gemini/gemini-1.5-flash"

    def test_get_provider_model_vertex_ai(self):
        """Test Vertex AI model naming."""
        model = get_provider_model("vertex_ai", "llm", "claude-sonnet-4")
        assert model == "vertex_ai/claude-sonnet-4"

    def test_get_provider_model_bedrock(self):
        """Test Bedrock model naming."""
        model = get_provider_model("bedrock", "llm", "anthropic.claude-3-5-sonnet-20241022-v1:0")
        assert model == "bedrock/anthropic.claude-3-5-sonnet-20241022-v1:0"

    def test_get_provider_model_nvidia_nim(self):
        """Test NVIDIA NIM model naming."""
        model = get_provider_model("nvidia_nim", "llm", "meta/llama3-70b-instruct")
        assert model == "nvidia_nim/meta/llama3-70b-instruct"

    def test_get_provider_model_nvidia_nim_embedding(self):
        """Test NVIDIA NIM embedding model naming."""
        model = get_provider_model("nvidia_nim", "embedding", "nvidia/nv-embedqa-e5-v5")
        assert model == "nvidia_nim/nvidia/nv-embedqa-e5-v5"

    def test_get_provider_model_embedding(self):
        """Test embedding model naming."""
        model = get_provider_model("openai", "embedding", "text-embedding-3-small")
        assert model == "openai/text-embedding-3-small"

    def test_get_provider_model_default(self):
        """Test default model selection."""
        model = get_provider_model("openai", "llm")
        assert model == "openai/" + DEFAULT_MODELS["openai"]["llm"]

    def test_get_provider_model_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="Invalid provider"):
            get_provider_model("invalid_provider", "llm")

    def test_get_provider_model_invalid_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            get_provider_model("openai", "invalid_type")

    def test_get_provider_model_unsupported_type(self):
        """Test unsupported model type for provider raises error."""
        with pytest.raises(ValueError, match="does not support embedding models"):
            get_provider_model("anthropic", "embedding")

    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = get_available_providers()
        assert set(providers) == set(PROVIDER_MODELS.keys())

    def test_get_available_models_openai(self):
        """Test getting available models for OpenAI."""
        models = get_available_models("openai", "llm")
        assert models == PROVIDER_MODELS["openai"]["llm"]

    def test_get_available_models_anthropic(self):
        """Test getting available models for Anthropic."""
        models = get_available_models("anthropic", "llm")
        assert models == PROVIDER_MODELS["anthropic"]["llm"]

    def test_get_available_models_nvidia_nim(self):
        """Test getting available models for NVIDIA NIM."""
        models = get_available_models("nvidia_nim", "llm")
        assert models == PROVIDER_MODELS["nvidia_nim"]["llm"]

    def test_create_provider_config(self):
        """Test creating provider configuration."""
        config = create_provider_config(
            provider="openai",
            model_type="llm",
            model_name="gpt-4o-mini",
            api_key="test-key",
        )
        assert config["model_name"] == "openai/gpt-4o-mini"
        assert config["api_key"] == "test-key"

    def test_create_provider_config_with_extra_kwargs(self):
        """Test creating provider configuration with extra kwargs."""
        config = create_provider_config(
            provider="bedrock",
            model_type="llm",
            model_name="anthropic.claude-3-5-sonnet-20241022-v1:0",
            extra_kwargs={
                "aws_access_key_id": "test-key",
                "aws_secret_access_key": "test-secret",
                "aws_region_name": "us-east-1",
            },
        )
        assert config["aws_access_key_id"] == "test-key"
        assert config["aws_secret_access_key"] == "test-secret"
        assert config["aws_region_name"] == "us-east-1"

    def test_create_provider_config_nvidia_nim(self):
        """Test creating NVIDIA NIM configuration."""
        config = create_provider_config(
            provider="nvidia_nim",
            model_type="llm",
            model_name="meta/llama3-70b-instruct",
            api_key="nvapi-test-key",
        )
        assert config["model_name"] == "nvidia_nim/meta/llama3-70b-instruct"
        assert config["api_key"] == "nvapi-test-key"

    def test_switch_provider(self):
        """Test switching provider."""
        current_config = create_provider_config(
            provider="openai",
            model_type="llm",
            model_name="gpt-4o-mini",
            api_key="sk-openai-key",
        )
        
        new_config = switch_provider(
            current_config,
            "anthropic",
            "llm",
            "claude-3-5-sonnet-20241022",
            "sk-anthropic-key",
        )
        
        assert new_config["model_name"] == "anthropic/claude-3-5-sonnet-20241022"
        assert new_config["api_key"] == "sk-anthropic-key"

    def test_switch_provider_preserves_settings(self):
        """Test that switch_provider preserves non-provider-specific settings."""
        current_config = create_provider_config(
            provider="openai",
            model_type="llm",
            model_name="gpt-4o-mini",
            api_key="sk-openai-key",
        )
        current_config["temperature"] = 0.7
        current_config["max_tokens"] = 1000
        
        new_config = switch_provider(
            current_config,
            "anthropic",
            "llm",
            "claude-3-5-sonnet-20241022",
            "sk-anthropic-key",
        )
        
        assert new_config["temperature"] == 0.7
        assert new_config["max_tokens"] == 1000

    def test_switch_provider_preserves_api_key(self):
        """Test that switch_provider preserves API key if not provided."""
        current_config = create_provider_config(
            provider="openai",
            model_type="llm",
            model_name="gpt-4o-mini",
            api_key="sk-openai-key",
        )
        
        new_config = switch_provider(
            current_config,
            "anthropic",
            "llm",
            "claude-3-5-sonnet-20241022",
        )
        
        assert new_config["api_key"] == "sk-openai-key"
