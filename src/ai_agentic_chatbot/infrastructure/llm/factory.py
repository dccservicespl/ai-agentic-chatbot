"""LLM Factory for creating LangChain-compatible clients with multi-provider support."""

from typing import Dict, Optional, Union, Any
from threading import Lock
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from ai_agentic_chatbot.settings import get_settings, ModelConfiguration
from ai_agentic_chatbot.types import LLMProvider, ModelType
from ai_agentic_chatbot.config import (
    AzureOpenAIConfig,
    OpenAIConfig,
    AnthropicConfig,
    AWSBedrockConfig,
)


LangChainLLM = BaseChatModel


class LLMFactory:
    """Singleton factory for creating LangChain-compatible LLM clients."""

    _instance: Optional["LLMFactory"] = None
    _lock = Lock()

    def __new__(cls) -> "LLMFactory":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the factory if not already done."""
        if not getattr(self, "_initialized", False):
            self._clients: Dict[str, LangChainLLM] = {}
            self._settings = get_settings()
            self._initialized = True

    def get_llm(
        self, provider: Optional[LLMProvider] = None, model: Optional[ModelType] = None
    ) -> LangChainLLM:
        """
        Get a LangChain-compatible LLM instance.

        Args:
            provider: The LLM provider (e.g., LLMProvider.AZURE_OPENAI).
            model: The model type (e.g., ModelType.FAST, ModelType.SMART).
                   If both are None, uses the default model from settings.

        Returns:
            LangChainLLM: LangChain-compatible chat model instance.

        Raises:
            ValueError: If the provider/model combination is not found.
        """
        # Use defaults if not specified
        if provider is None and model is None:
            model_key = self._settings.default_model
        else:
            # Build model key from model type only (provider is handled in config)
            if model is None:
                model = ModelType.FAST  # Default model type

            model_key = model.value

        # Return cached client if available
        if model_key in self._clients:
            return self._clients[model_key]

        # Get model configuration and create client
        model_config = self._settings.get_model_config(model_key)
        client = self._create_client(model_config)

        # Cache and return
        self._clients[model_key] = client
        return client

    def _create_client(self, model_config: ModelConfiguration) -> LangChainLLM:
        """Create a LangChain-compatible client based on provider type."""
        provider = model_config.provider
        config = model_config.config

        if provider == LLMProvider.AZURE_OPENAI:
            return self._create_azure_openai_client(config)
        elif provider == LLMProvider.OPENAI:
            return self._create_openai_client(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_azure_openai_client(self, config: AzureOpenAIConfig) -> AzureChatOpenAI:
        """Create Azure OpenAI LangChain client."""
        return AzureChatOpenAI(**config.get_client_kwargs())

    def _create_openai_client(self, config: OpenAIConfig) -> ChatOpenAI:
        """Create OpenAI LangChain client."""
        return ChatOpenAI(**config.get_client_kwargs())

    def get_available_models(self) -> list[str]:
        """Get list of available model keys."""
        return list(self._settings.models.keys())

    def get_models_by_provider(
        self, provider: LLMProvider
    ) -> Dict[str, ModelConfiguration]:
        """Get all models for a specific provider."""
        return self._settings.llm.get_models_by_provider(provider)

    def get_supported_providers(self) -> list[LLMProvider]:
        """Get list of providers with configured models."""
        providers = set()
        for model_config in self._settings.models.values():
            providers.add(model_config.provider)
        return list(providers)

    def clear_cache(self):
        """Clear all cached LLM clients."""
        self._clients.clear()

    def reload_settings(self):
        """Reload settings and clear cache."""
        from ai_agentic_chatbot.settings import reload_settings

        self._settings = reload_settings()
        self.clear_cache()


# Global factory instance
_factory: Optional[LLMFactory] = None


def get_llm_factory() -> LLMFactory:
    """Get the global LLM factory instance."""
    global _factory
    if _factory is None:
        _factory = LLMFactory()
    return _factory


def get_llm(
    provider: Optional[LLMProvider] = None, model: Optional[ModelType] = None
) -> LangChainLLM:
    """
    Convenience function to get a LangChain-compatible LLM instance.

    Args:
        provider: The LLM provider enum. If None, uses default.
        model: The model type enum. If None, uses default.

    Returns:
        LangChainLLM: LangChain-compatible chat model instance.

    Example:
        >>> from ai_agentic_chatbot.types import LLMProvider, ModelType
        >>> llm = get_llm(LLMProvider.AZURE_OPENAI, ModelType.FAST)
        >>> llm = get_llm()  # Get default model
        >>> response = llm.invoke("Hello, world!")
    """
    return get_llm_factory().get_llm(provider, model)
