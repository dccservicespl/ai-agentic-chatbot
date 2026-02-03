"""LLM Factory for creating LangChain-compatible clients with multi-provider support."""

from typing import Dict, Optional, Union, Any
from threading import Lock
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureChatOpenAI, ChatOpenAI, AzureOpenAIEmbeddings

from .config import AzureOpenAIConfig, AzureOpenAIEmbeddingConfig
from .settings import get_settings, ModelConfiguration
from .types import LLMProvider, ModelType

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
            self._embeddings: Dict[str, Embeddings] = {}
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
        if provider is None and model is None:
            model_key = self._settings.default_model
        else:
            if model is None:
                model = ModelType.FAST

            model_key = model.value

        if model_key in self._clients:
            return self._clients[model_key]

        model_config = self._settings.get_model_config(model_key)
        client = self._create_client(model_config)

        self._clients[model_key] = client
        return client

    def _create_client(self, model_config: ModelConfiguration) -> LangChainLLM:
        """Create a LangChain-compatible client based on provider type."""
        provider = model_config.provider
        config = model_config.config

        if provider == LLMProvider.AZURE_OPENAI:
            return self._create_azure_openai_client(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_azure_openai_client(self, config: AzureOpenAIConfig) -> AzureChatOpenAI:
        """Create Azure OpenAI LangChain client."""
        return AzureChatOpenAI(**config.get_client_kwargs())

    def get_embedding(
        self, provider: Optional[LLMProvider] = None, model: Optional[ModelType] = None
    ) -> Embeddings:
        """
        Get a LangChain-compatible embedding instance.

        Args:
            provider: The LLM provider (e.g., LLMProvider.AZURE_OPENAI).
            model: The model type (should be ModelType.EMBEDDING).

        Returns:
            Embeddings: LangChain-compatible embedding instance.

        Raises:
            ValueError: If the provider/model combination is not found.
        """
        if model is None:
            model = ModelType.EMBEDDING

        if model != ModelType.EMBEDDING:
            raise ValueError("Only embedding model type is supported for embeddings")

        model_key = f"{provider.value if provider else 'azure_openai'}.{model.value}"

        if model_key in self._embeddings:
            return self._embeddings[model_key]

        # Get embedding configuration from settings
        embedding_config = self._get_embedding_config(
            provider or LLMProvider.AZURE_OPENAI
        )
        embedding_client = self._create_embedding_client(
            provider or LLMProvider.AZURE_OPENAI, embedding_config
        )

        self._embeddings[model_key] = embedding_client
        return embedding_client

    def _get_embedding_config(
        self, provider: LLMProvider
    ) -> Union[AzureOpenAIEmbeddingConfig]:
        """Get embedding configuration for the specified provider."""
        if provider == LLMProvider.AZURE_OPENAI:
            # Get embedding config from settings
            embedding_model_config = self._settings.get_model_config("embedding")
            config_data = embedding_model_config.config.model_dump()
            return AzureOpenAIEmbeddingConfig(**config_data)
        else:
            raise ValueError(f"Unsupported provider for embeddings: {provider}")

    def _create_embedding_client(
        self, provider: LLMProvider, config: Union[AzureOpenAIEmbeddingConfig]
    ) -> Embeddings:
        """Create embedding client based on provider type."""
        if provider == LLMProvider.AZURE_OPENAI:
            return self._create_azure_openai_embedding_client(config)
        else:
            raise ValueError(f"Unsupported provider for embeddings: {provider}")

    def _create_azure_openai_embedding_client(
        self, config: AzureOpenAIEmbeddingConfig
    ) -> AzureOpenAIEmbeddings:
        """Create Azure OpenAI embedding client."""
        return AzureOpenAIEmbeddings(**config.get_client_kwargs())

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
        """Clear all cached LLM clients and embeddings."""
        self._clients.clear()
        self._embeddings.clear()

    def reload_settings(self):
        """Reload settings and clear cache."""
        from .settings import reload_settings

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
    """
    return get_llm_factory().get_llm(provider, model)


def get_embedding(
    provider: Optional[LLMProvider] = None, model: Optional[ModelType] = None
) -> Embeddings:
    """
    Convenience function to get a LangChain-compatible embedding instance.

    Args:
        provider: The LLM provider enum. If None, uses Azure OpenAI.
        model: The model type enum. If None, uses EMBEDDING.

    Returns:
        Embeddings: LangChain-compatible embedding instance.
    """
    return get_llm_factory().get_embedding(provider, model)
