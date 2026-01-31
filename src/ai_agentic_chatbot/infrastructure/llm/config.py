"""LLM Configuration models for multiple providers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from ai_agentic_chatbot.infrastructure.llm.types import (
    LLMProvider,
    PROVIDER_CONFIG_REGISTRY,
)


class BaseLLMConfig(BaseModel, ABC):
    """Base configuration class for all LLM providers."""

    model_name: str = Field(..., description="Model deployment/name")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Model temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens in response"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")

    @abstractmethod
    def get_client_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific client initialization arguments."""
        pass

    class Config:
        frozen = True
        extra = "forbid"


class AzureOpenAIConfig(BaseLLMConfig):
    """Configuration for Azure OpenAI models."""

    api_key: str = Field(..., description="Azure OpenAI API key")
    endpoint: str = Field(..., description="Azure OpenAI endpoint")
    api_version: str = Field(
        default="2024-02-15-preview", description="Azure OpenAI API version"
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Presence penalty"
    )

    @field_validator("endpoint")
    def validate_endpoint(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Endpoint must be a valid URL")
        return v.rstrip("/")

    def get_client_kwargs(self) -> Dict[str, Any]:
        return {
            "azure_deployment": self.model_name,
            "azure_endpoint": self.endpoint,
            "api_key": self.api_key,
            "api_version": self.api_version,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


PROVIDER_CONFIG_REGISTRY[LLMProvider.AZURE_OPENAI] = AzureOpenAIConfig

ProviderConfig = Union[AzureOpenAIConfig]


def get_provider_config_class(provider: LLMProvider):
    if provider not in PROVIDER_CONFIG_REGISTRY:
        raise ValueError(f"No configuration class registered for provider: {provider}")
    return PROVIDER_CONFIG_REGISTRY[provider]
