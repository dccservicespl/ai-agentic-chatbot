"""LLM Configuration models for multiple providers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from .types import LLMProvider, ModelType, PROVIDER_CONFIG_REGISTRY


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


class OpenAIConfig(BaseLLMConfig):
    """Configuration for OpenAI models."""

    api_key: str = Field(..., description="OpenAI API key")
    organization: Optional[str] = Field(default=None, description="OpenAI organization")
    base_url: Optional[str] = Field(default=None, description="Custom base URL")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Presence penalty"
    )

    def get_client_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.organization:
            kwargs["organization"] = self.organization
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


class AnthropicConfig(BaseLLMConfig):
    """Configuration for Anthropic Claude models."""

    api_key: str = Field(..., description="Anthropic API key")
    base_url: Optional[str] = Field(default=None, description="Custom base URL")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling")

    def get_client_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


class AWSBedrockConfig(BaseLLMConfig):
    """Configuration for AWS Bedrock models."""

    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret key"
    )
    aws_session_token: Optional[str] = Field(
        default=None, description="AWS session token"
    )
    region_name: str = Field(default="us-east-1", description="AWS region")

    def get_client_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "model_id": self.model_name,
            "region_name": self.region_name,
            "model_kwargs": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            kwargs["aws_session_token"] = self.aws_session_token
        return kwargs


# Register provider configurations
PROVIDER_CONFIG_REGISTRY[LLMProvider.AZURE_OPENAI] = AzureOpenAIConfig
PROVIDER_CONFIG_REGISTRY[LLMProvider.OPENAI] = OpenAIConfig
PROVIDER_CONFIG_REGISTRY[LLMProvider.ANTHROPIC] = AnthropicConfig
PROVIDER_CONFIG_REGISTRY[LLMProvider.AWS_BEDROCK] = AWSBedrockConfig


# Type alias for all provider configs
ProviderConfig = Union[
    AzureOpenAIConfig, OpenAIConfig, AnthropicConfig, AWSBedrockConfig
]


def get_provider_config_class(provider: LLMProvider):
    if provider not in PROVIDER_CONFIG_REGISTRY:
        raise ValueError(f"No configuration class registered for provider: {provider}")
    return PROVIDER_CONFIG_REGISTRY[provider]


# Legacy alias for backward compatibility
AzureLLMConfig = AzureOpenAIConfig
