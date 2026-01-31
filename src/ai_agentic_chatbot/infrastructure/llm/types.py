from enum import Enum
from typing import Dict, Type, Any


class LLMProvider(Enum):
    """Supported LLM providers with extensibility."""

    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws_bedrock"

    @classmethod
    def get_all_providers(cls) -> list[str]:
        """Get all available provider names."""
        return [provider.value for provider in cls]

    @classmethod
    def from_string(cls, provider_str: str) -> "LLMProvider":
        """Create provider enum from string, with validation."""
        for provider in cls:
            if provider.value == provider_str:
                return provider
        raise ValueError(
            f"Unknown provider: {provider_str}. Available: {cls.get_all_providers()}"
        )


class ModelType(Enum):
    """Model types/tiers across providers."""

    FAST = "fast"
    SMART = "smart"
    EMBEDDING = "embedding"
    VISION = "vision"

    @classmethod
    def get_all_types(cls) -> list[str]:
        """Get all available model types."""
        return [model_type.value for model_type in cls]


PROVIDER_CONFIG_REGISTRY: Dict[LLMProvider, Type[Any]] = {}
