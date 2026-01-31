"""Global settings for LLM configuration management."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from src.ai_agentic_chatbot.infrastructure.llm.types import LLMProvider, ModelType
from src.ai_agentic_chatbot.infrastructure.llm.config import ProviderConfig, get_provider_config_class


class ModelConfiguration(BaseModel):
    """Configuration for a specific model instance."""

    provider: LLMProvider = Field(..., description="LLM provider")
    model_type: ModelType = Field(..., description="Model type/tier")
    config: ProviderConfig = Field(..., description="Provider-specific configuration")

    class Config:
        arbitrary_types_allowed = True


class LLMSettings(BaseModel):
    """Multi-provider LLM settings."""

    default_model: str = Field(description="Default model key to use")
    models: Dict[str, ModelConfiguration] = Field(
        default_factory=dict, description="Model configurations by key"
    )

    def get_model_config(self, model_key: Optional[str] = None) -> ModelConfiguration:
        """Get model configuration by key, falling back to default."""
        key = model_key or self.default_model
        if key not in self.models:
            raise ValueError(
                f"Model '{key}' not found. Available: {list(self.models.keys())}"
            )
        return self.models[key]

    def get_models_by_provider(
        self, provider: LLMProvider
    ) -> Dict[str, ModelConfiguration]:
        """Get all models for a specific provider."""
        return {
            key: config
            for key, config in self.models.items()
            if config.provider == provider
        }

    def get_models_by_type(
        self, model_type: ModelType
    ) -> Dict[str, ModelConfiguration]:
        """Get all models of a specific type."""
        return {
            key: config
            for key, config in self.models.items()
            if config.model_type == model_type
        }

    class Config:
        arbitrary_types_allowed = True


class Settings(BaseModel):
    """Global settings for multi-provider LLM configuration."""

    llm: LLMSettings

    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from config.yaml file with environment variable overrides."""
        if config_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            config_path = project_root / "config.yaml"

        config_data: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}

        return cls._parse_config(config_data)

    @classmethod
    def _parse_config(cls, config_data: Dict[str, Any]) -> "Settings":
        """Parse configuration data into Settings object."""
        llm_config = config_data.get("llm", {})

        default_model_key = llm_config.get("default", "azure_openai.fast")
        if "." in default_model_key:
            default_model_key = default_model_key.split(".", 1)[1]

        models = {}

        for provider_name, provider_config in llm_config.items():
            if provider_name == "default":
                continue

            try:
                provider = LLMProvider.from_string(provider_name)
            except ValueError:
                continue

            config_class = get_provider_config_class(provider)

            if isinstance(provider_config, dict):
                for model_key, model_data in provider_config.items():
                    if isinstance(model_data, dict):
                        model_data = cls._apply_env_overrides(model_data, provider)

                        provider_model_config = config_class(**model_data)

                        model_type = cls._determine_model_type(model_key)

                        models[model_key] = ModelConfiguration(
                            provider=provider,
                            model_type=model_type,
                            config=provider_model_config,
                        )

        return cls(llm=LLMSettings(default_model=default_model_key, models=models))

    @staticmethod
    def _apply_env_overrides(
        model_data: Dict[str, Any], provider: LLMProvider
    ) -> Dict[str, Any]:
        """Apply environment variable overrides for provider-specific settings."""
        model_data = model_data.copy()

        if provider == LLMProvider.AZURE_OPENAI:
            model_data["api_key"] = os.getenv(
                "AZURE_OPENAI_API_KEY", model_data.get("api_key", "")
            )
            model_data["endpoint"] = os.getenv(
                "AZURE_OPENAI_ENDPOINT", model_data.get("endpoint", "")
            )
            model_data["api_version"] = os.getenv(
                "AZURE_OPENAI_API_VERSION",
                model_data.get("api_version", "2024-02-15-preview"),
            )
        elif provider == LLMProvider.OPENAI:
            model_data["api_key"] = os.getenv(
                "OPENAI_API_KEY", model_data.get("api_key", "")
            )
            if "OPENAI_ORGANIZATION" in os.environ:
                model_data["organization"] = os.getenv("OPENAI_ORGANIZATION")
        elif provider == LLMProvider.ANTHROPIC:
            model_data["api_key"] = os.getenv(
                "ANTHROPIC_API_KEY", model_data.get("api_key", "")
            )
        elif provider == LLMProvider.AWS_BEDROCK:
            if "AWS_ACCESS_KEY_ID" in os.environ:
                model_data["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
            if "AWS_SECRET_ACCESS_KEY" in os.environ:
                model_data["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
            if "AWS_SESSION_TOKEN" in os.environ:
                model_data["aws_session_token"] = os.getenv("AWS_SESSION_TOKEN")
            if "AWS_DEFAULT_REGION" in os.environ:
                model_data["region_name"] = os.getenv("AWS_DEFAULT_REGION")

        return model_data

    @staticmethod
    def _determine_model_type(model_key: str) -> ModelType:
        """Determine model type from key or configuration."""
        key_lower = model_key.lower()
        for model_type in ModelType:
            if model_type.value in key_lower:
                return model_type

        return ModelType.SMART

    def get_model_config(self, model_key: Optional[str] = None) -> ModelConfiguration:
        """Get model configuration by key, falling back to default."""
        return self.llm.get_model_config(model_key)

    @property
    def default_model(self) -> str:
        """Get the default model key."""
        return self.llm.default_model

    @property
    def models(self) -> Dict[str, ModelConfiguration]:
        """Get all model configurations."""
        return self.llm.models

    class Config:
        """Pydantic configuration."""

        extra = "forbid"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[Path] = None) -> Settings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings.from_config_file(config_path)
    return _settings


def reload_settings(config_path: Optional[Path] = None) -> Settings:
    """Reload settings from config file."""
    global _settings
    _settings = Settings.from_config_file(config_path)
    return _settings
