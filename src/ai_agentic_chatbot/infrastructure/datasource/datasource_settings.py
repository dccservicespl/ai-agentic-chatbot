"""Datasource settings management with environment variable support."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel

from .datasource_types import DataSourceProvider, DataSourceType
from .datasource_config import get_datasource_config_class, BaseDatasourceConfig
from ai_agentic_chatbot.infrastructure.datasource.factory import DataSourceConfiguration


class DataSourceSettings(BaseModel):
    """Settings for datasource configurations."""

    default_datasource: str
    datasources: Dict[str, DataSourceConfiguration]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config_file(
        cls, config_path: Optional[Path] = None
    ) -> "DataSourceSettings":
        """Load datasource settings from config.yaml file with environment variable overrides."""
        if config_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            config_path = project_root / "config.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
        else:
            config_data = {}

        return cls._parse_config(config_data)

    @classmethod
    def _parse_config(cls, config_data: Dict[str, Any]) -> "DataSourceSettings":
        """Parse configuration data into DataSourceSettings object."""
        datasource_config = config_data.get("datasources", {})

        default_datasource = datasource_config.get("default", "mysql.primary")
        if "." in default_datasource:
            default_datasource = default_datasource.split(".", 1)[1]

        datasources = {}

        for provider_name, provider_config in datasource_config.items():
            if provider_name == "default":
                continue

            try:
                provider = DataSourceProvider.from_string(provider_name)
            except ValueError:
                continue

            config_class = get_datasource_config_class(provider)

            if isinstance(provider_config, dict):
                for ds_name, ds_data in provider_config.items():
                    if isinstance(ds_data, dict):
                        ds_data = cls._apply_env_overrides(ds_data, provider)
                        provider_ds_config = config_class(**ds_data)
                        ds_type = cls._determine_datasource_type(ds_name)

                        full_ds_name = f"{provider_name}.{ds_name}"
                        datasources[full_ds_name] = DataSourceConfiguration(
                            provider=provider,
                            ds_type=ds_type,
                            config=provider_ds_config,
                        )

        return cls(default_datasource=default_datasource, datasources=datasources)

    @staticmethod
    def _apply_env_overrides(
        ds_data: Dict[str, Any], provider: DataSourceProvider
    ) -> Dict[str, Any]:
        """Apply environment variable overrides to datasource configuration."""
        ds_data = ds_data.copy()

        if provider == DataSourceProvider.MYSQL:
            if "MYSQL_HOST" in os.environ:
                ds_data["host"] = os.environ["MYSQL_HOST"]
            if "MYSQL_PORT" in os.environ:
                ds_data["port"] = int(os.environ["MYSQL_PORT"])
            if "MYSQL_DATABASE" in os.environ:
                ds_data["database"] = os.environ["MYSQL_DATABASE"]
            if "MYSQL_USERNAME" in os.environ:
                ds_data["username"] = os.environ["MYSQL_USERNAME"]
            if "MYSQL_PASSWORD" in os.environ:
                ds_data["password"] = os.environ["MYSQL_PASSWORD"]

        elif provider == DataSourceProvider.POSTGRESQL:
            if "POSTGRES_HOST" in os.environ:
                ds_data["host"] = os.environ["POSTGRES_HOST"]
            if "POSTGRES_PORT" in os.environ:
                ds_data["port"] = int(os.environ["POSTGRES_PORT"])
            if "POSTGRES_DB" in os.environ:
                ds_data["database"] = os.environ["POSTGRES_DB"]
            if "POSTGRES_USER" in os.environ:
                ds_data["username"] = os.environ["POSTGRES_USER"]
            if "POSTGRES_PASSWORD" in os.environ:
                ds_data["password"] = os.environ["POSTGRES_PASSWORD"]

        elif provider == DataSourceProvider.AZURE_SQL:
            if "AZURE_SQL_HOST" in os.environ:
                ds_data["host"] = os.environ["AZURE_SQL_HOST"]
            if "AZURE_SQL_DATABASE" in os.environ:
                ds_data["database"] = os.environ["AZURE_SQL_DATABASE"]
            if "AZURE_SQL_USERNAME" in os.environ:
                ds_data["username"] = os.environ["AZURE_SQL_USERNAME"]
            if "AZURE_SQL_PASSWORD" in os.environ:
                ds_data["password"] = os.environ["AZURE_SQL_PASSWORD"]

        elif provider in [
            DataSourceProvider.AWS_RDS_MYSQL,
            DataSourceProvider.AWS_RDS_POSTGRESQL,
        ]:
            if "AWS_RDS_HOST" in os.environ:
                ds_data["host"] = os.environ["AWS_RDS_HOST"]
            if "AWS_RDS_PORT" in os.environ:
                ds_data["port"] = int(os.environ["AWS_RDS_PORT"])
            if "AWS_RDS_DATABASE" in os.environ:
                ds_data["database"] = os.environ["AWS_RDS_DATABASE"]
            if "AWS_RDS_USERNAME" in os.environ:
                ds_data["username"] = os.environ["AWS_RDS_USERNAME"]
            if "AWS_RDS_PASSWORD" in os.environ:
                ds_data["password"] = os.environ["AWS_RDS_PASSWORD"]

        return ds_data

    @staticmethod
    def _determine_datasource_type(ds_name: str) -> DataSourceType:
        """Determine datasource type from name."""
        name_lower = ds_name.lower()
        for ds_type in DataSourceType:
            if ds_type.value in name_lower:
                return ds_type

        return DataSourceType.PRIMARY

    def get_datasource_config(
        self, ds_name: Optional[str] = None
    ) -> DataSourceConfiguration:
        """
        Get datasource configuration by name.

        Args:
            ds_name: Datasource name. If None, returns default datasource.

        Returns:
            DataSourceConfiguration: The datasource configuration

        Raises:
            ValueError: If datasource is not found
        """
        if ds_name is None:
            ds_name = self.default_datasource

        if ds_name not in self.datasources:
            available = list(self.datasources.keys())
            raise ValueError(
                f"Datasource '{ds_name}' not found. Available: {available}"
            )

        return self.datasources[ds_name]


# Global settings instance
_settings: Optional[DataSourceSettings] = None


def get_datasource_settings() -> DataSourceSettings:
    """Get the global datasource settings instance."""
    global _settings
    if _settings is None:
        _settings = DataSourceSettings.from_config_file()
    return _settings


def reload_datasource_settings() -> DataSourceSettings:
    """Reload datasource settings from config file."""
    global _settings
    _settings = DataSourceSettings.from_config_file()
    return _settings
