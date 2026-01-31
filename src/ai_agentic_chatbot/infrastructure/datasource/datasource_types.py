from enum import Enum
from typing import Dict, Type, Any


class DataSourceProvider(Enum):
    """Supported datasource providers with extensibility."""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    AZURE_SQL = "azure_sql"
    AWS_RDS_MYSQL = "aws_rds_mysql"
    AWS_RDS_POSTGRESQL = "aws_rds_postgresql"
    SQLITE = "sqlite"

    @classmethod
    def get_all_providers(cls) -> list[str]:
        """Get all available datasource provider names."""
        return [provider.value for provider in cls]

    @classmethod
    def from_string(cls, provider_str: str) -> "DataSourceProvider":
        """Create provider enum from string, with validation."""
        for provider in cls:
            if provider.value == provider_str:
                return provider
        raise ValueError(f"Unknown datasource provider: {provider_str}. Available: {cls.get_all_providers()}")


class DataSourceType(Enum):
    """Datasource types/purposes."""
    PRIMARY = "primary"
    ANALYTICS = "analytics"
    CACHE = "cache"
    LOGGING = "logging"
    BACKUP = "backup"

    @classmethod
    def get_all_types(cls) -> list[str]:
        """Get all available datasource types."""
        return [ds_type.value for ds_type in cls]


# Registry for dynamic datasource configuration classes
DATASOURCE_CONFIG_REGISTRY: Dict[DataSourceProvider, Type[Any]] = {}
