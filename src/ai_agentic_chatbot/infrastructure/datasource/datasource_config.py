"""Datasource configuration models for multiple database providers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from .datasource_types import (
    DataSourceProvider,
    DataSourceType,
    DATASOURCE_CONFIG_REGISTRY,
)


class BaseDatasourceConfig(BaseModel, ABC):
    """Base configuration class for all datasource providers."""

    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")

    # Connection pool settings
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")

    # Connection settings
    connect_timeout: int = Field(
        default=10, description="Connection timeout in seconds"
    )
    charset: str = Field(default="utf8mb4", description="Character set")

    @abstractmethod
    def get_connection_string(self) -> str:
        """Get database connection string."""
        pass

    @abstractmethod
    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Get SQLAlchemy engine initialization arguments."""
        pass

    class Config:
        frozen = True
        extra = "forbid"


class MySQLConfig(BaseDatasourceConfig):
    """Configuration for MySQL databases."""

    port: int = Field(default=3306, description="MySQL port")
    charset: str = Field(default="utf8mb4", description="MySQL character set")
    ssl_mode: str = Field(default="REQUIRED", description="SSL mode for connection")
    ssl_ca: Optional[str] = Field(default=None, description="SSL CA certificate path")
    ssl_cert: Optional[str] = Field(default=None, description="SSL certificate path")
    ssl_key: Optional[str] = Field(default=None, description="SSL key path")

    def get_connection_string(self) -> str:
        """Get MySQL connection string."""
        ssl_params = ""
        if self.ssl_mode:
            ssl_params = f"?ssl_mode={self.ssl_mode}"
            if self.ssl_ca:
                ssl_params += f"&ssl_ca={self.ssl_ca}"
            if self.ssl_cert:
                ssl_params += f"&ssl_cert={self.ssl_cert}"
            if self.ssl_key:
                ssl_params += f"&ssl_key={self.ssl_key}"

        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}{ssl_params}"

    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Get MySQL engine initialization arguments."""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "connect_args": {
                "ssl_ca": self.ssl_ca,
                "connect_timeout": self.connect_timeout,
                "charset": self.charset,
            },
        }


class PostgreSQLConfig(BaseDatasourceConfig):
    """Configuration for PostgreSQL databases."""

    port: int = Field(default=5432, description="PostgreSQL port")
    charset: str = Field(default="utf8", description="PostgreSQL character set")
    sslmode: str = Field(default="require", description="SSL mode for connection")
    application_name: str = Field(
        default="ai_agentic_chatbot", description="Application name"
    )

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.sslmode}&application_name={self.application_name}"

    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Get PostgreSQL engine initialization arguments."""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "connect_args": {
                "connect_timeout": self.connect_timeout,
            },
        }


class AzureSQLConfig(BaseDatasourceConfig):
    """Configuration for Azure SQL Database."""

    port: int = Field(default=1433, description="Azure SQL port")
    driver: str = Field(
        default="ODBC Driver 18 for SQL Server", description="ODBC driver"
    )
    encrypt: bool = Field(default=True, description="Enable encryption")
    trust_server_certificate: bool = Field(
        default=False, description="Trust server certificate"
    )
    connection_timeout: int = Field(default=30, description="Connection timeout")

    def get_connection_string(self) -> str:
        """Get Azure SQL connection string."""
        params = {
            "driver": self.driver,
            "encrypt": "yes" if self.encrypt else "no",
            "trustServerCertificate": "yes" if self.trust_server_certificate else "no",
            "connectionTimeout": str(self.connection_timeout),
        }

        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"mssql+pyodbc://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?{param_string}"

    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Get Azure SQL engine initialization arguments."""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
        }


class SQLiteConfig(BaseDatasourceConfig):
    """Configuration for SQLite databases."""

    # Override fields that don't apply to SQLite
    host: str = Field(default="localhost", description="Not used for SQLite")
    port: int = Field(default=0, description="Not used for SQLite")
    username: str = Field(default="", description="Not used for SQLite")
    password: str = Field(default="", description="Not used for SQLite")

    # SQLite-specific fields
    database_path: str = Field(..., description="Path to SQLite database file")
    timeout: int = Field(default=20, description="SQLite timeout in seconds")
    check_same_thread: bool = Field(default=False, description="Check same thread")

    def get_connection_string(self) -> str:
        """Get SQLite connection string."""
        return f"sqlite:///{self.database_path}"

    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Get SQLite engine initialization arguments."""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "connect_args": {
                "timeout": self.timeout,
                "check_same_thread": self.check_same_thread,
            },
        }


# Register configuration classes
DATASOURCE_CONFIG_REGISTRY[DataSourceProvider.MYSQL] = MySQLConfig
DATASOURCE_CONFIG_REGISTRY[DataSourceProvider.POSTGRESQL] = PostgreSQLConfig
DATASOURCE_CONFIG_REGISTRY[DataSourceProvider.AZURE_SQL] = AzureSQLConfig
DATASOURCE_CONFIG_REGISTRY[DataSourceProvider.AWS_RDS_MYSQL] = (
    MySQLConfig  # Reuse MySQL config
)
DATASOURCE_CONFIG_REGISTRY[DataSourceProvider.AWS_RDS_POSTGRESQL] = (
    PostgreSQLConfig  # Reuse PostgreSQL config
)
DATASOURCE_CONFIG_REGISTRY[DataSourceProvider.SQLITE] = SQLiteConfig


def get_datasource_config_class(provider: DataSourceProvider):
    """Get the configuration class for a datasource provider."""
    if provider not in DATASOURCE_CONFIG_REGISTRY:
        raise ValueError(
            f"No configuration class registered for datasource provider: {provider}"
        )
    return DATASOURCE_CONFIG_REGISTRY[provider]
