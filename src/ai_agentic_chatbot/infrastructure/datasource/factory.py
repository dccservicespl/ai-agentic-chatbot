"""Datasource factory for creating database connections with multi-provider support."""

from typing import Dict, Optional, Any
from threading import Lock
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from .datasource_config import (
    BaseDatasourceConfig,
    MySQLConfig,
    PostgreSQLConfig,
    AzureSQLConfig,
    SQLiteConfig,
    get_datasource_config_class,
)
from .datasource_types import DataSourceProvider, DataSourceType


class DataSourceConfiguration:
    """Container for datasource configuration and metadata."""

    def __init__(
        self,
        provider: DataSourceProvider,
        ds_type: DataSourceType,
        config: BaseDatasourceConfig,
    ):
        self.provider = provider
        self.ds_type = ds_type
        self.config = config


class DataSourceFactory:
    """Singleton factory for creating and managing database connections."""

    _instance: Optional["DataSourceFactory"] = None
    _lock = Lock()

    def __new__(cls) -> "DataSourceFactory":
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
            self._engines: Dict[str, Engine] = {}
            self._session_makers: Dict[str, sessionmaker] = {}
            self._configurations: Dict[str, DataSourceConfiguration] = {}
            self._initialized = True

    def register_datasource(
        self,
        name: str,
        provider: DataSourceProvider,
        ds_type: DataSourceType,
        config: BaseDatasourceConfig,
    ) -> None:
        """
        Register a datasource configuration.

        Args:
            name: Unique name for the datasource
            provider: Database provider type
            ds_type: Datasource type/purpose
            config: Provider-specific configuration
        """
        self._configurations[name] = DataSourceConfiguration(provider, ds_type, config)

    def get_engine(self, datasource_name: str) -> Engine:
        """
        Get a SQLAlchemy engine for the specified datasource.

        Args:
            datasource_name: Name of the registered datasource

        Returns:
            Engine: SQLAlchemy engine instance

        Raises:
            ValueError: If datasource is not registered
        """
        if datasource_name not in self._configurations:
            raise ValueError(f"Datasource '{datasource_name}' is not registered")

        if datasource_name in self._engines:
            return self._engines[datasource_name]

        config_obj = self._configurations[datasource_name]
        connection_string = config_obj.config.get_connection_string()
        engine_kwargs = config_obj.config.get_engine_kwargs()

        engine = create_engine(connection_string, **engine_kwargs)

        self._engines[datasource_name] = engine
        self._session_makers[datasource_name] = sessionmaker(bind=engine)

        return engine

    def get_session(self, datasource_name: str) -> Session:
        """
        Get a SQLAlchemy session for the specified datasource.

        Args:
            datasource_name: Name of the registered datasource

        Returns:
            Session: SQLAlchemy session instance

        Raises:
            ValueError: If datasource is not registered
        """
        if datasource_name not in self._configurations:
            raise ValueError(f"Datasource '{datasource_name}' is not registered")

        # Ensure engine and session maker exist
        if datasource_name not in self._session_makers:
            self.get_engine(datasource_name)

        return self._session_makers[datasource_name]()

    def get_datasource_info(self, datasource_name: str) -> Dict[str, Any]:
        """
        Get information about a registered datasource.

        Args:
            datasource_name: Name of the registered datasource

        Returns:
            Dict containing datasource information

        Raises:
            ValueError: If datasource is not registered
        """
        if datasource_name not in self._configurations:
            raise ValueError(f"Datasource '{datasource_name}' is not registered")

        config_obj = self._configurations[datasource_name]
        return {
            "name": datasource_name,
            "provider": config_obj.provider.value,
            "type": config_obj.ds_type.value,
            "host": config_obj.config.host,
            "port": config_obj.config.port,
            "database": config_obj.config.database,
            "pool_size": config_obj.config.pool_size,
        }

    def list_datasources(self) -> list[str]:
        """Get list of registered datasource names."""
        return list(self._configurations.keys())

    def get_datasources_by_type(
        self, ds_type: DataSourceType
    ) -> Dict[str, DataSourceConfiguration]:
        """Get all datasources of a specific type."""
        return {
            name: config
            for name, config in self._configurations.items()
            if config.ds_type == ds_type
        }

    def get_datasources_by_provider(
        self, provider: DataSourceProvider
    ) -> Dict[str, DataSourceConfiguration]:
        """Get all datasources of a specific provider."""
        return {
            name: config
            for name, config in self._configurations.items()
            if config.provider == provider
        }

    def close_all_connections(self):
        """Close all database connections and clear cache."""
        for engine in self._engines.values():
            engine.dispose()

        self._engines.clear()
        self._session_makers.clear()

    def test_connection(self, datasource_name: str) -> bool:
        """
        Test connection to a datasource.

        Args:
            datasource_name: Name of the registered datasource

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            engine = self.get_engine(datasource_name)
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False


# Global factory instance
_factory: Optional[DataSourceFactory] = None


def get_datasource_factory() -> DataSourceFactory:
    """Get the global datasource factory instance."""
    global _factory
    if _factory is None:
        _factory = DataSourceFactory()
    return _factory


def get_engine(datasource_name: str) -> Engine:
    """
    Convenience function to get a SQLAlchemy engine.

    Args:
        datasource_name: Name of the registered datasource

    Returns:
        Engine: SQLAlchemy engine instance

    Example:
        >>> engine = get_engine("primary_db")
        >>> with engine.connect() as conn:
        ...     result = conn.execute("SELECT * FROM users")
    """
    return get_datasource_factory().get_engine(datasource_name)


def get_session(datasource_name: str) -> Session:
    """
    Convenience function to get a SQLAlchemy session.

    Args:
        datasource_name: Name of the registered datasource

    Returns:
        Session: SQLAlchemy session instance

    Example:
        >>> session = get_session("primary_db")
        >>> users = session.query(User).all()
        >>> session.close()
    """
    return get_datasource_factory().get_session(datasource_name)


def register_mysql_datasource(
    name: str,
    host: str,
    database: str,
    username: str,
    password: str,
    port: int = 3306,
    ds_type: DataSourceType = DataSourceType.PRIMARY,
    **kwargs,
) -> None:
    """
    Convenience function to register a MySQL datasource.

    Args:
        name: Unique name for the datasource
        host: MySQL host
        database: Database name
        username: Username
        password: Password
        port: MySQL port (default: 3306)
        ds_type: Datasource type (default: PRIMARY)
        **kwargs: Additional MySQL configuration options
    """
    config = MySQLConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        **kwargs,
    )

    get_datasource_factory().register_datasource(
        name=name, provider=DataSourceProvider.MYSQL, ds_type=ds_type, config=config
    )


def register_postgresql_datasource(
    name: str,
    host: str,
    database: str,
    username: str,
    password: str,
    port: int = 5432,
    ds_type: DataSourceType = DataSourceType.PRIMARY,
    **kwargs,
) -> None:
    """
    Convenience function to register a PostgreSQL datasource.

    Args:
        name: Unique name for the datasource
        host: PostgreSQL host
        database: Database name
        username: Username
        password: Password
        port: PostgreSQL port (default: 5432)
        ds_type: Datasource type (default: PRIMARY)
        **kwargs: Additional PostgreSQL configuration options
    """
    config = PostgreSQLConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        **kwargs,
    )

    get_datasource_factory().register_datasource(
        name=name,
        provider=DataSourceProvider.POSTGRESQL,
        ds_type=ds_type,
        config=config,
    )
