"""Initialize datasources from configuration."""

from .datasource_settings import get_datasource_settings
from ai_agentic_chatbot.infrastructure.datasource.factory import get_datasource_factory
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


def initialize_datasources():
    """Initialize all datasources from configuration."""
    logger.debug("Loading datasource settings from configuration")
    settings = get_datasource_settings()
    factory = get_datasource_factory()

    logger.info(f"Found {len(settings.datasources)} datasource(s) to initialize")

    for ds_name, ds_config in settings.datasources.items():
        try:
            logger.debug(
                f"Registering datasource '{ds_name}' with provider '{ds_config.provider.value}'"
            )
            factory.register_datasource(
                name=ds_name,
                provider=ds_config.provider,
                ds_type=ds_config.ds_type,
                config=ds_config.config,
            )
            logger.info(
                f"Successfully registered datasource '{ds_name}' ({ds_config.provider.value})"
            )
        except Exception as e:
            logger.error(
                f"Failed to register datasource '{ds_name}': {e}", exc_info=True
            )
            raise

    logger.info("All datasources registered successfully")
    return factory


def get_default_datasource():
    """Get the default datasource name from settings."""
    settings = get_datasource_settings()
    logger.debug(f"Default datasource: {settings.default_datasource}")
    return settings.default_datasource
