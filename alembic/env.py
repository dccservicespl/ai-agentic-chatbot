import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

load_dotenv()

# Alembic Config object — provides access to values in alembic.ini
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import Base and register all ORM models so autogenerate can detect them
from ai_agentic_chatbot.infrastructure.database import Base
import ai_agentic_chatbot.auth.models  # noqa: F401 — registers User with Base.metadata
from pgvector.sqlalchemy import Vector  # noqa: F401 — registers vector type so Alembic does not warn on pgvector columns

target_metadata = Base.metadata


def include_object(object, name, type_, reflected, compare_to):
    """Only manage tables that are explicitly defined in our ORM models.

    This prevents Alembic from detecting or dropping pre-existing business
    tables (orders, sales, customer, etc.) that have no ORM model.
    Any new model added to Base will automatically be picked up.
    """
    if type_ == "table":
        return name in Base.metadata.tables
    return True


def get_url() -> str:
    user = os.environ["POSTGRESQL_USER"]
    password = os.environ["POSTGRESQL_PASSWORD"]
    host = os.environ["POSTGRESQL_HOST"]
    port = os.environ.get("POSTGRESQL_PORT", "5432")
    db = os.environ["POSTGRESQL_DB"]
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (emits SQL to stdout)."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()