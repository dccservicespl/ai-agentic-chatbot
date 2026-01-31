from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from ai_agentic_chatbot.infrastructure.db_config import db_settings

engine = create_async_engine(
    db_settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=False,
    connect_args={
        "ssl": {
            "ca": db_settings.ssl_ca
        }
    }
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)
