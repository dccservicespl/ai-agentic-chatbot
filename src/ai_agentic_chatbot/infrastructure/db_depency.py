from typing import AsyncGenerator
from ai_agentic_chatbot.infrastructure.db_session import AsyncSessionLocal


async def get_db() -> AsyncGenerator:
    async with AsyncSessionLocal() as session:
        yield session
