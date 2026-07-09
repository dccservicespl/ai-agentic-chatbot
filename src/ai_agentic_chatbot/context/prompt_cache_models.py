from datetime import datetime
from typing import List, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ai_agentic_chatbot.infrastructure.database import Base


class PromptCache(Base):
    __tablename__ = "prompt_cache"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    normalized_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    db_context_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("app.db_contexts.id", ondelete="CASCADE"),
        nullable=False,
    )
    schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    generated_sql: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chart_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    chart_config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    result_columns: Mapped[Optional[List[dict]]] = mapped_column(JSONB, nullable=True)
    hit_count: Mapped[int] = mapped_column(BigInteger, server_default="0", nullable=False)
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )