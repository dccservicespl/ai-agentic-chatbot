from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ai_agentic_chatbot.infrastructure.database import Base


class SchemaVersion(Base):
    __tablename__ = "schema_versions"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    db_context_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("app.db_contexts.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Denormalized: captured at write time, not derived via join, since
    # db_contexts.schema_name can be overwritten by seed_contexts_from_config.
    schema_name: Mapped[str] = mapped_column(String(100), nullable=False)
    schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    captured_by_user_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    table_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)