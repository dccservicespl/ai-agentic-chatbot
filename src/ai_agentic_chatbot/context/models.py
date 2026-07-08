from datetime import datetime
from typing import List, Optional

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ai_agentic_chatbot.infrastructure.database import Base


class DbContext(Base):
    __tablename__ = "db_contexts"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    context_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    schema_name: Mapped[str] = mapped_column(String(100), nullable=False)
    system_prompt_path: Mapped[str] = mapped_column(Text, nullable=False)
    router_prompt_path: Mapped[str] = mapped_column(Text, nullable=False)
    schema_dir: Mapped[str] = mapped_column(Text, nullable=False)
    vector_collection_name: Mapped[str] = mapped_column(String(255), nullable=False)
    include_tables: Mapped[List[str]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    schema_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    schema_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class UserContext(Base):
    __tablename__ = "user_contexts"
    __table_args__ = {"schema": "app"}

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    context_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("app.db_contexts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")