"""add prompt history table

Revision ID: d2f0915eecc3
Revises: 55c866772cf7
Create Date: 2026-07-08 00:00:00.000000

NOTE: Written by hand — do NOT replace with --autogenerate.
      alembic/env.py include_object cannot detect app-schema ORM models
      until Base.metadata is populated from context/models.py imports.
      This is Phase 1b of Section 38's caching plan, chained on top of
      Phase 1a's prompt_cache migration (55c866772cf7). This table is
      deliberately separate from the existing prompt_logs/PromptLog table
      (auth/models.py) — PromptLog/create_prompt_log/count_prompts_today
      are untouched, per Section 38's decision documented in
      technical_doc.md (bolting SQL-specific columns onto prompt_logs
      would leave them null on most rows since it's written for every
      message, not just SQL queries, and risks the daily-rate-limit
      query).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op


revision: str = 'd2f0915eecc3'
down_revision: Union[str, Sequence[str], None] = '55c866772cf7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── prompt_history: per-user log of prompts/results (View is free) ──
    op.create_table(
        "prompt_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("db_context_id", sa.BigInteger(), nullable=False),
        sa.Column("raw_prompt", sa.Text(), nullable=False),
        sa.Column("prompt_cache_id", sa.BigInteger(), nullable=True),
        sa.Column("generated_sql", sa.Text(), nullable=True),
        sa.Column("was_cache_hit", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("chart_type", sa.String(length=50), nullable=True),
        sa.Column("result_snapshot", JSONB(), nullable=True),
        sa.Column(
            "executed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            ondelete="CASCADE",
            name="fk_prompt_history_user_id",
        ),
        sa.ForeignKeyConstraint(
            ["db_context_id"], ["app.db_contexts.id"],
            ondelete="CASCADE",
            name="fk_prompt_history_context_id",
        ),
        sa.ForeignKeyConstraint(
            ["prompt_cache_id"], ["app.prompt_cache.id"],
            ondelete="SET NULL",
            name="fk_prompt_history_cache_id",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_prompt_history_user_executed",
        "prompt_history",
        ["user_id", "executed_at"],
    )
    op.create_index(
        "ix_prompt_history_thread_id",
        "prompt_history",
        ["thread_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_prompt_history_user_executed", table_name="prompt_history")
    op.drop_index("ix_prompt_history_thread_id", table_name="prompt_history")
    op.drop_table("prompt_history")
