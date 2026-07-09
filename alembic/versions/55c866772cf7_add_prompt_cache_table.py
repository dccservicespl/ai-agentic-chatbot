"""add prompt cache table

Revision ID: 55c866772cf7
Revises: 98a43e44f4e9
Create Date: 2026-07-08 00:00:00.000000

NOTE: Written by hand — do NOT replace with --autogenerate.
      alembic/env.py include_object cannot detect app-schema ORM models
      until Base.metadata is populated from context/models.py imports.
      This is Phase 1a of Section 38's prompt-caching plan (renumbered —
      the original 1a, adding schema_hash/schema_updated_at to db_contexts,
      was superseded by Section 39 and shipped in migration 98a43e44f4e9).
      Phase 1b (the per-user prompt_history table, which FKs onto
      app.prompt_cache.id) is a separate follow-up migration chained on
      top of this one — not included here.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op


revision: str = '55c866772cf7'
down_revision: Union[str, Sequence[str], None] = '98a43e44f4e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── prompt_cache: shared, SQL-only cache keyed on (prompt, context, schema) ──
    op.create_table(
        "prompt_cache",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("normalized_prompt", sa.Text(), nullable=False),
        sa.Column("db_context_id", sa.BigInteger(), nullable=False),
        sa.Column("schema_hash", sa.String(length=64), nullable=False),
        sa.Column("generated_sql", sa.Text(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("chart_type", sa.String(length=50), nullable=True),
        sa.Column("chart_config", JSONB(), nullable=True),
        sa.Column("result_columns", JSONB(), nullable=True),
        sa.Column("hit_count", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column(
            "last_used_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["db_context_id"], ["app.db_contexts.id"],
            ondelete="CASCADE",
            name="fk_prompt_cache_context_id",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "normalized_prompt", "db_context_id", "schema_hash",
            name="uq_prompt_cache_key",
        ),
        schema="app",
    )
    op.create_index(
        "ix_prompt_cache_context_schema",
        "prompt_cache",
        ["db_context_id", "schema_hash"],
        schema="app",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_prompt_cache_context_schema",
        table_name="prompt_cache",
        schema="app",
    )
    op.drop_table("prompt_cache", schema="app")
