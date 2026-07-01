"""add app schema and context tables

Revision ID: c7b3a9f2e1d4
Revises: 1ab190c811b5
Create Date: 2026-07-01 00:00:00.000000

NOTE: Written by hand — do NOT replace with --autogenerate.
      alembic/env.py include_object cannot detect app-schema ORM models
      until Base.metadata is populated from context/models.py imports.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op


revision: str = 'c7b3a9f2e1d4'
down_revision: Union[str, Sequence[str], None] = '1ab190c811b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Schemas ──────────────────────────────────────────────────────────────
    op.execute("CREATE SCHEMA IF NOT EXISTS app")
    op.execute("CREATE SCHEMA IF NOT EXISTS vector_store")

    # ── db_contexts ──────────────────────────────────────────────────────────
    op.create_table(
        "db_contexts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("context_id", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("schema_name", sa.String(length=100), nullable=False),
        sa.Column("system_prompt_path", sa.Text(), nullable=False),
        sa.Column("router_prompt_path", sa.Text(), nullable=False),
        sa.Column("schema_dir", sa.Text(), nullable=False),
        sa.Column("vector_collection_name", sa.String(length=255), nullable=False),
        sa.Column("include_tables", JSONB(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("context_id", name="uq_db_contexts_context_id"),
        schema="app",
    )

    # ── user_contexts ─────────────────────────────────────────────────────────
    op.create_table(
        "user_contexts",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("context_id", sa.BigInteger(), nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["public.users.id"],
            ondelete="CASCADE",
            name="fk_user_contexts_user_id",
        ),
        sa.ForeignKeyConstraint(
            ["context_id"], ["app.db_contexts.id"],
            ondelete="CASCADE",
            name="fk_user_contexts_context_id",
        ),
        sa.PrimaryKeyConstraint("user_id", "context_id"),
        schema="app",
    )

    # Partial unique index — only one default context allowed per user
    op.create_index(
        "uq_user_default_context",
        "user_contexts",
        ["user_id"],
        schema="app",
        unique=True,
        postgresql_where=text("is_default = TRUE"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_user_default_context",
        table_name="user_contexts",
        schema="app",
    )
    op.drop_table("user_contexts", schema="app")
    op.drop_table("db_contexts", schema="app")
    op.execute("DROP SCHEMA IF EXISTS vector_store")
    op.execute("DROP SCHEMA IF EXISTS app")