"""add schema versions table

Revision ID: 98a43e44f4e9
Revises: c7b3a9f2e1d4
Create Date: 2026-07-08 00:00:00.000000

NOTE: Written by hand — do NOT replace with --autogenerate.
      alembic/env.py include_object cannot detect app-schema ORM models
      until Base.metadata is populated from context/models.py imports.
      This migration precedes Section 38's planned prompt_cache/prompt_history
      migration, whose down_revision should point at this revision instead
      of c7b3a9f2e1d4 once that work starts.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = '98a43e44f4e9'
down_revision: Union[str, Sequence[str], None] = 'c7b3a9f2e1d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── db_contexts: denormalized "current" pointer, fed by schema_versions ──
    op.add_column("db_contexts", sa.Column("schema_hash", sa.String(length=64), nullable=True), schema="app")
    op.add_column("db_contexts", sa.Column("schema_updated_at", sa.DateTime(timezone=True), nullable=True), schema="app")

    # ── schema_versions: append-only audit history ──────────────────────────
    op.create_table(
        "schema_versions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("db_context_id", sa.BigInteger(), nullable=False),
        sa.Column("schema_name", sa.String(length=100), nullable=False),
        sa.Column("schema_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("captured_by_user_id", sa.BigInteger(), nullable=True),
        sa.Column("table_count", sa.Integer(), nullable=True),
        sa.Column("column_count", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["db_context_id"], ["app.db_contexts.id"],
            ondelete="CASCADE",
            name="fk_schema_versions_context_id",
        ),
        sa.ForeignKeyConstraint(
            ["captured_by_user_id"], ["public.users.id"],
            ondelete="SET NULL",
            name="fk_schema_versions_captured_by",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="app",
    )
    op.create_index(
        "ix_schema_versions_context_captured",
        "schema_versions",
        ["db_context_id", "captured_at"],
        schema="app",
    )
    op.create_index(
        "ix_schema_versions_schema_name",
        "schema_versions",
        ["schema_name", "captured_at"],
        schema="app",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_schema_versions_schema_name",
        table_name="schema_versions",
        schema="app",
    )
    op.drop_index(
        "ix_schema_versions_context_captured",
        table_name="schema_versions",
        schema="app",
    )
    op.drop_table("schema_versions", schema="app")
    op.drop_column("db_contexts", "schema_updated_at", schema="app")
    op.drop_column("db_contexts", "schema_hash", schema="app")