"""add prompt_history composite thread_id/user_id index

Revision ID: 87b3bf2d60cf
Revises: d2f0915eecc3
Create Date: 2026-07-09 00:00:00.000000

NOTE: Written by hand, matching d2f0915eecc3's precedent (alembic/env.py
      include_object still can't autogenerate app-schema ORM models).
      Supports GET /history/thread/{thread_id}, which filters on both
      columns together (thread_id AND user_id, for ownership) — the
      existing ix_prompt_history_thread_id (thread_id alone) and
      ix_prompt_history_user_executed (user_id, executed_at) each only
      cover half of that predicate.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '87b3bf2d60cf'
down_revision: Union[str, Sequence[str], None] = 'd2f0915eecc3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_prompt_history_thread_user",
        "prompt_history",
        ["thread_id", "user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_prompt_history_thread_user", table_name="prompt_history")