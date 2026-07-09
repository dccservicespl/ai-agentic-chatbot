"""add prompt_history composite thread_id/executed_at index

Revision ID: 9df84a1b2e49
Revises: 87b3bf2d60cf
Create Date: 2026-07-10 00:00:00.000000

NOTE: Written by hand, matching d2f0915eecc3/87b3bf2d60cf's precedent
      (alembic/env.py include_object still can't autogenerate app-schema ORM
      models). Supports GET /history's new session-grouped query
      (list_history_for_user in history/repository.py), which does
      DISTINCT ON (thread_id) ORDER BY thread_id, executed_at ASC plus a
      MAX(executed_at) OVER (PARTITION BY thread_id) window — neither the
      existing ix_prompt_history_thread_id (thread_id alone) nor
      ix_prompt_history_thread_user (thread_id, user_id) covers an
      executed_at-ordered scan within each thread_id group.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '9df84a1b2e49'
down_revision: Union[str, Sequence[str], None] = '87b3bf2d60cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_prompt_history_thread_executed",
        "prompt_history",
        ["thread_id", "executed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_prompt_history_thread_executed", table_name="prompt_history")