"""PromptHistory CRUD — view/refresh/regenerate endpoints read/write through here.

Kept separate from auth/repository.py (which owns PromptLog, the lightweight
per-request rate-limiting log) and context/repository.py (context assignment) —
prompt_history is its own domain: the durable, user-facing "past prompt" record.
"""
from typing import Optional

from sqlalchemy import Row, func, select
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.models import PromptHistory


def write_prompt_history(
    db: Session,
    *,
    user_id: int,
    thread_id: str,
    db_context_id: int,
    raw_prompt: str,
    prompt_cache_id: Optional[int] = None,
    generated_sql: Optional[str] = None,
    was_cache_hit: bool = False,
    chart_type: Optional[str] = None,
    result_snapshot: Optional[dict] = None,
) -> PromptHistory:
    row = PromptHistory(
        user_id=user_id,
        thread_id=thread_id,
        db_context_id=db_context_id,
        raw_prompt=raw_prompt,
        prompt_cache_id=prompt_cache_id,
        generated_sql=generated_sql,
        was_cache_hit=was_cache_hit,
        chart_type=chart_type,
        result_snapshot=result_snapshot,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_prompt_history_after_regenerate(
    db: Session,
    row: PromptHistory,
    *,
    generated_sql: str,
    chart_type: Optional[str],
    result_snapshot: Optional[dict],
) -> PromptHistory:
    row.generated_sql = generated_sql
    row.was_cache_hit = False
    row.chart_type = chart_type
    row.result_snapshot = result_snapshot
    db.commit()
    db.refresh(row)
    return row


def get_history_item(db: Session, history_id: int) -> Optional[PromptHistory]:
    return db.get(PromptHistory, history_id)


def list_history_for_user(
    db: Session, user_id: int, *, context_db_id: Optional[int] = None, limit: int = 20, offset: int = 0
) -> list[Row]:
    # Session-level list: one row per thread_id, not one row per turn.
    # DISTINCT ON (thread_id) + ORDER BY thread_id, executed_at ASC picks each
    # thread's earliest turn (its raw_prompt doubles as a stable session
    # title) — Postgres requires ORDER BY to start with the DISTINCT ON
    # column(s), which is why "most recent session first" can't be expressed
    # in this same ORDER BY. latest_activity is a window MAX computed before
    # DISTINCT ON collapses the rows, carried through to an outer query that
    # re-sorts by it — so an old thread with a fresh reply still bubbles up.
    latest_activity = func.max(PromptHistory.executed_at).over(
        partition_by=PromptHistory.thread_id
    ).label("latest_activity")

    inner = select(
        PromptHistory.id,
        PromptHistory.thread_id,
        PromptHistory.db_context_id,
        PromptHistory.raw_prompt,
        PromptHistory.chart_type,
        PromptHistory.was_cache_hit,
        latest_activity,
    ).where(PromptHistory.user_id == user_id)
    if context_db_id is not None:
        inner = inner.where(PromptHistory.db_context_id == context_db_id)
    inner = inner.distinct(PromptHistory.thread_id).order_by(
        PromptHistory.thread_id, PromptHistory.executed_at.asc()
    )
    subq = inner.subquery()

    stmt = select(subq).order_by(subq.c.latest_activity.desc()).limit(limit).offset(offset)
    return list(db.execute(stmt).all())


def list_history_for_thread(
    db: Session, thread_id: str, user_id: int, *, context_db_id: Optional[int] = None, limit: int = 20, offset: int = 0
) -> list[PromptHistory]:
    # user_id filters ownership here (not a post-query check) so a thread_id
    # from another user's session can never leak rows. Ascending order (oldest
    # first), unlike list_history_for_user's recency ordering, since this
    # reconstructs a session's message flow for chat replay.
    stmt = select(PromptHistory).where(
        PromptHistory.thread_id == thread_id,
        PromptHistory.user_id == user_id,
    )
    if context_db_id is not None:
        stmt = stmt.where(PromptHistory.db_context_id == context_db_id)
    return list(
        db.execute(
            stmt.order_by(PromptHistory.executed_at.asc()).limit(limit).offset(offset)
        ).scalars().all()
    )