"""Context CRUD — user-to-context assignment operations.

Kept separate from auth/repository.py: that module owns JWT tokens, user
management, refresh token lifecycle, and prompt logging. Context management
is a distinct domain.
"""
from typing import Optional

from sqlalchemy import select, update, func
from sqlalchemy.orm import Session

from ai_agentic_chatbot.context.models import DbContext, UserContext
from ai_agentic_chatbot.context.schema_version_models import SchemaVersion
from ai_agentic_chatbot.infrastructure.context.context_settings import DbContextRegistry


def get_contexts_for_user(db: Session, user_id: int) -> list[DbContext]:
    return list(
        db.execute(
            select(DbContext)
            .join(UserContext, UserContext.context_id == DbContext.id)
            .where(UserContext.user_id == user_id, DbContext.is_active.is_(True))
            .order_by(DbContext.display_name)
        ).scalars().all()
    )


def get_default_context(db: Session, user_id: int) -> Optional[DbContext]:
    return db.execute(
        select(DbContext)
        .join(UserContext, UserContext.context_id == DbContext.id)
        .where(UserContext.user_id == user_id, UserContext.is_default.is_(True))
    ).scalar_one_or_none()


def get_context_by_slug(db: Session, context_id: str) -> Optional[DbContext]:
    return db.execute(
        select(DbContext).where(DbContext.context_id == context_id)
    ).scalar_one_or_none()


def is_user_assigned_to_context(db: Session, user_id: int, context_db_id: int) -> bool:
    return db.get(UserContext, (user_id, context_db_id)) is not None


def list_active_contexts(db: Session) -> list[DbContext]:
    return list(
        db.execute(
            select(DbContext).where(DbContext.is_active.is_(True)).order_by(DbContext.display_name)
        ).scalars().all()
    )


def remove_context_assignment(db: Session, *, user_id: int, context_db_id: int) -> bool:
    assignment = db.get(UserContext, (user_id, context_db_id))
    if assignment is None:
        return False
    db.delete(assignment)
    db.commit()
    return True


def assign_context_to_user(
    db: Session,
    *,
    user_id: int,
    context_db_id: int,
    is_default: bool = False,
) -> UserContext:
    if is_default:
        # Clear any existing default first — the partial unique index
        # (uq_user_default_context) only allows one is_default=TRUE row per user.
        db.execute(
            update(UserContext)
            .where(UserContext.user_id == user_id, UserContext.is_default.is_(True))
            .values(is_default=False)
        )

    assignment = db.get(UserContext, (user_id, context_db_id))
    if assignment is not None:
        assignment.is_default = is_default
    else:
        assignment = UserContext(user_id=user_id, context_id=context_db_id, is_default=is_default)
        db.add(assignment)

    db.commit()
    db.refresh(assignment)
    return assignment


def record_schema_version(
    db: Session,
    *,
    context_db_id: int,
    schema_name: str,
    schema_hash: str,
    captured_by_user_id: int,
    table_count: int,
    column_count: int,
) -> SchemaVersion:
    """Appends an audit row AND updates db_contexts' current-value pointer, in one transaction.

    schema_name is captured here (not joined later) so history rows stay correct even if a
    context's db_contexts.schema_name is ever repointed.
    """
    version = SchemaVersion(
        db_context_id=context_db_id,
        schema_name=schema_name,
        schema_hash=schema_hash,
        captured_by_user_id=captured_by_user_id,
        table_count=table_count,
        column_count=column_count,
    )
    db.add(version)
    db.execute(
        update(DbContext)
        .where(DbContext.id == context_db_id)
        .values(schema_hash=schema_hash, schema_updated_at=func.now())
    )
    db.commit()
    db.refresh(version)
    return version


def seed_contexts_from_config(db: Session, registry: DbContextRegistry) -> None:
    """Upsert config.yaml contexts into app.db_contexts.

    config.yaml is the source of truth; this keeps db_contexts in sync with
    it on every startup. Never insert a context here that isn't in config.yaml.
    """
    for ctx in registry.contexts.values():
        row = db.execute(
            select(DbContext).where(DbContext.context_id == ctx.context_id)
        ).scalar_one_or_none()

        display_name = ctx.display_name or ctx.context_id

        if row is None:
            db.add(DbContext(
                context_id=ctx.context_id,
                display_name=display_name,
                schema_name=ctx.schema_name,
                system_prompt_path=ctx.system_prompt_path,
                router_prompt_path=ctx.router_prompt_path,
                schema_dir=ctx.schema_dir,
                vector_collection_name=ctx.vector_collection_name,
                include_tables=ctx.include_tables,
            ))
        else:
            row.display_name = display_name
            row.schema_name = ctx.schema_name
            row.system_prompt_path = ctx.system_prompt_path
            row.router_prompt_path = ctx.router_prompt_path
            row.schema_dir = ctx.schema_dir
            row.vector_collection_name = ctx.vector_collection_name
            row.include_tables = ctx.include_tables

    db.commit()