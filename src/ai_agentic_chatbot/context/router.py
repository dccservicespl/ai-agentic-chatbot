"""Context management endpoints — /context/admin and /context.

config.yaml is the source of truth for context *definitions* — app.db_contexts
is a DB mirror populated by seed_contexts_from_config() at startup. These
endpoints only manage which users can access which already-configured contexts;
there is deliberately no POST /context/admin to create new ones.

Depends on auth (get_current_user, get_auth_db, get_user_by_username) for
identity and session handling — auth must never depend back on context.
"""
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.dependencies import get_auth_db, get_current_user
from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.auth.repository import get_user_by_username
from ai_agentic_chatbot.context.repository import (
    assign_context_to_user,
    get_context_by_slug,
    get_contexts_for_user,
    get_default_context,
    list_active_contexts,
    remove_context_assignment,
    seed_contexts_from_config,
)
from ai_agentic_chatbot.context.schemas import ContextAdminResponse, UserContextResponse
from ai_agentic_chatbot.infrastructure.context.context_settings import reload_context_registry

router = APIRouter(prefix="/context", tags=["Context"])


@router.get(
    "/admin",
    response_model=list[ContextAdminResponse],
    summary="List all active database contexts",
    description=(
        "Superuser-only. Lists all active contexts mirrored from config.yaml "
        "into app.db_contexts. To add a new context, add it to config.yaml and "
        "call POST /context/admin/reload (or restart the service) — this "
        "endpoint does not create contexts."
    ),
)
def list_admin_contexts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser access required")
    return [ContextAdminResponse.model_validate(c) for c in list_active_contexts(db)]


@router.post(
    "/admin/reload",
    response_model=list[ContextAdminResponse],
    summary="Reload contexts from config.yaml without restarting the service",
    description=(
        "Superuser-only. Re-reads config.yaml from disk into the in-process "
        "DbContextRegistry, then upserts the result into app.db_contexts — the "
        "same two steps that normally only happen once, at process startup. "
        "Does not create PostgreSQL schemas/tables/grants or run the schema "
        "pipeline (/schemaJson, /schemaText, /ingest) — those are still "
        "separate manual steps for a brand-new context."
    ),
)
def reload_contexts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser access required")
    registry = reload_context_registry()
    seed_contexts_from_config(db, registry)
    return [ContextAdminResponse.model_validate(c) for c in list_active_contexts(db)]


@router.put(
    "/admin/{context_slug}/users/{username}",
    response_model=UserContextResponse,
    summary="Assign a context to a user",
    description=(
        "Superuser-only. Assigns the given context to the given user. "
        "Pass `is_default=true` to make it the user's default context — this "
        "automatically clears any previously assigned default for that user."
    ),
)
def assign_context(
    context_slug: str,
    username: str,
    is_default: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser access required")
    target_user = get_user_by_username(db, username)
    if target_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    ctx = get_context_by_slug(db, context_slug)
    if ctx is None or not ctx.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Context not found")

    assign_context_to_user(db, user_id=target_user.id, context_db_id=ctx.id, is_default=is_default)
    return UserContextResponse(context_id=ctx.context_id, display_name=ctx.display_name, is_default=is_default)


@router.delete(
    "/admin/{context_slug}/users/{username}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a user's access to a context",
    description="Superuser-only. Returns HTTP 404 if the user was not assigned to this context.",
)
def unassign_context(
    context_slug: str,
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser access required")
    target_user = get_user_by_username(db, username)
    if target_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    ctx = get_context_by_slug(db, context_slug)
    if ctx is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Context not found")

    removed = remove_context_assignment(db, user_id=target_user.id, context_db_id=ctx.id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User is not assigned to this context")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "",
    response_model=list[UserContextResponse],
    summary="List the calling user's assigned contexts",
    description=(
        "Returns the contexts assigned to the authenticated user, with an "
        "is_default flag — used by the frontend to render a context switcher."
    ),
)
def list_my_contexts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    contexts = get_contexts_for_user(db, current_user.id)
    default_ctx = get_default_context(db, current_user.id)
    default_id = default_ctx.id if default_ctx else None
    return [
        UserContextResponse(
            context_id=c.context_id,
            display_name=c.display_name,
            is_default=(c.id == default_id),
        )
        for c in contexts
    ]