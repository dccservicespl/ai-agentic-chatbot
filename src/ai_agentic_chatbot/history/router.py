"""Prompt history endpoints — View (free), Refresh (re-execute, no LLM), Regenerate (full re-run).

Refresh and Regenerate deliberately bypass the parent agent graph and router
entirely: Refresh has zero LLM calls in its path (calls execute_query_node
directly); Regenerate skips only the router (intent is already known from the
history row) but still runs the full SQL subgraph with force_regenerate=True.

Scope decision: Refresh never mutates the PromptHistory row it's invoked from,
or writes a new one — it's a pure "give me current results for this past
prompt" read with no persistence side effects. Regenerate, by contrast,
updates the existing PromptHistory row it's invoked from in place (never
inserting a new row) with the fresh generated_sql/chart_type/result_snapshot,
in addition to its existing side effect on app.prompt_cache (via
sync_prompt_cache).
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.dependencies import get_auth_db, get_current_user
from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.context.repository import get_context_by_id, get_context_by_slug
from ai_agentic_chatbot.history.repository import (
    get_history_item, list_history_for_thread, list_history_for_user, update_prompt_history_after_regenerate,
)
from ai_agentic_chatbot.history.schemas import (
    PromptHistoryDetail, PromptHistoryListItem, RefreshResponse, RegenerateResponse,
)
from ai_agentic_chatbot.agent.nodes.visualizer import VisualizationNode
from ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.execute_query import execute_query_node
from ai_agentic_chatbot.agent.subgraphs.sql_query.graph import sql_subgraph
from ai_agentic_chatbot.agent.subgraphs.sql_query.cache_sync import sync_prompt_cache
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/history", tags=["History"])


@router.get("", response_model=list[PromptHistoryListItem])
def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    # One row per thread_id (session), not per turn — see
    # list_history_for_user's docstring for the earliest/latest split.
    # Use GET /history/thread/{thread_id} for a session's full turn list.
    context_db_id = None
    if context_id is not None:
        ctx_row = get_context_by_slug(db, context_id)
        if ctx_row is None:
            raise HTTPException(status_code=404, detail=f"Unknown context: {context_id}")
        context_db_id = ctx_row.id
    rows = list_history_for_user(db, current_user.id, context_db_id=context_db_id, limit=limit, offset=offset)
    return [
        PromptHistoryListItem(
            id=r.id,
            thread_id=r.thread_id,
            db_context_id=r.db_context_id,
            raw_prompt=r.raw_prompt,
            chart_type=r.chart_type,
            was_cache_hit=r.was_cache_hit,
            executed_at=r.latest_activity,
        )
        for r in rows
    ]


@router.get("/thread/{thread_id}", response_model=list[PromptHistoryDetail])
def list_history_for_thread_endpoint(
    thread_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    # PromptHistoryDetail (not ListItem) — a session replay needs each turn's
    # result_snapshot (visualization + analysis), not just list metadata.
    context_db_id = None
    if context_id is not None:
        ctx_row = get_context_by_slug(db, context_id)
        if ctx_row is None:
            raise HTTPException(status_code=404, detail=f"Unknown context: {context_id}")
        context_db_id = ctx_row.id
    rows = list_history_for_thread(
        db, thread_id, current_user.id, context_db_id=context_db_id, limit=limit, offset=offset
    )
    return [PromptHistoryDetail.model_validate(r) for r in rows]


@router.get("/{history_id}", response_model=PromptHistoryDetail)
def get_history(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    row = get_history_item(db, history_id)
    if row is None or row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="History item not found")
    return PromptHistoryDetail.model_validate(row)


@router.post("/{history_id}/refresh", response_model=RefreshResponse)
def refresh_prompt(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    row = get_history_item(db, history_id)
    if row is None or row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="History item not found")
    if not row.generated_sql:
        raise HTTPException(status_code=422, detail="This history item has no stored SQL to refresh")

    ctx_row = get_context_by_id(db, row.db_context_id)
    if ctx_row is None:
        raise HTTPException(status_code=404, detail="Context for this history item no longer exists")

    exec_result = execute_query_node({
        "is_safe": True,
        "generated_sql": row.generated_sql,
        "db_context_id": ctx_row.context_id,
    })
    if exec_result.get("execution_error"):
        raise HTTPException(status_code=502, detail=f"Refresh failed: {exec_result['execution_error']}")

    viz_result = VisualizationNode().determine_visualization(
        {"query_result": exec_result.get("query_result", []), "generated_sql": row.generated_sql, "explanation": ""},
        forced_type=row.chart_type,
    )
    return RefreshResponse(visualization=viz_result["visualization"], generated_sql=row.generated_sql)


@router.post("/{history_id}/regenerate", response_model=RegenerateResponse)
def regenerate_prompt(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    row = get_history_item(db, history_id)
    if row is None or row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="History item not found")

    ctx_row = get_context_by_id(db, row.db_context_id)
    if ctx_row is None:
        raise HTTPException(status_code=404, detail="Context for this history item no longer exists")

    subgraph_input = {
        "user_query": row.raw_prompt,
        "router_table_hints": [],
        "db_context_id": ctx_row.context_id,
        "generation_attempts": 0,
        "max_retries": 2,
        "is_safe": False,
        "validation_errors": [],
        "retrieved_tables": None,
        "generated_sql": None,
        "explanation": None,
        "confidence": 0.0,
        "tables_used": [],
        "query_result": None,
        "execution_error": None,
        "force_regenerate": True,
    }
    subgraph_result = sql_subgraph.invoke(subgraph_input)

    if subgraph_result.get("validation_errors"):
        raise HTTPException(status_code=422, detail="; ".join(subgraph_result["validation_errors"]))
    if subgraph_result.get("execution_error"):
        raise HTTPException(status_code=502, detail=f"Regenerate failed: {subgraph_result['execution_error']}")

    viz_result = VisualizationNode().determine_visualization(subgraph_result)
    visualization = viz_result["visualization"]

    try:
        sync_prompt_cache(subgraph_result, visualization)
    except Exception as exc:
        logger.warning(f"Prompt cache sync failed during regenerate (non-fatal): {exc}")

    try:
        update_prompt_history_after_regenerate(
            db, row,
            generated_sql=subgraph_result.get("generated_sql", ""),
            chart_type=visualization.get("type"),
            result_snapshot=visualization,
        )
    except Exception as exc:
        logger.warning(f"Prompt history update failed during regenerate (non-fatal): {exc}")

    return RegenerateResponse(
        visualization=visualization,
        generated_sql=subgraph_result.get("generated_sql", ""),
        explanation=subgraph_result.get("explanation"),
    )