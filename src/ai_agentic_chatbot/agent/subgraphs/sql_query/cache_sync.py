"""Writes/updates app.prompt_cache after a SQL subgraph run completes.

Called as a plain function (not a graph node) from agent/graph.py's
sql_query_node, after visualizer_node runs — chart_type/chart_config are
only known once the parent graph computes `visualization`, which never
happens inside the subgraph itself.
"""
from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ai_agentic_chatbot.infrastructure.datasource.factory import get_session
from ai_agentic_chatbot.context.repository import get_context_by_slug
from ai_agentic_chatbot.context.prompt_cache_models import PromptCache
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)

_FALLBACK_CONTEXT_ID = "sales"


def sync_prompt_cache(subgraph_result: dict, visualization: dict) -> None:
    """No-op on any failed generation/execution or when schema_hash is unknown
    (context never had /schemaJson run) — never cache a failed or unkeyable result."""
    if subgraph_result.get("execution_error") or subgraph_result.get("validation_errors"):
        return

    normalized_prompt = subgraph_result.get("normalized_prompt")
    schema_hash = subgraph_result.get("schema_hash")
    generated_sql = subgraph_result.get("generated_sql")
    if not normalized_prompt or not schema_hash or not generated_sql:
        return

    db_context_slug = subgraph_result.get("db_context_id") or _FALLBACK_CONTEXT_ID
    chart_type = visualization.get("type") if visualization else None
    # Only the small axis/column-mapping sub-dict — never row data (that's
    # per-user, lives in prompt_history instead, not in this shared cache).
    chart_config = visualization.get("config") if visualization else None
    result_columns = visualization.get("columns") if visualization else None

    db = get_session("postgresql.primary")
    try:
        ctx_row = get_context_by_slug(db, db_context_slug)
        if ctx_row is None:
            return

        if subgraph_result.get("cache_hit"):
            cache_row_id = subgraph_result.get("cache_row_id")
            if cache_row_id:
                db.execute(
                    update(PromptCache)
                    .where(PromptCache.id == cache_row_id)
                    .values(hit_count=PromptCache.hit_count + 1, last_used_at=func.now())
                )
        else:
            stmt = pg_insert(PromptCache).values(
                normalized_prompt=normalized_prompt,
                db_context_id=ctx_row.id,
                schema_hash=schema_hash,
                generated_sql=generated_sql,
                explanation=subgraph_result.get("explanation"),
                chart_type=chart_type,
                chart_config=chart_config,
                result_columns=result_columns,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["normalized_prompt", "db_context_id", "schema_hash"],
                set_=dict(
                    generated_sql=stmt.excluded.generated_sql,
                    explanation=stmt.excluded.explanation,
                    chart_type=stmt.excluded.chart_type,
                    chart_config=stmt.excluded.chart_config,
                    result_columns=stmt.excluded.result_columns,
                    hit_count=PromptCache.hit_count + 1,
                    last_used_at=func.now(),
                ),
            )
            db.execute(stmt)
        db.commit()
    except Exception as e:
        logger.error(f"[Cache Sync] failed to persist prompt_cache row: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()