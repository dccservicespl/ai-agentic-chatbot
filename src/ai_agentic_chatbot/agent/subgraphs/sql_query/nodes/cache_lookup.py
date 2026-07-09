"""Prompt-cache lookup node — subgraph entry point."""

from sqlalchemy import select

from ai_agentic_chatbot.infrastructure.datasource.factory import get_session
from ai_agentic_chatbot.context.repository import get_context_by_slug
from ai_agentic_chatbot.context.prompt_cache_models import PromptCache
from ai_agentic_chatbot.utils.text_normalize import normalize_prompt
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)

# TODO 18 (not yet done): /stream will inject the real db_context_id. Until
# then, every request resolves against this one real, configured context —
# "default" is not a registered context_id in config.yaml.
_FALLBACK_CONTEXT_ID = "sales"


def cache_lookup_node(state: dict) -> dict:
    """
    Node 1 (entry point): Look up a previously cached SQL generation for this
    normalized prompt/context/schema combination.

    On a hit, returns the cached SQL/explanation directly (already validated
    at write time) so the subgraph can route straight to execute_query,
    skipping retrieve_schemas -> generate_sql -> validate_query entirely.
    On a miss, returns just enough state for downstream nodes/route to fall
    through to normal retrieval.
    """
    normalized = normalize_prompt(state["user_query"])
    db_context_slug = state.get("db_context_id") or _FALLBACK_CONTEXT_ID
    force_regenerate = state.get("force_regenerate", False)

    db = get_session("postgresql.primary")
    try:
        ctx_row = get_context_by_slug(db, db_context_slug)
        schema_hash = ctx_row.schema_hash if ctx_row else None

        if ctx_row is None or schema_hash is None or force_regenerate:
            return {"normalized_prompt": normalized, "schema_hash": schema_hash, "cache_hit": False}

        cache_row = db.execute(
            select(PromptCache).where(
                PromptCache.normalized_prompt == normalized,
                PromptCache.db_context_id == ctx_row.id,
                PromptCache.schema_hash == schema_hash,
            )
        ).scalar_one_or_none()

        if cache_row is None:
            return {"normalized_prompt": normalized, "schema_hash": schema_hash, "cache_hit": False}

        logger.info(f"[Cache Lookup] Hit for normalized prompt (context={db_context_slug})")
        return {
            "normalized_prompt": normalized,
            "schema_hash": schema_hash,
            "cache_hit": True,
            "cache_row_id": cache_row.id,
            "generated_sql": cache_row.generated_sql,
            "explanation": cache_row.explanation,
            "confidence": 1.0,
            "is_safe": True,
            "cached_chart_type": cache_row.chart_type,
            "cached_chart_config": cache_row.chart_config,
        }
    except Exception as e:
        logger.error(f"[Cache Lookup] failed, falling through to normal generation: {e}", exc_info=True)
        return {"normalized_prompt": normalized, "cache_hit": False}
    finally:
        db.close()