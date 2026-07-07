"""Schema retrieval node for semantic table search."""

from pathlib import Path

from typing import List, Tuple, Dict, Any
from langchain_core.runnables import RunnableConfig
from ai_agentic_chatbot.schema_extractor.schema_loader import get_schema_loader
from ai_agentic_chatbot.logging_config import get_logger
from ai_agentic_chatbot.infrastructure.vector_store.pgvector_store import PgVectorSchemaStore
from ai_agentic_chatbot.infrastructure.context.context_settings import get_context_registry

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[6]

# TODO 18 (not yet done): /stream will inject the real db_context_id. Until
# then, every request resolves against this one real, configured context —
# "default" is not a registered context_id in config.yaml.
_FALLBACK_CONTEXT_ID = "sales"


def retrieve_schemas_node(state: dict, config: RunnableConfig) -> dict:
    """
    Semantic retrieval of relevant table schemas.
    """
    logger.info("[Retrieve Schemas] Starting semantic search")

    user_query = state["user_query"]
    router_hints = state.get("router_table_hints", [])
    db_context_id = state.get("db_context_id") or _FALLBACK_CONTEXT_ID

    try:
        ctx = get_context_registry().get_context(db_context_id)
        schema_loader = get_schema_loader(context_id=db_context_id, schema_dir=PROJECT_ROOT / ctx.schema_dir)
        table_docs = schema_loader.get_table_docs_for_search()

        retrieved = _semantic_searchV2(user_query, table_docs, router_hints, ctx.vector_collection_name)

        if not retrieved:
            logger.warning("No tables retrieved from semantic search")
            return {
                "retrieved_tables": [],
                "validation_errors": ["No relevant tables found for query"],
            }

        for table_name, _, score in retrieved:
            logger.info(f"  Retrieved: {table_name} (score: {score:.3f})")

        expanded = _expand_related_tables(table_docs, retrieved)
        if len(expanded) > len(retrieved):
            logger.info(f"Expanded to {len(expanded)} tables (including related)")
            retrieved = expanded

        return {
            "retrieved_tables": retrieved,
            "is_safe": True,
        }

    except Exception as e:
        logger.error(f"Schema retrieval failed: {e}", exc_info=True)
        return {
            "retrieved_tables": [],
            "validation_errors": [f"Schema retrieval error: {str(e)}"],
        }


def _semantic_searchV2(
    query: str,
    table_docs: List[Dict[str, Any]],
    router_hints: List[str],
    collection_name: str,
    k: int = 5,
    score_threshold: float = 0.3,
) -> List[Tuple[str, str, float]]:
    """Semantic search using pgvector stored embeddings — 1 API call per query.

    Replaces _semantic_search() which re-embedded all schema text on every request
    (~102 API calls). This version calls embed_query() exactly once (internally inside
    similarity_search_with_relevance_scores), then compares against pre-computed vectors
    stored in pgvector at /ingest time.

    Falls back to router hints if pgvector raises (e.g. /ingest was never run).
    """
    try:
        # Build a lookup dict for DDL and relationship data from SchemaLoader.
        # get_table_docs_for_search() is a file read — zero API calls.
        table_doc_map = {doc["name"]: doc for doc in table_docs}

        # Single API call: pgvector embeds the query internally and compares
        # against all stored schema vectors using cosine similarity in SQL.
        # Returns (table_name, relevance_score) sorted descending, already filtered
        # by score_threshold. Scores are 0-1 (higher = more similar).
        vector_store = PgVectorSchemaStore(collection_name=collection_name)
        pgvector_results = vector_store.search(query, k=k, score_threshold=score_threshold)

        if not pgvector_results:
            logger.warning("pgvector returned no results above threshold — falling back to router hints")
            return _router_hint_fallback(table_doc_map, router_hints, k)

        results: List[Tuple[str, str, float]] = []
        for table_name, score in pgvector_results:
            table_doc = table_doc_map.get(table_name)
            if not table_doc:
                logger.warning(f"pgvector returned table '{table_name}' not found in schema loader — skipping")
                continue

            final_score = score

            # Router hint boost — 30% score bump for tables the router identified.
            # Pure math, zero API calls.
            if router_hints and table_name in router_hints:
                final_score *= 1.3
                logger.info(f"Router hint boost for {table_name}: {final_score:.3f}")

            # Relationship boost — if the query explicitly names a related table.
            # Pure string match, zero API calls.
            relationships = table_doc.get("relationships") or []
            query_lower = query.lower()
            for rel in relationships:
                related_table = rel.get("related_table", "").lower()
                if related_table and related_table in query_lower:
                    final_score *= 1.2
                    logger.info(f"Relationship boost for {table_name} -> {rel.get('related_table')}")

            results.append((table_name, table_doc["ddl"], final_score))

        results.sort(key=lambda x: x[2], reverse=True)

        logger.info("Top semantic matches (V2):")
        for name, _, score in results[:3]:
            logger.info(f"  {name}: {score:.3f}")

        return results[:k]

    except Exception as e:
        logger.error(f"_semantic_searchV2 failed: {e}", exc_info=True)
        table_doc_map = {doc["name"]: doc for doc in table_docs}
        return _router_hint_fallback(table_doc_map, router_hints, k)


def _router_hint_fallback(
    table_doc_map: Dict[str, Any],
    router_hints: List[str],
    k: int,
) -> List[Tuple[str, str, float]]:
    """Return router-hinted tables with a fixed score when vector search is unavailable."""
    if not router_hints:
        return []
    results = []
    for table_name in router_hints:
        table_doc = table_doc_map.get(table_name)
        if table_doc:
            results.append((table_name, table_doc["ddl"], 0.8))
    return results[:k]


def _expand_related_tables(
    table_docs: List[Dict[str, Any]], retrieved: List[Tuple[str, str, float]]
) -> List[Tuple[str, str, float]]:
    """Helper: Add FK-related tables."""
    retrieved_names = {name for name, _, _ in retrieved}
    expanded = list(retrieved)

    # Find tables that reference or are referenced by retrieved tables
    for table_doc in table_docs:
        if table_doc["name"] in retrieved_names:
            continue

        # Check if this table has relationships with retrieved tables
        table_ddl = table_doc["ddl"].upper()
        for retrieved_name, _, _ in retrieved:
            if f"REFERENCES {retrieved_name.upper()}" in table_ddl:
                # This table references a retrieved table
                expanded.append((table_doc["name"], table_doc["ddl"], 0.5))
                logger.info(
                    f"Added related table: {table_doc['name']} -> {retrieved_name}"
                )
                break

    return expanded
