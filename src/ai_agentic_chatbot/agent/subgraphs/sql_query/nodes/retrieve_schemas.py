"""Schema retrieval node for semantic table search."""

from typing import List, Tuple, Dict, Any
from langchain_core.runnables import RunnableConfig
from ai_agentic_chatbot.schema_extractor.schema_loader import get_schema_loader
from ai_agentic_chatbot.infrastructure.llm.factory import get_embedding
from ai_agentic_chatbot.infrastructure.llm.types import LLMProvider, ModelType
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


def retrieve_schemas_node(state: dict, config: RunnableConfig) -> dict:
    """
    Semantic retrieval of relevant table schemas.
    """
    logger.info("[Retrieve Schemas] Starting semantic search")

    user_query = state["user_query"]
    router_hints = state.get("router_table_hints", [])

    try:
        schema_loader = get_schema_loader()
        table_docs = schema_loader.get_table_docs_for_search()

        retrieved = _semantic_search(user_query, table_docs, router_hints)

        if not retrieved:
            logger.warning("No tables retrieved from semantic search")
            return {
                "retrieved_tables": [],
                "validation_errors": ["No relevant tables found for query"],
            }

        # Log retrieval results
        for table_name, _, score in retrieved:
            logger.info(f"  Retrieved: {table_name} (score: {score:.3f})")

        # Optionally expand with foreign key related tables
        expanded = _expand_related_tables(table_docs, retrieved)
        if len(expanded) > len(retrieved):
            logger.info(f"Expanded to {len(expanded)} tables (including related)")
            retrieved = expanded

        return {
            "retrieved_tables": retrieved,
            "is_safe": True,  # Schema retrieval is always safe
        }

    except Exception as e:
        logger.error(f"Schema retrieval failed: {e}", exc_info=True)
        return {
            "retrieved_tables": [],
            "validation_errors": [f"Schema retrieval error: {str(e)}"],
        }


def _semantic_search(
    query: str,
    table_docs: List[Dict[str, Any]],
    router_hints: List[str],
    k: int = 5,
    score_threshold: float = 0.3,
) -> List[Tuple[str, str, float]]:
    """Perform semantic search on table schemas."""
    try:
        # Get embedding model
        embedding_model = get_embedding(LLMProvider.AZURE_OPENAI, ModelType.EMBEDDING)

        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)

        # Generate embeddings for all tables
        table_embeddings = []
        for table_doc in table_docs:
            table_embedding = embedding_model.embed_query(table_doc["search_text"])
            table_embeddings.append(table_embedding)

        # Calculate similarity scores
        similarities = []
        for i, table_doc in enumerate(table_docs):
            # Calculate cosine similarity
            score = _cosine_similarity(query_embedding, table_embeddings[i])

            # Boost score if table is in router hints
            if router_hints and table_doc["name"] in router_hints:
                score *= 1.5  # 50% boost for router suggestions
                logger.info(f"Boosted score for router hint: {table_doc['name']}")

            similarities.append((table_doc["name"], table_doc["ddl"], score))

        # Sort by score and filter
        similarities.sort(key=lambda x: x[2], reverse=True)
        filtered = [
            (name, ddl, score)
            for name, ddl, score in similarities
            if score >= score_threshold
        ]

        # Return top k results
        return filtered[:k]

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        # Fallback to router hints if available
        if router_hints:
            logger.info("Falling back to router hints")
            fallback_results = []
            for table_doc in table_docs:
                if table_doc["name"] in router_hints:
                    fallback_results.append((table_doc["name"], table_doc["ddl"], 0.8))
            return fallback_results[:k]
        return []


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


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
