"""Schema retrieval node for semantic table search."""

import os
from typing import List, Tuple, Dict, Any
from ai_agentic_chatbot.infrastructure.embedding.embedding_connection import (
    get_azure_openai_embedding,
)
from ai_agentic_chatbot.infrastructure.llm.config import AzureOpenAIEmbeddingConfig
from langchain_core.runnables import RunnableConfig
from ai_agentic_chatbot.schema_extractor.schema_loader import get_schema_loader
from ai_agentic_chatbot.infrastructure.llm.factory import get_embedding
from ai_agentic_chatbot.infrastructure.llm.types import LLMProvider, ModelType
from ai_agentic_chatbot.logging_config import get_logger
from langchain_openai import AzureOpenAIEmbeddings

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


# def _semantic_search(
#     query: str,
#     table_docs: List[Dict[str, Any]],
#     router_hints: List[str],
#     k: int = 5,
#     score_threshold: float = 0.3,
# ) -> List[Tuple[str, str, float]]:
#     """Enhanced semantic search with intelligent query boosting

#     Args:
#         query (str): user query
#         table_docs (List[Dict[str, Any]]): list of table documents
#         router_hints (List[str]): list of router hints
#         k (int, optional): number of tables to return. Defaults to 5.
#         score_threshold (float, optional): minimum similarity score. Defaults to 0.3.

#     Returns:
#         List[Tuple[str, str, float]]: list of (table_name, table_description, similarity_score)
#     """


def _semantic_search(
    query: str,
    table_docs: List[Dict[str, Any]],
    router_hints: List[str],
    k: int = 5,
    score_threshold: float = 0.3,
) -> List[Tuple[str, str, float]]:
    """
    Enhanced semantic search using your preprocessed schema with business context.
    Leverages example questions, business purpose, and field meanings for better matching.
    """
    try:
        # TODO: Get embedding model from factory
        embedding_model = get_azure_openai_embedding()

        query_embedding = embedding_model.embed_query(query)

        # Calculate similarity scores with enhanced context matching
        similarities = []
        for table_doc in table_docs:
            # Multi-level semantic matching
            scores = []

            # 1. Match against example questions (highest weight)
            example_questions = table_doc.get("example_questions", [])
            if example_questions:
                for question in example_questions:
                    question_embedding = embedding_model.embed_query(question)
                    question_score = _cosine_similarity(
                        query_embedding, question_embedding
                    )
                    scores.append(
                        question_score * 2.0
                    )  # 2x weight for example questions

            # 2. Match against business purpose (high weight)
            business_purpose = table_doc.get("business_purpose", "")
            if business_purpose:
                purpose_embedding = embedding_model.embed_query(business_purpose)
                purpose_score = _cosine_similarity(query_embedding, purpose_embedding)
                scores.append(purpose_score * 1.5)  # 1.5x weight for business purpose

            # 3. Match against full search text (base weight)
            search_text = table_doc.get("search_text", "")
            if search_text:
                text_embedding = embedding_model.embed_query(search_text)
                text_score = _cosine_similarity(query_embedding, text_embedding)
                scores.append(text_score)

            # 4. Match against key field meanings (medium weight)
            key_fields = table_doc.get("key_fields", [])
            for field in key_fields:
                field_meaning = field.get("meaning", "")
                if field_meaning:
                    field_embedding = embedding_model.embed_query(field_meaning)
                    field_score = _cosine_similarity(query_embedding, field_embedding)
                    scores.append(field_score * 1.2)  # 1.2x weight for field meanings

            # Take the maximum score from all matches
            final_score = max(scores) if scores else 0.0

            # Router hint boost
            if router_hints and table_doc["name"] in router_hints:
                final_score *= 1.3  # 30% boost for router suggestions
                logger.info(
                    f"Router hint boost for {table_doc['name']}: {final_score:.3f}"
                )

            # Relationship boost - if query mentions related tables
            relationships = table_doc.get("relationships", [])
            if relationships:
                query_lower = query.lower()
                for rel in relationships:
                    related_table = rel.get("related_table", "").lower()
                    if related_table in query_lower:
                        final_score *= 1.2  # 20% boost for relationship context
                        logger.info(
                            f"Relationship boost for {table_doc['name']} -> {related_table}"
                        )

            similarities.append((table_doc["name"], table_doc["ddl"], final_score))

        # Sort by score and filter
        similarities.sort(key=lambda x: x[2], reverse=True)
        filtered = [
            (name, ddl, score)
            for name, ddl, score in similarities
            if score >= score_threshold
        ]

        # Log top matches for debugging
        logger.info("Top semantic matches:")
        for name, _, score in filtered[:3]:
            logger.info(f"  {name}: {score:.3f}")

        # Return top k results
        return filtered[:k]

    except Exception as e:
        logger.error(f"Enhanced semantic search failed: {e}")
        # Fallback to router hints if available
        if router_hints:
            fallback_results = []
            for table_doc in table_docs:
                if table_doc["name"] in router_hints:
                    fallback_results.append((table_doc["name"], table_doc["ddl"], 0.8))
            return fallback_results[:k]


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
