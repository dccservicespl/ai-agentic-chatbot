"""Routing logic for SQL query subgraph with retry mechanism."""

from typing import Literal
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


def route_after_cache_lookup(state: dict) -> Literal["execute_query", "retrieve_schemas"]:
    """Route after cache lookup — hit skips straight to execution, miss falls through to normal retrieval."""
    if state.get("cache_hit"):
        logger.info("Cache hit - skipping retrieval/generation/validation")
        return "execute_query"
    return "retrieve_schemas"


def route_after_retrieval(state: dict) -> Literal["generate_sql", "END"]:
    """Route after schema retrieval."""
    retrieved_tables = state.get("retrieved_tables", [])

    if not retrieved_tables:
        logger.warning("No tables retrieved - ending subgraph")
        return "END"

    logger.info(f"Retrieved {len(retrieved_tables)} tables - proceeding to generation")
    return "generate_sql"


def route_after_generation(state: dict) -> Literal["validate_query", "END"]:
    """Route after SQL generation."""
    generated_sql = state.get("generated_sql")
    validation_errors = state.get("validation_errors", [])

    if validation_errors:
        logger.warning(f"Generation errors: {validation_errors} - ending subgraph")
        return "END"

    if not generated_sql:
        logger.warning("No SQL generated - ending subgraph")
        return "END"

    logger.info("SQL generated successfully - proceeding to validation")
    return "validate_query"


def route_after_validation(state: dict) -> Literal["execute_query", "END"]:
    """Route after validation."""
    is_safe = state.get("is_safe", False)
    validation_errors = state.get("validation_errors", [])

    if not is_safe:
        logger.warning(f"Query unsafe: {validation_errors} - ending subgraph")
        return "END"

    logger.info("Query validated successfully - proceeding to execution")
    return "execute_query"


def route_after_execution(state: dict) -> Literal["generate_sql", "retrieve_schemas", "END"]:
    """Route after execution - implements retry logic."""
    execution_error = state.get("execution_error")
    generation_attempts = state.get("generation_attempts", 0)
    max_retries = state.get("max_retries", 2)
    error_category = state.get("error_category", "unknown")

    if not execution_error:
        logger.info("✅ Execution successful - ending subgraph")
        return "END"

    # A cache-hit that fails execution with a schema-related error is treated as a
    # stale hit (schema drifted since the row was cached) and falls through to full
    # regeneration exactly once. Guard: generation_attempts is still 0 here because
    # generate_sql_node never ran on the cache-hit path (retrieve_schemas was
    # skipped) — after this fallback runs once, generation_attempts becomes >=1,
    # so this branch can never fire twice for the same request (no loop risk).
    if (
        state.get("cache_hit")
        and error_category == "not_found"
        and generation_attempts == 0
    ):
        logger.warning("Cache hit failed with not_found error - falling back to full regeneration")
        return "retrieve_schemas"

    if generation_attempts >= max_retries:
        logger.warning(f"Max retries ({max_retries}) exceeded - ending subgraph")
        return "END"

    if not _is_retryable_error(error_category):
        logger.warning(
            f"Non-retryable error category '{error_category}' - ending subgraph"
        )
        return "END"

    logger.info(
        f"Retrying generation (attempt {generation_attempts + 1}/{max_retries}) for {error_category} error"
    )
    return "generate_sql"


def _is_retryable_error(error_category: str) -> bool:
    """Determine if an error category is worth retrying."""
    retryable_categories = {"syntax", "not_found", "type", "unknown"}

    non_retryable_categories = {"permission", "connection"}

    if error_category in retryable_categories:
        return True
    elif error_category in non_retryable_categories:
        return False
    else:
        return True


def should_expand_related_tables(state: dict) -> bool:
    """Determine if we should expand to include related tables."""
    retrieved_tables = state.get("retrieved_tables", [])
    user_query = state.get("user_query", "").lower()

    if len(retrieved_tables) >= 5:
        return False

    join_indicators = [
        "join",
        "relationship",
        "related",
        "connect",
        "customer order",
        "user profile",
        "product category",
        "total",
        "count",
        "sum",
        "average",
    ]

    return any(indicator in user_query for indicator in join_indicators)


def get_retry_strategy(error_category: str, attempt: int) -> dict:
    """Get retry strategy based on error category and attempt number."""
    strategies = {
        "syntax": {
            "focus": "Fix SQL syntax errors, check parentheses, quotes, and keywords",
            "temperature": 0.1,
        },
        "not_found": {
            "focus": "Verify column and table names exist in the provided schema",
            "temperature": 0.2,
        },
        "type": {
            "focus": "Fix data type mismatches and casting issues",
            "temperature": 0.1,
        },
        "unknown": {
            "focus": "Review the entire query for potential issues",
            "temperature": 0.3,
        },
    }

    base_strategy = strategies.get(error_category, strategies["unknown"])

    if attempt > 1:
        base_strategy["temperature"] = min(base_strategy["temperature"] + 0.1, 0.5)

    return base_strategy
