"""Query execution node for database operations."""

import re

from sqlalchemy import text
from ai_agentic_chatbot.infrastructure.datasource.factory import get_engine
from ai_agentic_chatbot.infrastructure.context.context_settings import get_context_registry
from ai_agentic_chatbot.logging_config import get_logger
import time

logger = get_logger(__name__)

# Configuration constants
QUERY_TIMEOUT = 30  # seconds
MAX_QUERY_RESULTS = 1000

# SET cannot use bound params, so schema_name is interpolated directly into
# the SQL string — it must be validated first even though it traces back to
# a trusted config.yaml value, not raw user input.
_VALID_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")

# TODO 18 (not yet done): /stream will inject the real db_context_id. Until
# then, every request resolves against this one real, configured context —
# "default" is not a registered context_id in config.yaml.
_FALLBACK_CONTEXT_ID = "sales"


def execute_query_node(state: dict) -> dict:
    """
    Node 4: Execute validated SQL query.
    Single responsibility: Database execution.
    """
    logger.info("[Execute Query] Running SQL")

    # Check if query is safe
    if not state.get("is_safe", False):
        logger.warning("Skipping execution - query not validated")
        return {"execution_error": "Query failed safety validation"}

    sql_query = state.get("generated_sql")

    if not sql_query:
        return {"execution_error": "No SQL query to execute"}

    try:
        db_context_id = state.get("db_context_id") or _FALLBACK_CONTEXT_ID
        ctx = get_context_registry().get_context(db_context_id)

        if not _VALID_SCHEMA_NAME.match(ctx.schema_name):
            raise ValueError(f"Invalid schema name: {ctx.schema_name!r}")

        engine = get_engine("postgresql.primary")

        start_time = time.time()

        # Explicit transaction (conn.begin()), NOT autocommit — SET LOCAL is
        # transaction-scoped. With autocommit, SQLAlchemy would commit the
        # implicit transaction from the SET statement before running the SQL
        # query, silently clearing search_path and querying "public" instead
        # of this context's schema with no error raised.
        with engine.connect() as conn:
            with conn.begin():
                # "public" fallback covers auth tables (users, refresh_tokens,
                # prompt_logs), which stay in the public schema permanently.
                conn.execute(
                    text(f"SET LOCAL search_path TO {ctx.schema_name}, app, public")
                )
                result = conn.execute(text(sql_query))

                rows = result.fetchmany(MAX_QUERY_RESULTS)

                has_more = len(rows) == MAX_QUERY_RESULTS
                if has_more:
                    extra_row = result.fetchone()
                    if extra_row:
                        logger.warning(
                            f"Query returned more than {MAX_QUERY_RESULTS} rows - truncated"
                        )

                execution_time = time.time() - start_time

                if rows and result.keys():
                    data = [
                        {
                            key: _serialize_value(value)
                            for key, value in zip(result.keys(), row)
                        }
                        for row in rows
                    ]
                else:
                    data = []
            # transaction commits here; SET LOCAL resets automatically — no pool leak

            logger.info(
                f"✅ Query executed successfully: {len(data)} rows in {execution_time:.2f}s"
            )

            return {
                "query_result": data,
                "execution_error": None,
                "execution_time": execution_time,
                "row_count": len(data),
                "has_more_results": has_more,
            }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Query execution failed: {error_msg}")
        error_category = _categorize_error(error_msg)
        return {
            "query_result": None,
            "execution_error": error_msg,
            "error_category": error_category,
        }


def _serialize_value(value):
    """Serialize database values to JSON-compatible types."""
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "__float__"):
        try:
            return float(value)
        except (ValueError, TypeError):
            pass

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary data: {len(value)} bytes>"

    if hasattr(value, "__dict__") or isinstance(value, (list, dict, tuple)):
        return str(value)

    return value


def _categorize_error(error_msg: str) -> str:
    """Categorize database errors for better retry logic."""
    error_lower = error_msg.lower()

    # Syntax errors - can be fixed by regeneration
    syntax_indicators = [
        "syntax error",
        "invalid syntax",
        "unexpected token",
        "missing",
        "expected",
        "parse error",
    ]

    if any(indicator in error_lower for indicator in syntax_indicators):
        return "syntax"

    # Column/table not found - can be fixed by regeneration
    not_found_indicators = [
        "column",
        "table",
        "relation",
        "does not exist",
        "not found",
        "unknown column",
        "unknown table",
    ]

    if any(indicator in error_lower for indicator in not_found_indicators):
        return "not_found"

    # Permission errors - cannot be fixed by retry
    permission_indicators = [
        "permission denied",
        "access denied",
        "insufficient privileges",
        "not authorized",
    ]

    if any(indicator in error_lower for indicator in permission_indicators):
        return "permission"

    # Connection errors - might be temporary
    connection_indicators = ["connection", "timeout", "network", "host", "unreachable"]

    if any(indicator in error_lower for indicator in connection_indicators):
        return "connection"

    # Data type errors - can be fixed by regeneration
    type_indicators = ["type", "cast", "convert", "invalid input", "data type"]

    if any(indicator in error_lower for indicator in type_indicators):
        return "type"

    # Default category
    return "unknown"
