"""Safety validation node for SQL queries."""

import re
from typing import List, Tuple
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


def validate_query_node(state: dict) -> dict:
    """
    Node 3: Validate SQL query safety.
    Single responsibility: Security and safety checks.
    """
    logger.info("[Validate Query] Running safety checks")

    sql_query = state.get("generated_sql")

    if not sql_query:
        return {"is_safe": False, "validation_errors": ["No SQL query to validate"]}

    errors = []

    # Check 1: Must be SELECT only
    if not _is_select_only(sql_query):
        errors.append("Only SELECT queries are allowed")

    # Check 2: No dangerous keywords
    dangerous = _check_dangerous_keywords(sql_query)
    if dangerous:
        errors.extend(dangerous)

    # Check 3: No SQL injection patterns
    injection = _check_injection_patterns(sql_query)
    if injection:
        errors.extend(injection)

    # Check 4: Syntax validation (basic)
    syntax = _check_basic_syntax(sql_query)
    if syntax:
        errors.extend(syntax)

    # Check 5: Resource limits — auto-appends LIMIT 100 if missing rather than failing
    sql_query, resource = _check_resource_limits(sql_query)
    if resource:
        errors.extend(resource)

    is_safe = len(errors) == 0

    if is_safe:
        logger.info("✅ Query passed all safety checks")
    else:
        logger.warning(f"❌ Query failed validation: {errors}")

    return {"is_safe": is_safe, "validation_errors": errors, "generated_sql": sql_query}


def _is_select_only(query: str) -> bool:
    """Check if query is SELECT only."""
    query_upper = query.upper().strip()

    # Must start with SELECT (or WITH for CTEs)
    if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
        return False

    # Should not have multiple statements (except for CTEs)
    statements = query_upper.split(";")
    non_empty_statements = [s.strip() for s in statements if s.strip()]

    if len(non_empty_statements) > 1:
        return False

    return True


def _check_dangerous_keywords(query: str) -> List[str]:
    """Check for dangerous SQL operations."""
    query_upper = query.upper()
    errors = []

    dangerous_keywords = [
        "DROP",
        "DELETE",
        "TRUNCATE",
        "INSERT",
        "UPDATE",
        "ALTER",
        "CREATE",
        "GRANT",
        "REVOKE",
        "MERGE",
        "EXEC",
        "EXECUTE",
        "xp_cmdshell",
        "sp_executesql",
        "BULK",
        "OPENROWSET",
        "OPENDATASOURCE",
    ]

    for keyword in dangerous_keywords:
        # Use word boundaries to avoid false positives
        if re.search(rf"\b{keyword}\b", query_upper):
            errors.append(f"Dangerous keyword detected: {keyword}")

    return errors


def _check_injection_patterns(query: str) -> List[str]:
    """Check for SQL injection patterns."""
    query_upper = query.upper()
    errors = []

    injection_patterns = [
        (r";.*DROP", "Potential injection: multiple statements with DROP"),
        (r";.*DELETE", "Potential injection: multiple statements with DELETE"),
        (r";.*INSERT", "Potential injection: multiple statements with INSERT"),
        (r";.*UPDATE", "Potential injection: multiple statements with UPDATE"),
        (r"--.*DROP", "Potential injection: comment with DROP"),
        (r"/\*.*\*/", "Block comments not allowed for security"),
        (
            r"UNION.*SELECT.*FROM.*information_schema",
            "Potential schema enumeration attack",
        ),
        (r"UNION.*SELECT.*FROM.*pg_", "Potential PostgreSQL system table access"),
        (r"'\s*OR\s*'", "Potential OR-based injection"),
        (r"'\s*AND\s*'", "Potential AND-based injection"),
        (r"1\s*=\s*1", "Potential tautology injection"),
        (r"'.*;\s*--", "Potential comment-based injection"),
        (r"WAITFOR\s+DELAY", "Time-based attack pattern"),
        (r"BENCHMARK\s*\(", "MySQL benchmark attack pattern"),
        (r"pg_sleep\s*\(", "PostgreSQL sleep attack pattern"),
    ]

    for pattern, message in injection_patterns:
        if re.search(pattern, query_upper, re.DOTALL):
            errors.append(message)

    return errors


def _check_basic_syntax(query: str) -> List[str]:
    """Basic syntax validation."""
    errors = []

    # Check balanced parentheses
    if query.count("(") != query.count(")"):
        errors.append("Unbalanced parentheses")

    # Check balanced quotes
    single_quotes = query.count("'")
    if single_quotes % 2 != 0:
        errors.append("Unbalanced single quotes")

    double_quotes = query.count('"')
    if double_quotes % 2 != 0:
        errors.append("Unbalanced double quotes")

    # Check for empty query
    if not query.strip():
        errors.append("Empty query")

    # Check for SELECT without FROM (unless it's a constant or function)
    query_upper = query.upper()
    if "SELECT" in query_upper and "FROM" not in query_upper:
        # Allow SELECT without FROM for constants, functions, or system queries
        allowed_patterns = [
            r"SELECT\s+\d+",  # SELECT 1
            r"SELECT\s+\'[^\']*\'",  # SELECT 'string'
            r"SELECT\s+NOW\(\)",  # SELECT NOW()
            r"SELECT\s+CURRENT_",  # SELECT CURRENT_TIMESTAMP, etc.
            r"SELECT\s+VERSION\(\)",  # SELECT VERSION()
        ]

        is_allowed = any(
            re.search(pattern, query_upper) for pattern in allowed_patterns
        )
        if not is_allowed:
            errors.append("SELECT requires FROM clause")

    # Check for proper semicolon usage
    if query.count(";") > 1:
        errors.append("Multiple semicolons detected - only one statement allowed")

    return errors


def _check_resource_limits(query: str):
    """Enforce resource limits. Returns (query, errors).

    Missing LIMIT is auto-fixed by appending LIMIT 100 (matching the default
    in system_prompt.md Rule 5) rather than failing the query outright.
    Pure aggregate queries (COUNT/SUM/AVG/MAX/MIN with no GROUP BY) are exempt.
    """
    errors = []
    query_upper = query.upper()

    aggregate_only_patterns = [
        r"SELECT\s+COUNT\(",
        r"SELECT\s+MAX\(",
        r"SELECT\s+MIN\(",
        r"SELECT\s+AVG\(",
        r"SELECT\s+SUM\(",
    ]
    is_aggregate_only = any(
        re.search(p, query_upper) for p in aggregate_only_patterns
    ) and "GROUP BY" not in query_upper

    if "LIMIT" not in query_upper:
        if not is_aggregate_only:
            # Strip trailing semicolon, append LIMIT, restore semicolon if needed
            stripped = query.rstrip()
            if stripped.endswith(";"):
                query = stripped[:-1].rstrip() + " LIMIT 100;"
            else:
                query = stripped + " LIMIT 100"
            logger.debug("LIMIT clause missing — auto-appended LIMIT 100")
    else:
        limit_match = re.search(r"LIMIT\s+(\d+)", query_upper)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > 10000:
                errors.append(
                    f"LIMIT {limit_value} is too high - maximum allowed is 10000"
                )

    return query, errors
