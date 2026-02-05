"""Safety validation node for SQL queries."""

import re
from typing import List
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
        return {
            "is_safe": False,
            "validation_errors": ["No SQL query to validate"]
        }
    
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
    
    # Check 5: Resource limits
    resource = _check_resource_limits(sql_query)
    if resource:
        errors.extend(resource)
    
    is_safe = len(errors) == 0
    
    if is_safe:
        logger.info("✅ Query passed all safety checks")
    else:
        logger.warning(f"❌ Query failed validation: {errors}")
    
    return {
        "is_safe": is_safe,
        "validation_errors": errors
    }


def _is_select_only(query: str) -> bool:
    """Check if query is SELECT only."""
    query_upper = query.upper().strip()
    
    # Must start with SELECT (or WITH for CTEs)
    if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
        return False
    
    # Should not have multiple statements (except for CTEs)
    statements = query_upper.split(';')
    non_empty_statements = [s.strip() for s in statements if s.strip()]
    
    if len(non_empty_statements) > 1:
        return False
    
    return True


def _check_dangerous_keywords(query: str) -> List[str]:
    """Check for dangerous SQL operations."""
    query_upper = query.upper()
    errors = []
    
    dangerous_keywords = [
        'DROP', 'DELETE', 'TRUNCATE', 'INSERT', 'UPDATE',
        'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'MERGE',
        'EXEC', 'EXECUTE', 'xp_cmdshell', 'sp_executesql',
        'BULK', 'OPENROWSET', 'OPENDATASOURCE'
    ]
    
    for keyword in dangerous_keywords:
        # Use word boundaries to avoid false positives
        if re.search(rf'\b{keyword}\b', query_upper):
            errors.append(f"Dangerous keyword detected: {keyword}")
    
    return errors


def _check_injection_patterns(query: str) -> List[str]:
    """Check for SQL injection patterns."""
    query_upper = query.upper()
    errors = []
    
    injection_patterns = [
        (r';.*DROP', "Potential injection: multiple statements with DROP"),
        (r';.*DELETE', "Potential injection: multiple statements with DELETE"),
        (r';.*INSERT', "Potential injection: multiple statements with INSERT"),
        (r';.*UPDATE', "Potential injection: multiple statements with UPDATE"),
        (r'--.*DROP', "Potential injection: comment with DROP"),
        (r'/\*.*\*/', "Block comments not allowed for security"),
        (r'UNION.*SELECT.*FROM.*information_schema', "Potential schema enumeration attack"),
        (r'UNION.*SELECT.*FROM.*pg_', "Potential PostgreSQL system table access"),
        (r"'.*OR.*'.*=.*'", "Potential OR-based injection"),
        (r"'.*AND.*'.*=.*'", "Potential AND-based injection"),
        (r'1\s*=\s*1', "Potential tautology injection"),
        (r"'.*;\s*--", "Potential comment-based injection"),
        (r'WAITFOR\s+DELAY', "Time-based attack pattern"),
        (r'BENCHMARK\s*\(', "MySQL benchmark attack pattern"),
        (r'pg_sleep\s*\(', "PostgreSQL sleep attack pattern")
    ]
    
    for pattern, message in injection_patterns:
        if re.search(pattern, query_upper, re.DOTALL):
            errors.append(message)
    
    return errors


def _check_basic_syntax(query: str) -> List[str]:
    """Basic syntax validation."""
    errors = []
    
    # Check balanced parentheses
    if query.count('(') != query.count(')'):
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
    if 'SELECT' in query_upper and 'FROM' not in query_upper:
        # Allow SELECT without FROM for constants, functions, or system queries
        allowed_patterns = [
            r'SELECT\s+\d+',  # SELECT 1
            r'SELECT\s+\'[^\']*\'',  # SELECT 'string'
            r'SELECT\s+NOW\(\)',  # SELECT NOW()
            r'SELECT\s+CURRENT_',  # SELECT CURRENT_TIMESTAMP, etc.
            r'SELECT\s+VERSION\(\)',  # SELECT VERSION()
        ]
        
        is_allowed = any(re.search(pattern, query_upper) for pattern in allowed_patterns)
        if not is_allowed:
            errors.append("SELECT requires FROM clause")
    
    # Check for proper semicolon usage
    if query.count(';') > 1:
        errors.append("Multiple semicolons detected - only one statement allowed")
    
    return errors


def _check_resource_limits(query: str) -> List[str]:
    """Check for resource-intensive operations."""
    errors = []
    query_upper = query.upper()
    
    # Check for LIMIT clause
    if 'LIMIT' not in query_upper:
        # Check if it's a simple query that might not need LIMIT
        simple_patterns = [
            r'SELECT\s+COUNT\(',  # COUNT queries
            r'SELECT\s+MAX\(',    # MAX queries
            r'SELECT\s+MIN\(',    # MIN queries
            r'SELECT\s+AVG\(',    # AVG queries
            r'SELECT\s+SUM\(',    # SUM queries
        ]
        
        is_aggregate = any(re.search(pattern, query_upper) for pattern in simple_patterns)
        
        if not is_aggregate:
            errors.append("Query should include LIMIT clause to prevent excessive results")
    else:
        # Check if LIMIT is reasonable
        limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > 10000:
                errors.append(f"LIMIT {limit_value} is too high - maximum allowed is 10000")
    
    # Check for potentially expensive operations
    expensive_patterns = [
        (r'SELECT\s+\*\s+FROM.*JOIN.*JOIN.*JOIN', "Multiple JOINs without specific columns may be expensive"),
        (r'ORDER\s+BY.*,.*,.*,', "Complex ORDER BY with many columns may be expensive"),
        (r'GROUP\s+BY.*,.*,.*,.*,', "Complex GROUP BY with many columns may be expensive"),
        (r'LIKE\s+\'%.*%\'', "Leading wildcard LIKE patterns are expensive"),
        (r'NOT\s+IN\s+\(SELECT', "NOT IN with subquery can be expensive - consider NOT EXISTS"),
    ]
    
    for pattern, message in expensive_patterns:
        if re.search(pattern, query_upper):
            errors.append(f"Performance warning: {message}")
    
    return errors
