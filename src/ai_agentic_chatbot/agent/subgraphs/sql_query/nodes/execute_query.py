"""Query execution node for database operations."""

from sqlalchemy import create_engine, text
from ai_agentic_chatbot.infrastructure.datasource.factory import get_engine
from ai_agentic_chatbot.logging_config import get_logger
import time

logger = get_logger(__name__)

# Configuration constants
QUERY_TIMEOUT = 30  # seconds
MAX_QUERY_RESULTS = 1000


def execute_query_node(state: dict) -> dict:
    """
    Node 4: Execute validated SQL query.
    Single responsibility: Database execution.
    """
    logger.info("[Execute Query] Running SQL")
    
    # Check if query is safe
    if not state.get("is_safe", False):
        logger.warning("Skipping execution - query not validated")
        return {
            "execution_error": "Query failed safety validation"
        }
    
    sql_query = state.get("generated_sql")
    
    if not sql_query:
        return {
            "execution_error": "No SQL query to execute"
        }
    
    try:
        # Get database engine
        engine = get_engine("mysql.primary")  # Use primary datasource
        
        start_time = time.time()
        
        with engine.connect() as conn:
            # Execute with timeout and result limit
            result = conn.execute(
                text(sql_query).execution_options(
                    compiled_cache={},
                    autocommit=True
                )
            )
            
            # Fetch results with limit
            rows = result.fetchmany(MAX_QUERY_RESULTS)
            
            # Check if there are more results
            has_more = len(rows) == MAX_QUERY_RESULTS
            if has_more:
                # Try to fetch one more to confirm
                extra_row = result.fetchone()
                if extra_row:
                    logger.warning(f"Query returned more than {MAX_QUERY_RESULTS} rows - truncated")
            
            execution_time = time.time() - start_time
            
            # Convert to list of dicts
            if rows and result.keys():
                data = [
                    {key: _serialize_value(value) for key, value in zip(result.keys(), row)} 
                    for row in rows
                ]
            else:
                data = []
            
            logger.info(f"✅ Query executed successfully: {len(data)} rows in {execution_time:.2f}s")
            
            return {
                "query_result": data,
                "execution_error": None,
                "execution_time": execution_time,
                "row_count": len(data),
                "has_more_results": has_more
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Query execution failed: {error_msg}")
        
        # Categorize error for better retry logic
        error_category = _categorize_error(error_msg)
        
        return {
            "query_result": None,
            "execution_error": error_msg,
            "error_category": error_category
        }


def _serialize_value(value):
    """Serialize database values to JSON-compatible types."""
    if value is None:
        return None
    
    # Handle datetime objects
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    
    # Handle decimal objects
    if hasattr(value, '__float__'):
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
    
    # Handle bytes
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return f"<binary data: {len(value)} bytes>"
    
    # Convert to string for complex types
    if hasattr(value, '__dict__') or isinstance(value, (list, dict, tuple)):
        return str(value)
    
    return value


def _categorize_error(error_msg: str) -> str:
    """Categorize database errors for better retry logic."""
    error_lower = error_msg.lower()
    
    # Syntax errors - can be fixed by regeneration
    syntax_indicators = [
        'syntax error',
        'invalid syntax',
        'unexpected token',
        'missing',
        'expected',
        'parse error'
    ]
    
    if any(indicator in error_lower for indicator in syntax_indicators):
        return "syntax"
    
    # Column/table not found - can be fixed by regeneration
    not_found_indicators = [
        'column',
        'table',
        'relation',
        'does not exist',
        'not found',
        'unknown column',
        'unknown table'
    ]
    
    if any(indicator in error_lower for indicator in not_found_indicators):
        return "not_found"
    
    # Permission errors - cannot be fixed by retry
    permission_indicators = [
        'permission denied',
        'access denied',
        'insufficient privileges',
        'not authorized'
    ]
    
    if any(indicator in error_lower for indicator in permission_indicators):
        return "permission"
    
    # Connection errors - might be temporary
    connection_indicators = [
        'connection',
        'timeout',
        'network',
        'host',
        'unreachable'
    ]
    
    if any(indicator in error_lower for indicator in connection_indicators):
        return "connection"
    
    # Data type errors - can be fixed by regeneration
    type_indicators = [
        'type',
        'cast',
        'convert',
        'invalid input',
        'data type'
    ]
    
    if any(indicator in error_lower for indicator in type_indicators):
        return "type"
    
    # Default category
    return "unknown"
