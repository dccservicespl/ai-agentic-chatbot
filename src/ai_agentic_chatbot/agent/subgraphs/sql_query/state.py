"""State definition for the SQL query subgraph."""

from typing import TypedDict, Optional, List, Tuple
from typing_extensions import Annotated
from operator import add


class SQLSubgraphState(TypedDict):
    """State for the SQL query subgraph following single responsibility principle."""
    
    # Input (from parent graph)
    user_query: str
    router_table_hints: Optional[List[str]]
    
    # Schema Retrieval
    retrieved_tables: Optional[List[Tuple[str, str, float]]]  # (name, ddl, score)
    
    # Generation
    generated_sql: Optional[str]
    explanation: Optional[str]
    confidence: float
    tables_used: List[str]
    
    # Validation
    is_safe: bool
    validation_errors: Annotated[List[str], add]
    
    # Execution
    query_result: Optional[List[dict]]
    execution_error: Optional[str]
    
    # Retry tracking
    generation_attempts: int
    max_retries: int
