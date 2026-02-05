"""SQL generation node with structured LLM output."""

from ai_agentic_chatbot.infrastructure.llm.factory import get_llm
from ai_agentic_chatbot.infrastructure.llm.types import LLMProvider, ModelType
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import List, Optional
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


class SQLGeneration(BaseModel):
    """Structured LLM output for SQL generation."""
    query: str = Field(description="Generated SQL query")
    explanation: str = Field(description="Plain English explanation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    tables_used: List[str] = Field(description="Tables referenced in query")
    warnings: Optional[List[str]] = None


def generate_sql_node(state: dict) -> dict:
    """
    Node 2: Generate SQL query from retrieved schemas.
    Single responsibility: LLM-based SQL generation.
    """
    logger.info("[Generate SQL] Creating query")
    
    retrieved_tables = state.get("retrieved_tables", [])
    
    # Check if we have schemas
    if not retrieved_tables:
        return {
            "validation_errors": ["Cannot generate SQL without table schemas"]
        }
    
    # Format schemas for prompt
    schema_text = "\n\n".join([
        f"-- Table: {name} (Relevance: {score:.2f})\n{ddl}"
        for name, ddl, score in retrieved_tables
    ])
    
    user_query = state["user_query"]
    
    # Check if this is a retry attempt
    previous_error = state.get("execution_error")
    generation_attempts = state.get("generation_attempts", 0)
    
    try:
        # Get LLM with structured output
        llm = get_llm(LLMProvider.AZURE_OPENAI, ModelType.SMART)
        structured_llm = llm.with_structured_output(SQLGeneration, strict=True)
        
        # Create generation prompt
        prompt_content = _create_generation_prompt(
            schema_text=schema_text,
            user_query=user_query,
            previous_error=previous_error,
            generation_attempts=generation_attempts
        )
        
        prompt = SystemMessage(content=prompt_content)
        result: SQLGeneration = structured_llm.invoke([prompt])
        
        logger.info(f"Generated SQL: {result.query[:100]}...")
        logger.info(f"Confidence: {result.confidence}")
        
        return {
            "generated_sql": result.query,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "tables_used": result.tables_used,
            "generation_attempts": generation_attempts + 1
        }
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}", exc_info=True)
        return {
            "validation_errors": [f"Generation error: {str(e)}"],
            "generation_attempts": generation_attempts + 1
        }


def _create_generation_prompt(
    schema_text: str,
    user_query: str,
    previous_error: Optional[str] = None,
    generation_attempts: int = 0
) -> str:
    """Create the SQL generation prompt."""
    
    base_prompt = f"""You are an expert SQL query generator. Generate a SQL query based on the user's request and the provided database schema.

DATABASE SCHEMA:
{schema_text}

USER REQUEST:
{user_query}

REQUIREMENTS:
1. Generate ONLY SELECT queries - no INSERT, UPDATE, DELETE, DROP, etc.
2. Use proper JOIN syntax when combining tables
3. Include appropriate WHERE clauses for filtering
4. Use GROUP BY and aggregate functions when needed
5. Limit results to a reasonable number (max 1000 rows)
6. Use PostgreSQL syntax and functions
7. Be precise with column names and table references
8. Handle NULL values appropriately

RESPONSE FORMAT:
- query: The complete SQL query
- explanation: Clear explanation of what the query does
- confidence: Your confidence level (0.0 to 1.0)
- tables_used: List of table names used in the query
- warnings: Any potential issues or limitations (optional)

EXAMPLE RESPONSE:
{{
    "query": "SELECT customer_name, SUM(order_total) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.order_date >= '2024-01-01' GROUP BY customer_name ORDER BY total_spent DESC LIMIT 100;",
    "explanation": "This query finds the total amount spent by each customer since January 1st, 2024, ordered by highest spenders first.",
    "confidence": 0.95,
    "tables_used": ["customers", "orders"],
    "warnings": ["Results limited to top 100 customers"]
}}"""

    # Add error feedback if retrying
    if previous_error and generation_attempts > 0:
        base_prompt += f"""

PREVIOUS ATTEMPT FAILED WITH ERROR:
{previous_error}

Generate a CORRECTED query that fixes this error.
Common fixes:
- Check column names and spelling (use exact names from schema)
- Verify JOIN conditions match foreign key relationships
- Ensure proper GROUP BY clauses (include all non-aggregate columns)
- Handle data type conversions properly
- Check table aliases and references
- Verify aggregate function usage
- Ensure WHERE clause syntax is correct

This is attempt #{generation_attempts + 1}. Be extra careful with:
1. Column name accuracy
2. Table relationship correctness
3. SQL syntax validation
"""

    return base_prompt
