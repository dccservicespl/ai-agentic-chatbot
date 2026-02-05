"""SQL subgraph builder with proper node connections and retry logic."""

from langgraph.graph import StateGraph, END
from ai_agentic_chatbot.agent.subgraphs.sql_query.state import SQLSubgraphState
from ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.retrieve_schemas import (
    retrieve_schemas_node,
)
from ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.generate_sql import (
    generate_sql_node,
)
from ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.validate_query import (
    validate_query_node,
)
from ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.execute_query import (
    execute_query_node,
)
from ai_agentic_chatbot.agent.subgraphs.sql_query.routes import (
    route_after_retrieval,
    route_after_generation,
    route_after_validation,
    route_after_execution,
)
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


def create_sql_subgraph():
    """
    Build the SQL query processing subgraph.

    Flow:
    1. retrieve_schemas: Semantic search for relevant tables
    2. generate_sql: LLM generates SQL query
    3. validate_query: Safety and syntax validation
    4. execute_query: Database execution
    5. (Optional) Retry loop back to generate_sql on error

    This follows the "one action per node" principle for:
    - Proper streaming support
    - Individual node testing
    - Clear error isolation
    - Retry mechanism
    """
    logger.info("Building SQL subgraph with multi-node architecture")

    workflow = StateGraph(SQLSubgraphState)

    workflow.add_node("retrieve_schemas", retrieve_schemas_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_query", validate_query_node)
    workflow.add_node("execute_query", execute_query_node)

    workflow.set_entry_point("retrieve_schemas")

    workflow.add_conditional_edges(
        "retrieve_schemas",
        route_after_retrieval,
        {"generate_sql": "generate_sql", "END": END},
    )

    workflow.add_conditional_edges(
        "generate_sql",
        route_after_generation,
        {"validate_query": "validate_query", "END": END},
    )

    workflow.add_conditional_edges(
        "validate_query",
        route_after_validation,
        {"execute_query": "execute_query", "END": END},
    )

    workflow.add_conditional_edges(
        "execute_query",
        route_after_execution,
        {"generate_sql": "generate_sql", "END": END},
    )

    logger.info("SQL subgraph compiled successfully")
    return workflow.compile()


sql_subgraph = create_sql_subgraph()
