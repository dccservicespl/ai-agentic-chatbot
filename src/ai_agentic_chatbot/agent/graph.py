from langchain_core.messages import SystemMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from ai_agentic_chatbot.agent.router import RouterNode
from ai_agentic_chatbot.agent.state import AgentState
from ai_agentic_chatbot.infrastructure.llm import get_llm
from ai_agentic_chatbot.infrastructure.llm.types import LLMProvider, ModelType
from ai_agentic_chatbot.agent.subgraphs.sql_query.graph import sql_subgraph

fast_llm = get_llm(provider=LLMProvider.AZURE_OPENAI, model=ModelType.FAST)


def router(state: AgentState) -> dict:
    return RouterNode(state).classify()


def greeting_node(state: AgentState) -> dict:
    prompt = SystemMessage(
        "You are a helpful chat assistant. User has greeted you. Greet them warmly and ask how can you help them."
    )
    response = fast_llm.invoke([prompt, *state["messages"]])
    return {"messages": [AIMessage(content=response.content)]}


def fallback_node(state: AgentState) -> dict:
    prompt = SystemMessage(
        "You are a helpful chat assistant. User has sent a message that does not make sense. Ask them to rephrase."
    )
    response = fast_llm.invoke([prompt, *state["messages"]])
    return {"messages": [AIMessage(content=response.content)]}


def clarification_node(state: AgentState) -> dict:
    return {}


def sql_query_node(state: AgentState) -> dict:
    """
    Adapter node that invokes the SQL subgraph.
    Maps parent state to subgraph input, runs subgraph, maps output back.
    """
    from ai_agentic_chatbot.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("[Parent] Invoking SQL subgraph")

    # Map parent state to subgraph input
    subgraph_input = {
        "user_query": state["messages"][-1].content,
        "router_table_hints": state.get("relevant_tables", []),
        "generation_attempts": 0,
        "max_retries": 2,
        "is_safe": False,
        "validation_errors": [],
        "retrieved_tables": None,
        "generated_sql": None,
        "explanation": None,
        "confidence": 0.0,
        "tables_used": [],
        "query_result": None,
        "execution_error": None,
    }

    try:
        # Invoke subgraph
        subgraph_result = sql_subgraph.invoke(subgraph_input)

        # Check for errors
        if subgraph_result.get("validation_errors"):
            error_msg = "\n".join(subgraph_result["validation_errors"])
            return {
                "messages": [
                    AIMessage(content=f"I encountered an error:\n{error_msg}")
                ],
                "next_step": "end",
            }

        if subgraph_result.get("execution_error"):
            error_msg = subgraph_result["execution_error"]
            return {
                "messages": [
                    AIMessage(content=f"Query execution failed:\n{error_msg}")
                ],
                "next_step": "end",
            }

        # Format success response
        response = format_sql_response(subgraph_result)

        return {"messages": [AIMessage(content=response)], "next_step": "end"}

    except Exception as e:
        logger.error(f"SQL subgraph failed: {e}", exc_info=True)
        return {
            "messages": [
                AIMessage(content=f"I encountered an unexpected error: {str(e)}")
            ],
            "next_step": "end",
        }


def format_sql_response(subgraph_result: dict) -> str:
    """Format the final user-facing response."""
    sql = subgraph_result.get("generated_sql", "")
    explanation = subgraph_result.get("explanation", "")
    data = subgraph_result.get("query_result", [])
    tables = subgraph_result.get("tables_used", [])
    execution_time = subgraph_result.get("execution_time", 0)

    # Format SQL with basic formatting (fallback if sqlparse not available)
    try:
        import sqlparse

        formatted_sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
    except ImportError:
        # Fallback formatting
        formatted_sql = sql

    response = f"""{explanation}

**Tables Used:** {', '.join(tables)}

**SQL Query:**
```sql
{formatted_sql}
```

**Results:** {len(data)} rows"""

    if execution_time > 0:
        response += f" (executed in {execution_time:.2f}s)"

    if data:
        response += "\n\n" + format_as_markdown_table(data[:10])
        if len(data) > 10:
            response += f"\n\n_Showing 10 of {len(data)} rows_"

    return response


def format_as_markdown_table(data: list) -> str:
    """Format results as markdown table."""
    if not data:
        return ""

    headers = list(data[0].keys())
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in data:
        values = [str(row.get(h, "")) for h in headers]
        table.append("| " + " | ".join(values) + " |")

    return "\n".join(table)


_checkpointer = MemorySaver()


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router_node", router)
    workflow.add_node("greeting_node", greeting_node)
    workflow.add_node("fallback_node", fallback_node)
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("sql_query_node", sql_query_node)

    workflow.add_edge(START, "router_node")

    def routing_policy(state: AgentState) -> str:
        return state["next_step"]

    workflow.add_conditional_edges(
        "router_node",
        routing_policy,
        {
            "greeting": "greeting_node",
            "sql_query": "sql_query_node",
            "nonsense": "fallback_node",
            "ask_clarification": "clarification_node",
        },
    )

    workflow.add_edge("greeting_node", END)
    workflow.add_edge("fallback_node", END)
    workflow.add_edge("clarification_node", END)
    workflow.add_edge("sql_query_node", END)

    return workflow.compile(checkpointer=_checkpointer)
