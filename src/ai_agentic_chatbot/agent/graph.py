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
from ai_agentic_chatbot.agent.nodes.visualizer import visualizer_node

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

        # Add visualization analysis to the state
        state.update(subgraph_result)

        # Generate visualization configuration
        viz_result = visualizer_node(state)

        # Create structured response data
        visualization = viz_result.get("visualization", {})
        content = _generate_brief_content(visualization)

        return {
            "messages": [AIMessage(content=content)],
            "visualization": visualization,
            "next_step": "end",
        }

    except Exception as e:
        logger.error(f"SQL subgraph failed: {e}", exc_info=True)
        return {
            "messages": [
                AIMessage(content=f"I encountered an unexpected error: {str(e)}")
            ],
            "next_step": "end",
        }


def create_clean_json_response(subgraph_result: dict, viz_result: dict) -> str:
    """Create a clean JSON response for frontend consumption."""
    import json

    visualization = viz_result.get("visualization", {})

    # Create clean response structure
    response_data = {
        "content": _generate_brief_content(visualization),
        "visualization": visualization,
    }

    return json.dumps(response_data, indent=2)


def _generate_brief_content(visualization: dict) -> str:
    """Generate brief, contextual content based on visualization type."""
    viz_type = visualization.get("type", "")
    title = visualization.get("title", "")

    if viz_type == "kpi":
        return f"Here's the {title.lower()}:"
    elif viz_type == "bar_chart":
        return f"Here's the {title.lower()} comparison:"
    elif viz_type == "line_chart":
        return f"Here's the {title.lower()} trend:"
    elif viz_type == "pie_chart":
        return f"Here's the {title.lower()} distribution:"
    elif viz_type == "table":
        return f"Here are the query results:"
    else:
        return "Here's your data:"


def format_sql_response_with_visualization(
    subgraph_result: dict, viz_result: dict
) -> str:
    """Format the SQL query results with visualization data for frontend consumption."""
    import json

    explanation = subgraph_result.get("explanation", "")
    sql = subgraph_result.get("generated_sql", "")
    tables = subgraph_result.get("tables_used", [])
    data = subgraph_result.get("query_result", [])
    execution_time = subgraph_result.get("execution_time", 0)
    visualization = viz_result.get("visualization", {})

    # Format SQL with basic formatting (fallback if sqlparse not available)
    try:
        import sqlparse

        formatted_sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
    except ImportError:
        formatted_sql = sql

    # Create human-readable response
    response = f"""{explanation}

**Tables Used:** {', '.join(tables)}

**SQL Query:**
```sql
{formatted_sql}
```

**Results:** {len(data)} rows"""

    if execution_time > 0:
        response += f" (executed in {execution_time:.2f}s)"

    # Add visualization insights
    if visualization.get("summary"):
        response += f"\n\n**Key Insights:** {visualization['summary']}"

    # Add visualization type recommendation
    if visualization.get("type"):
        viz_type = visualization["type"].replace("_", " ").title()
        response += f"\n\n**Recommended Visualization:** {viz_type}"

    # Add formatted table for readability (limited rows)
    if data and visualization.get("type") != "kpi":
        response += "\n\n" + format_as_markdown_table(data[:10])
        if len(data) > 10:
            response += f"\n\n_Showing 10 of {len(data)} rows_"
    elif visualization.get("type") == "kpi" and visualization.get("config", {}).get(
        "value"
    ):
        # For KPI, show the formatted value prominently
        kpi_value = visualization["config"]["value"]
        response += f"\n\n**{visualization.get('title', 'Result')}:** {kpi_value}"

    # Embed structured data for frontend (as JSON comment)
    frontend_data = {
        "type": "sql_result",
        "explanation": explanation,
        "tables_used": tables,
        "sql_query": formatted_sql,
        "execution_time": execution_time,
        "row_count": len(data),
        "visualization": visualization,
    }

    response += f"\n\n<!-- FRONTEND_DATA: {json.dumps(frontend_data)} -->"

    return response


def format_sql_response(subgraph_result: dict) -> str:
    """Format the SQL query results for user display with enhanced frontend support."""
    import json

    explanation = subgraph_result.get("explanation", "")
    sql = subgraph_result.get("generated_sql", "")
    tables = subgraph_result.get("tables_used", [])
    data = subgraph_result.get("query_result", [])
    execution_time = subgraph_result.get("execution_time", 0)

    # Format SQL with basic formatting (fallback if sqlparse not available)
    try:
        import sqlparse

        formatted_sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
    except ImportError:
        formatted_sql = sql

    # Analyze data for visualization suggestions and formatting
    viz_suggestions = _analyze_for_visualizations(data, sql)
    formatted_data = _format_data_values(data)

    # Create structured response for frontend
    structured_response = {
        "type": "sql_result",
        "explanation": explanation,
        "tables_used": tables,
        "sql_query": formatted_sql,
        "execution_time": execution_time,
        "row_count": len(data),
        "data": formatted_data[:100],  # Limit to 100 rows for performance
        "has_more_data": len(data) > 100,
        "visualization_suggestions": viz_suggestions,
        "summary": _generate_data_summary(formatted_data, sql),
    }

    # Create human-readable response with embedded JSON for frontend
    response = f"""{explanation}

**Tables Used:** {', '.join(tables)}

**SQL Query:**
```sql
{formatted_sql}
```

**Results:** {len(data)} rows"""

    if execution_time > 0:
        response += f" (executed in {execution_time:.2f}s)"

    # Add summary insights
    if structured_response["summary"]:
        response += f"\n\n**Key Insights:**\n{structured_response['summary']}"

    # Add visualization suggestions
    if viz_suggestions:
        response += f"\n\n**Recommended Visualizations:** {', '.join(viz_suggestions)}"

    # Add table data (limited for readability)
    if data:
        response += "\n\n" + format_as_markdown_table(formatted_data[:10])
        if len(data) > 10:
            response += f"\n\n_Showing 10 of {len(data)} rows_"

    # Embed structured data for frontend (hidden in HTML comment)
    response += f"\n\n<!-- FRONTEND_DATA: {json.dumps(structured_response)} -->"

    return response


def _analyze_for_visualizations(data: list, sql: str) -> list:
    """Analyze data and query to suggest appropriate visualizations."""
    if not data:
        return []

    suggestions = []
    sql_lower = sql.lower()

    # Check if it's aggregated data (good for charts)
    if any(
        keyword in sql_lower
        for keyword in ["sum(", "count(", "avg(", "max(", "min(", "group by"]
    ):
        suggestions.append("Bar Chart")
        suggestions.append("Pie Chart")

    # Check for time series data
    if any(
        keyword in sql_lower for keyword in ["date", "time", "year", "month", "day"]
    ):
        suggestions.append("Line Chart")
        suggestions.append("Time Series")

    # Check for numerical data
    headers = list(data[0].keys()) if data else []
    numeric_cols = []
    for header in headers:
        if data and isinstance(data[0].get(header), (int, float)):
            numeric_cols.append(header)

    if len(numeric_cols) >= 2:
        suggestions.append("Scatter Plot")

    # Always suggest table for detailed view
    suggestions.append("Data Table")

    return list(set(suggestions))  # Remove duplicates


def _format_data_values(data: list) -> list:
    """Format data values for better display (currency, percentages, etc.)."""
    if not data:
        return data

    formatted_data = []
    for row in data:
        formatted_row = {}
        for key, value in row.items():
            if isinstance(value, float):
                # Check if it looks like currency (large numbers)
                if value > 1000 and any(
                    keyword in key.lower()
                    for keyword in [
                        "sales",
                        "revenue",
                        "amount",
                        "price",
                        "cost",
                        "total",
                    ]
                ):
                    formatted_row[key] = f"${value:,.2f}"
                # Check if it looks like a percentage
                elif 0 <= value <= 1 and any(
                    keyword in key.lower() for keyword in ["rate", "percent", "ratio"]
                ):
                    formatted_row[key] = f"{value:.2%}"
                else:
                    formatted_row[key] = round(value, 2)
            else:
                formatted_row[key] = value
        formatted_data.append(formatted_row)

    return formatted_data


def _generate_data_summary(data: list, sql: str) -> str:
    """Generate a summary of the data insights."""
    if not data:
        return "No data returned from query."

    summary_parts = []

    # Basic stats
    row_count = len(data)
    if row_count == 1:
        summary_parts.append("Single result returned")
    else:
        summary_parts.append(f"{row_count} records found")

    # Analyze first row for insights
    if data:
        first_row = data[0]
        for key, value in first_row.items():
            if isinstance(value, (int, float)) and value != 0:
                if "total" in key.lower():
                    if isinstance(value, str) and value.startswith("$"):
                        summary_parts.append(f"Total value: {value}")
                    elif isinstance(value, (int, float)):
                        summary_parts.append(f"Total {key.lower()}: {value:,.2f}")

    return " • ".join(summary_parts) if summary_parts else ""


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
