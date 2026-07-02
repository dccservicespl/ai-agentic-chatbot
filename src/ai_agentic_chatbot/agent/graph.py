import json
from pathlib import Path

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
from ai_agentic_chatbot.utils.prompt_loader import load_file_content

fast_llm = get_llm(provider=LLMProvider.AZURE_OPENAI, model=ModelType.FAST)

_ANALYSIS_PROMPT_PATH = (
        Path(__file__).resolve().parent.parent / "prompts" / "analysis_prompt.md"
)


def router(state: AgentState) -> dict:
    return RouterNode(state).classify()


def greeting_node(state: AgentState) -> dict:
    prompt = SystemMessage(
        "You are a helpful chat assistant. User has greeted you. Greet them warmly and ask how can you help them."
    )
    response = fast_llm.invoke([prompt, *state["messages"]])
    return {"messages": [AIMessage(content=response.content)]}


def fallback_node(state: AgentState) -> dict:
    # router_node already attaches the full user-facing message for the
    # "nonsense" next_step (covers both gibberish/nonsense and out_of_scope
    # chit-chat) — calling the LLM here would just stack a second, generic
    # reply on top of it.
    return {}


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
        # TODO 18: /stream will inject db_context_id into AgentState for every
        # request; until then this falls back to "sales" — the one real
        # registered context_id in config.yaml ("default" is not registered).
        "db_context_id": state.get("db_context_id", "sales"),
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
                "visualization": None,
                "analysis": None,
            }

        if subgraph_result.get("execution_error"):
            error_msg = subgraph_result["execution_error"]
            return {
                "messages": [
                    AIMessage(content=f"Query execution failed:\n{error_msg}")
                ],
                "next_step": "end",
                "visualization": None,
                "analysis": None,
            }

        # Add visualization analysis to the state
        state.update(subgraph_result)

        # Generate visualization configuration
        viz_result = visualizer_node(state)

        # Create structured response data
        visualization = viz_result.get("visualization", {})
        content = _generate_brief_content(visualization)

        # Generate LLM analysis of the result
        analysis = None
        try:
            prompt_template = load_file_content(_ANALYSIS_PROMPT_PATH)
            analysis_prompt = prompt_template.format(
                user_query=state["messages"][-1].content,
                viz_type=visualization.get("type", ""),
                viz_title=visualization.get("title", ""),
                row_count=visualization.get("row_count", 0),
                first_3_rows_as_json=json.dumps(
                    visualization.get("data", [])[:3], indent=2, default=str
                ),
            )
            response = fast_llm.invoke([SystemMessage(content=analysis_prompt)])
            analysis = response.content
        except Exception as exc:
            logger.warning(f"Analysis generation failed, falling back to explanation: {exc}")
            analysis = subgraph_result.get("explanation", "")

        return {
            "messages": [AIMessage(content=content)],
            "visualization": visualization,
            "analysis": analysis,
            "next_step": "end",
        }

    except Exception as e:
        logger.error(f"SQL subgraph failed: {e}", exc_info=True)
        return {
            "messages": [
                AIMessage(content=f"I encountered an unexpected error: {str(e)}")
            ],
            "next_step": "end",
            "visualization": None,
            "analysis": None,
        }


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
