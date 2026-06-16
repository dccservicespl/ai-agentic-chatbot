"""End-to-end regression test for issue #19's acceptance criteria.

Same thread_id, four turns through the REAL compiled graph (router branching,
MemorySaver checkpointing, fallback/greeting nodes, visualizer) with only the
network-touching boundaries mocked (router LLM, schema loader, greeting LLM,
SQL subgraph):

  1. SQL query  -> produces a real pie_chart visualization
  2. "how are you" (out_of_scope) -> must NOT echo turn 1's chart, no LLM call
  3. "hi" (real greeting) -> LLM-generated reply, still no chart
  4. gibberish (nonsense) -> static reply, no LLM call, no chart
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ai_agentic_chatbot.agent.router import RouterDecision


def make_decision(**overrides) -> RouterDecision:
    base = dict(
        intent="sql_query",
        reasoning="test",
        is_answerable=True,
        missing_data_reason=None,
        clarification=None,
        relevant_tables=["orders"],
    )
    base.update(overrides)
    return RouterDecision(**base)


SQL_SUBGRAPH_RESULT = {
    "validation_errors": [],
    "execution_error": None,
    "explanation": "Distribution of delivery method.",
    "generated_sql": "SELECT delivery_method, ... FROM orders GROUP BY delivery_method",
    "tables_used": ["orders"],
    "query_result": [
        {"delivery_method": "DELIVERY", "percentage": 94.2},
        {"delivery_method": "PICKUP", "percentage": 5.8},
    ],
}


@pytest.fixture
def router_prompt_path(tmp_path, monkeypatch):
    prompt_file = tmp_path / "router_prompt.md"
    prompt_file.write_text("Schema:\n{schema_text}\n")
    monkeypatch.setenv("ROUTER_PROMPT_PATH", str(prompt_file))


@pytest.fixture
def graph_mocks(router_prompt_path):
    """Build the real compiled graph with only network boundaries mocked."""
    with patch("ai_agentic_chatbot.agent.router.get_llm") as mock_get_llm, patch(
        "ai_agentic_chatbot.agent.router.SchemaLoader"
    ) as mock_schema_loader_cls, patch(
        "ai_agentic_chatbot.agent.router.get_system_prompt", return_value="SYSTEM"
    ), patch(
        "ai_agentic_chatbot.agent.graph.fast_llm"
    ) as mock_fast_llm, patch(
        "ai_agentic_chatbot.agent.graph.sql_subgraph"
    ) as mock_sql_subgraph:
        mock_structured_llm = MagicMock()
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_llm.return_value = mock_llm

        mock_schema_loader_cls.return_value.load_schema_summary.return_value = {
            "tables": [{"table": "orders", "bussiness_purpose": "Customer orders"}]
        }

        mock_fast_llm.invoke.return_value = AIMessage(
            content="Hello! How can I help you today?"
        )
        mock_sql_subgraph.invoke.return_value = SQL_SUBGRAPH_RESULT

        from ai_agentic_chatbot.agent.graph import build_graph

        graph = build_graph()
        yield graph, mock_structured_llm, mock_fast_llm, mock_sql_subgraph


def test_chart_then_chitchat_then_greeting_then_gibberish(graph_mocks):
    graph, mock_structured_llm, mock_fast_llm, mock_sql_subgraph = graph_mocks
    config = {"configurable": {"thread_id": f"e2e-{uuid.uuid4()}"}}

    # Turn 1 — SQL query that produces a pie chart.
    mock_structured_llm.invoke.return_value = make_decision(
        intent="sql_query", relevant_tables=["orders"]
    )
    state1 = graph.invoke(
        {"messages": [HumanMessage(content="delivery method split")]}, config=config
    )
    assert state1["next_step"] == "end"
    assert state1["visualization"]["type"] == "pie_chart"
    mock_sql_subgraph.invoke.assert_called_once()
    mock_fast_llm.invoke.assert_not_called()

    # Turn 2 — "how are you" (out_of_scope): must NOT echo turn 1's chart,
    # and must not call the LLM for the reply (router already wrote it).
    mock_structured_llm.invoke.return_value = make_decision(
        intent="out_of_scope",
        is_answerable=False,
        missing_data_reason="Casual chit-chat, no data requested.",
    )
    state2 = graph.invoke(
        {"messages": [HumanMessage(content="how are you")]}, config=config
    )
    assert state2["visualization"] is None
    assert state2["next_step"] == "nonsense"
    assert "I can help you with" in state2["messages"][-1].content
    mock_fast_llm.invoke.assert_not_called()

    # Turn 3 — "hi" (real greeting): gets an LLM-generated reply, no chart.
    mock_structured_llm.invoke.return_value = make_decision(intent="greeting")
    state3 = graph.invoke({"messages": [HumanMessage(content="hi")]}, config=config)
    assert state3["visualization"] is None
    assert state3["messages"][-1].content == "Hello! How can I help you today?"
    mock_fast_llm.invoke.assert_called_once()

    # Turn 4 — gibberish (nonsense): static reply, no LLM call, no chart.
    mock_fast_llm.invoke.reset_mock()
    mock_structured_llm.invoke.return_value = make_decision(
        intent="nonsense",
        is_answerable=False,
        missing_data_reason="Input is gibberish.",
    )
    state4 = graph.invoke(
        {"messages": [HumanMessage(content="asdkjasdj")]}, config=config
    )
    assert state4["visualization"] is None
    mock_fast_llm.invoke.assert_not_called()
