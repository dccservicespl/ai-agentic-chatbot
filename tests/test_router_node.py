"""Regression tests for issue #19 — visualization/relevant_tables state leaking
across conversation turns because RouterNode.classify() didn't reset them.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from ai_agentic_chatbot.agent.router import (
    ClarificationDecision,
    RouterDecision,
    RouterNode,
)


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


@pytest.fixture
def router_prompt_path(tmp_path, monkeypatch):
    prompt_file = tmp_path / "router_prompt.md"
    prompt_file.write_text("Schema:\n{schema_text}\n")
    monkeypatch.setenv("ROUTER_PROMPT_PATH", str(prompt_file))


def classify_with(decision: RouterDecision, router_prompt_path) -> dict:
    """Run RouterNode.classify() with the LLM, schema loader, and system
    prompt mocked out so the test runs offline and deterministically."""
    state = {"messages": [HumanMessage(content="hello")]}

    with patch("ai_agentic_chatbot.agent.router.get_llm") as mock_get_llm, patch(
        "ai_agentic_chatbot.agent.router.SchemaLoader"
    ) as mock_schema_loader_cls, patch(
        "ai_agentic_chatbot.agent.router.get_system_prompt", return_value="SYSTEM"
    ):
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = decision
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_llm.return_value = mock_llm

        mock_schema_loader_cls.return_value.load_schema_summary.return_value = {
            "tables": [{"table": "orders", "bussiness_purpose": "Customer orders"}]
        }

        node = RouterNode(state)
        return node.classify()


def test_greeting_resets_visualization_and_tables(router_prompt_path):
    result = classify_with(make_decision(intent="greeting"), router_prompt_path)

    assert result["next_step"] == "greeting"
    assert result["visualization"] is None
    assert result["relevant_tables"] is None


def test_not_answerable_resets_visualization_and_tables(router_prompt_path):
    decision = make_decision(
        intent="nonsense",
        is_answerable=False,
        missing_data_reason="We do not have employee salary data.",
    )
    result = classify_with(decision, router_prompt_path)

    assert result["next_step"] == "nonsense"
    assert result["visualization"] is None
    assert result["relevant_tables"] is None


def test_ambiguous_sql_query_resets_visualization_and_tables(router_prompt_path):
    decision = make_decision(
        intent="sql_query",
        is_answerable=True,
        clarification=ClarificationDecision(
            is_ambiguous=True, clarification_question="Which time period?"
        ),
    )
    result = classify_with(decision, router_prompt_path)

    assert result["next_step"] == "ask_clarification"
    assert result["visualization"] is None
    assert result["relevant_tables"] is None


def test_clean_sql_query_resets_visualization_but_keeps_relevant_tables(
    router_prompt_path,
):
    decision = make_decision(
        intent="sql_query",
        is_answerable=True,
        clarification=None,
        relevant_tables=["orders", "sales"],
    )
    result = classify_with(decision, router_prompt_path)

    assert result["next_step"] == "sql_query"
    assert result["visualization"] is None
    assert result["relevant_tables"] == ["orders", "sales"]