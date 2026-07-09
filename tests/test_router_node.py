"""Regression tests for issue #19 — visualization/relevant_tables state leaking
across conversation turns because RouterNode.classify() didn't reset them.
"""

from unittest.mock import MagicMock, patch

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


def classify_with(decision: RouterDecision) -> dict:
    """Run RouterNode.classify() with the LLM, schema loader, context registry,
    and prompt loaders all mocked out so the test runs offline and
    deterministically, independent of config.yaml/real prompt file contents."""
    state = {"messages": [HumanMessage(content="hello")]}

    fake_ctx = MagicMock(
        router_prompt_path="router_prompt.md",
        system_prompt_path="system_prompt.md",
        schema_dir="schema_dir/",
        out_of_scope_message="I can only help with sales-related questions.",
    )

    with patch("ai_agentic_chatbot.agent.router.get_llm") as mock_get_llm, \
         patch("ai_agentic_chatbot.agent.router.get_schema_loader") as mock_get_schema_loader, \
         patch("ai_agentic_chatbot.agent.router.get_system_prompt", return_value="SYSTEM"), \
         patch("ai_agentic_chatbot.agent.router.get_router_prompt", return_value="Schema:\n{schema_text}\n"), \
         patch("ai_agentic_chatbot.agent.router.get_context_registry") as mock_get_context_registry:
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = decision
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_llm.return_value = mock_llm

        mock_get_schema_loader.return_value.load_schema_summary.return_value = {
            "tables": [{"table": "orders", "bussiness_purpose": "Customer orders"}]
        }

        mock_get_context_registry.return_value.get_context.return_value = fake_ctx

        node = RouterNode(state)
        return node.classify()


def test_greeting_resets_visualization_and_tables():
    result = classify_with(make_decision(intent="greeting"))

    assert result["next_step"] == "greeting"
    assert result["visualization"] is None
    assert result["relevant_tables"] is None


def test_not_answerable_resets_visualization_and_tables():
    decision = make_decision(
        intent="nonsense",
        is_answerable=False,
        missing_data_reason="We do not have employee salary data.",
    )
    result = classify_with(decision)

    assert result["next_step"] == "nonsense"
    assert result["visualization"] is None
    assert result["relevant_tables"] is None


def test_ambiguous_sql_query_resets_visualization_and_tables():
    decision = make_decision(
        intent="sql_query",
        is_answerable=True,
        clarification=ClarificationDecision(
            is_ambiguous=True, clarification_question="Which time period?"
        ),
    )
    result = classify_with(decision)

    assert result["next_step"] == "ask_clarification"
    assert result["visualization"] is None
    assert result["relevant_tables"] is None


def test_out_of_scope_routes_to_nonsense_static_path():
    # "how are you" — coherent, not data-related, must not fall into the
    # 'greeting' branch (the original bug) and must not crash routing by
    # producing a next_step the graph doesn't know about.
    decision = make_decision(
        intent="out_of_scope",
        is_answerable=False,
        missing_data_reason="Casual chit-chat, no data requested.",
    )
    result = classify_with(decision)

    assert result["next_step"] == "nonsense"
    assert result["visualization"] is None
    assert result["relevant_tables"] is None


def test_out_of_scope_with_is_answerable_true_still_routes_safely():
    # Defensive case: even if the LLM mis-sets is_answerable=True for
    # out_of_scope chit-chat, next_step must still resolve to a graph edge
    # that exists ("nonsense"), not the raw "out_of_scope" string.
    decision = make_decision(intent="out_of_scope", is_answerable=True)
    result = classify_with(decision)

    assert result["next_step"] == "nonsense"
    assert result["visualization"] is None
    assert result["relevant_tables"] is None


def test_clean_sql_query_resets_visualization_but_keeps_relevant_tables():
    decision = make_decision(
        intent="sql_query",
        is_answerable=True,
        clarification=None,
        relevant_tables=["orders", "sales"],
    )
    result = classify_with(decision)

    assert result["next_step"] == "sql_query"
    assert result["visualization"] is None
    assert result["relevant_tables"] == ["orders", "sales"]