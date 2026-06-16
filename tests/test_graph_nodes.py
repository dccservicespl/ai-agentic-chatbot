"""Regression tests for issue #19 — fallback_node used to call the LLM
redundantly after router_node had already attached the full user-facing
message for the "nonsense" (gibberish + out_of_scope) path, producing a
duplicate reply and an unnecessary API call.
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from ai_agentic_chatbot.agent.graph import fallback_node, greeting_node


def test_fallback_node_makes_no_llm_call_and_returns_empty_update():
    state = {"messages": [HumanMessage(content="asdkjasdj")]}

    with patch("ai_agentic_chatbot.agent.graph.fast_llm") as mock_llm:
        result = fallback_node(state)

    mock_llm.invoke.assert_not_called()
    assert result == {}


def test_greeting_node_still_calls_llm():
    state = {"messages": [HumanMessage(content="hi")]}

    with patch("ai_agentic_chatbot.agent.graph.fast_llm") as mock_llm:
        mock_llm.invoke.return_value = AIMessage(content="Hello! How can I help?")
        result = greeting_node(state)

    mock_llm.invoke.assert_called_once()
    assert result["messages"][0].content == "Hello! How can I help?"