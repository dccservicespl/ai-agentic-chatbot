"""Regression tests for issue #19 — server.py used to echo whatever
visualization was left in accumulated_state on every turn, even when the
current turn never touched sql_query_node.
"""

from ai_agentic_chatbot.server import build_stream_response_data


def test_sql_turn_keeps_its_visualization():
    accumulated_state = {
        "next_step": "end",
        "visualization": {"type": "pie_chart", "title": "Distribution"},
    }

    result = build_stream_response_data("Here's your data:", accumulated_state)

    assert result["visualization"] == {"type": "pie_chart", "title": "Distribution"}


def test_greeting_turn_drops_stale_visualization_from_earlier_sql_turn():
    # Simulates a checkpoint where an earlier SQL turn's chart is still
    # sitting in state, but this turn's next_step is "greeting".
    accumulated_state = {
        "next_step": "greeting",
        "visualization": {"type": "pie_chart", "title": "Distribution"},
    }

    result = build_stream_response_data("Hi there!", accumulated_state)

    assert result["visualization"] is None


def test_nonsense_turn_drops_stale_visualization():
    accumulated_state = {
        "next_step": "nonsense",
        "visualization": {"type": "bar_chart", "title": "Sales by Region"},
    }

    result = build_stream_response_data("I can help you with: orders, sales.", accumulated_state)

    assert result["visualization"] is None


def test_missing_next_step_defaults_to_no_visualization():
    accumulated_state = {"visualization": {"type": "kpi"}}

    result = build_stream_response_data("Some reply", accumulated_state)

    assert result["visualization"] is None