"""Unit tests for Section 38 Phase 6 — /history endpoints (View/Refresh/Regenerate)
and the /stream post-turn prompt_history write.

Same convention as tests/test_context_isolation.py: endpoint functions called
directly with MagicMock db/current_user, no TestClient, no conftest.
"""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from sqlalchemy import text

from ai_agentic_chatbot.history.router import (
    get_history, list_history, list_history_for_thread_endpoint, refresh_prompt, regenerate_prompt,
)

_ROUTER_MODULE = "ai_agentic_chatbot.history.router"


# ---------------------------------------------------------------------------
# List (context filtering)
# ---------------------------------------------------------------------------

def test_list_history_resolves_context_slug_to_db_id():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_ctx_row = MagicMock(id=7)

    with patch(f"{_ROUTER_MODULE}.get_context_by_slug", return_value=fake_ctx_row) as mock_get_slug, \
         patch(f"{_ROUTER_MODULE}.list_history_for_user", return_value=[]) as mock_list:
        list_history(limit=20, offset=0, context_id="demo_01", current_user=fake_current_user, db=fake_db)

    mock_get_slug.assert_called_once_with(fake_db, "demo_01")
    call_kwargs = mock_list.call_args.kwargs
    assert call_kwargs["context_db_id"] == 7


def test_list_history_unknown_context_slug_raises_404():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)

    with patch(f"{_ROUTER_MODULE}.get_context_by_slug", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            list_history(limit=20, offset=0, context_id="unknown_ctx", current_user=fake_current_user, db=fake_db)

    assert exc_info.value.status_code == 404


def test_list_history_without_context_id_passes_none():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)

    with patch(f"{_ROUTER_MODULE}.get_context_by_slug") as mock_get_slug, \
         patch(f"{_ROUTER_MODULE}.list_history_for_user", return_value=[]) as mock_list:
        list_history(limit=20, offset=0, context_id=None, current_user=fake_current_user, db=fake_db)

    mock_get_slug.assert_not_called()
    call_kwargs = mock_list.call_args.kwargs
    assert call_kwargs["context_db_id"] is None


# ---------------------------------------------------------------------------
# List (session grouping) — real Postgres, DISTINCT ON + window function
# ---------------------------------------------------------------------------
#
# list_history_for_user's DISTINCT ON (thread_id) + MAX(...) OVER (...) query
# can only be verified against genuine Postgres, not a mock — this is the one
# integration-style test in this file. Mirrors test_context_isolation.py's
# skip-gracefully-if-unreachable convention; the helper is duplicated here per
# this repo's no-conftest.py convention (test_db_connection.py does the same).

def _get_pg_session_or_skip():
    from ai_agentic_chatbot.infrastructure.datasource.datasource_init import (
        initialize_datasources,
    )
    from ai_agentic_chatbot.infrastructure.datasource.factory import get_session

    try:
        initialize_datasources()
    except Exception:
        pass

    try:
        session = get_session("postgresql.primary")
        session.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {e}")
        return None

    return session


def test_list_history_for_user_groups_by_thread_earliest_prompt_latest_activity():
    from ai_agentic_chatbot.auth.models import PromptHistory, User
    from ai_agentic_chatbot.context.models import DbContext
    from ai_agentic_chatbot.history.repository import list_history_for_user

    db = _get_pg_session_or_skip()

    suffix = uuid.uuid4().hex[:8]
    user = User(
        username=f"test_hist_{suffix}", email=f"test_hist_{suffix}@example.com", hashed_password="x",
    )
    ctx = DbContext(
        context_id=f"test_ctx_{suffix}", display_name="Test Context", schema_name="public",
        system_prompt_path="unused", router_prompt_path="unused", schema_dir="unused",
        vector_collection_name=f"test_vec_{suffix}", include_tables=[],
    )
    user_id = ctx_id = None

    try:
        db.add_all([user, ctx])
        db.commit()
        db.refresh(user)
        db.refresh(ctx)
        user_id, ctx_id = user.id, ctx.id

        now = datetime.now(timezone.utc)
        db.add_all([
            PromptHistory(user_id=user_id, thread_id="thread-A", db_context_id=ctx_id,
                          raw_prompt="first A", executed_at=now - timedelta(minutes=30)),
            PromptHistory(user_id=user_id, thread_id="thread-A", db_context_id=ctx_id,
                          raw_prompt="second A", executed_at=now - timedelta(minutes=20)),
            PromptHistory(user_id=user_id, thread_id="thread-A", db_context_id=ctx_id,
                          raw_prompt="third A", executed_at=now - timedelta(minutes=5)),
            PromptHistory(user_id=user_id, thread_id="thread-B", db_context_id=ctx_id,
                          raw_prompt="only B", executed_at=now - timedelta(minutes=15)),
        ])
        db.commit()

        results = list_history_for_user(db, user_id, limit=20, offset=0)

        assert len(results) == 2  # one row per thread_id, not one per turn

        # thread-A's latest turn (5 min ago) is more recent than thread-B's
        # only turn (15 min ago), so A sorts first — but its raw_prompt is
        # still the *earliest* turn's, not the latest's.
        assert results[0].thread_id == "thread-A"
        assert results[0].raw_prompt == "first A"
        assert results[0].latest_activity == now - timedelta(minutes=5)

        assert results[1].thread_id == "thread-B"
        assert results[1].raw_prompt == "only B"
        assert results[1].latest_activity == now - timedelta(minutes=15)
    finally:
        if user_id is not None:
            db.execute(text("DELETE FROM prompt_history WHERE user_id = :uid"), {"uid": user_id})
        if ctx_id is not None:
            db.execute(text("DELETE FROM app.db_contexts WHERE id = :cid"), {"cid": ctx_id})
        if user_id is not None:
            db.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})
        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# List by thread_id (context filtering, pagination, ownership from JWT)
# ---------------------------------------------------------------------------

def test_list_history_for_thread_resolves_context_slug_to_db_id():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_ctx_row = MagicMock(id=7)

    with patch(f"{_ROUTER_MODULE}.get_context_by_slug", return_value=fake_ctx_row) as mock_get_slug, \
         patch(f"{_ROUTER_MODULE}.list_history_for_thread", return_value=[]) as mock_list:
        list_history_for_thread_endpoint(
            thread_id="thread-1", limit=20, offset=0, context_id="demo_01",
            current_user=fake_current_user, db=fake_db,
        )

    mock_get_slug.assert_called_once_with(fake_db, "demo_01")
    call_kwargs = mock_list.call_args.kwargs
    assert call_kwargs["context_db_id"] == 7


def test_list_history_for_thread_unknown_context_slug_raises_404():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)

    with patch(f"{_ROUTER_MODULE}.get_context_by_slug", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            list_history_for_thread_endpoint(
                thread_id="thread-1", limit=20, offset=0, context_id="unknown_ctx",
                current_user=fake_current_user, db=fake_db,
            )

    assert exc_info.value.status_code == 404


def test_list_history_for_thread_scopes_by_current_user_not_a_user_id_param():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=42)

    with patch(f"{_ROUTER_MODULE}.list_history_for_thread", return_value=[]) as mock_list:
        list_history_for_thread_endpoint(
            thread_id="thread-1", limit=20, offset=0, context_id=None,
            current_user=fake_current_user, db=fake_db,
        )

    call_args, call_kwargs = mock_list.call_args
    assert call_args[1] == "thread-1"
    assert call_args[2] == 42  # current_user.id, not a client-supplied user_id


def test_list_history_for_thread_passes_pagination_through():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)

    with patch(f"{_ROUTER_MODULE}.list_history_for_thread", return_value=[]) as mock_list:
        list_history_for_thread_endpoint(
            thread_id="thread-1", limit=5, offset=10, context_id=None,
            current_user=fake_current_user, db=fake_db,
        )

    call_kwargs = mock_list.call_args.kwargs
    assert call_kwargs["limit"] == 5
    assert call_kwargs["offset"] == 10


# ---------------------------------------------------------------------------
# Ownership check
# ---------------------------------------------------------------------------

def test_get_history_404_for_another_users_item():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_row = MagicMock(user_id=2)  # belongs to a different user

    with patch(f"{_ROUTER_MODULE}.get_history_item", return_value=fake_row):
        with pytest.raises(HTTPException) as exc_info:
            get_history(history_id=42, current_user=fake_current_user, db=fake_db)

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

def test_refresh_never_calls_the_llm():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_row = MagicMock(user_id=1, generated_sql="SELECT count(*) AS n", chart_type="kpi", db_context_id=7)
    fake_ctx_row = MagicMock(context_id="sales")

    def _fail_if_called(*a, **kw):
        raise AssertionError("Refresh must never invoke SQL generation")

    with patch(f"{_ROUTER_MODULE}.get_history_item", return_value=fake_row), \
         patch(f"{_ROUTER_MODULE}.get_context_by_id", return_value=fake_ctx_row), \
         patch(f"{_ROUTER_MODULE}.execute_query_node", return_value={"query_result": [{"n": 42}], "execution_error": None}), \
         patch(
             "ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.generate_sql.generate_sql_node",
             side_effect=_fail_if_called,
         ):
        result = refresh_prompt(history_id=1, current_user=fake_current_user, db=fake_db)

    assert result.generated_sql == "SELECT count(*) AS n"
    assert result.visualization["type"] == "kpi"


def test_refresh_reuses_forced_chart_type_when_shape_still_fits():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_row = MagicMock(user_id=1, generated_sql="SELECT count(*) AS n", chart_type="kpi", db_context_id=7)
    fake_ctx_row = MagicMock(context_id="sales")

    with patch(f"{_ROUTER_MODULE}.get_history_item", return_value=fake_row), \
         patch(f"{_ROUTER_MODULE}.get_context_by_id", return_value=fake_ctx_row), \
         patch(f"{_ROUTER_MODULE}.execute_query_node", return_value={"query_result": [{"n": 42}], "execution_error": None}):
        result = refresh_prompt(history_id=1, current_user=fake_current_user, db=fake_db)

    assert result.visualization["type"] == "kpi"
    assert result.visualization["type_reused"] is True


def test_refresh_404_for_another_users_item():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_row = MagicMock(user_id=2)

    with patch(f"{_ROUTER_MODULE}.get_history_item", return_value=fake_row):
        with pytest.raises(HTTPException) as exc_info:
            refresh_prompt(history_id=1, current_user=fake_current_user, db=fake_db)

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Regenerate
# ---------------------------------------------------------------------------

def test_regenerate_forces_cache_miss_and_overwrites_via_sync_prompt_cache():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_row = MagicMock(user_id=1, raw_prompt="top 10 customers", db_context_id=7)
    fake_ctx_row = MagicMock(context_id="sales")

    captured_input = {}

    def fake_invoke(subgraph_input):
        captured_input.update(subgraph_input)
        return {
            "generated_sql": "SELECT * FROM customers LIMIT 10",
            "explanation": "regenerated",
            "query_result": [{"id": 1}],
            "execution_error": None,
            "validation_errors": [],
            "cache_hit": False,
        }

    with patch(f"{_ROUTER_MODULE}.get_history_item", return_value=fake_row), \
         patch(f"{_ROUTER_MODULE}.get_context_by_id", return_value=fake_ctx_row), \
         patch(f"{_ROUTER_MODULE}.sql_subgraph") as mock_subgraph, \
         patch(f"{_ROUTER_MODULE}.sync_prompt_cache") as mock_sync:
        mock_subgraph.invoke.side_effect = fake_invoke

        result = regenerate_prompt(history_id=1, current_user=fake_current_user, db=fake_db)

    assert captured_input["force_regenerate"] is True
    mock_sync.assert_called_once()
    assert result.generated_sql == "SELECT * FROM customers LIMIT 10"


def test_regenerate_updates_prompt_history_row_in_place():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_row = MagicMock(user_id=1, raw_prompt="top 10 customers", db_context_id=7)
    fake_ctx_row = MagicMock(context_id="sales")

    def fake_invoke(subgraph_input):
        return {
            "generated_sql": "SELECT * FROM customers LIMIT 10",
            "explanation": "regenerated",
            "query_result": [{"id": 1}],
            "execution_error": None,
            "validation_errors": [],
            "cache_hit": False,
        }

    with patch(f"{_ROUTER_MODULE}.get_history_item", return_value=fake_row), \
         patch(f"{_ROUTER_MODULE}.get_context_by_id", return_value=fake_ctx_row), \
         patch(f"{_ROUTER_MODULE}.sql_subgraph") as mock_subgraph, \
         patch(f"{_ROUTER_MODULE}.sync_prompt_cache"), \
         patch(f"{_ROUTER_MODULE}.update_prompt_history_after_regenerate") as mock_update_history:
        mock_subgraph.invoke.side_effect = fake_invoke

        regenerate_prompt(history_id=1, current_user=fake_current_user, db=fake_db)

    mock_update_history.assert_called_once()
    call_args, call_kwargs = mock_update_history.call_args
    assert call_args[0] is fake_db
    assert call_args[1] is fake_row
    assert call_kwargs["generated_sql"] == "SELECT * FROM customers LIMIT 10"


def test_regenerate_prompt_history_update_failure_is_non_fatal():
    fake_db = MagicMock()
    fake_current_user = MagicMock(id=1)
    fake_row = MagicMock(user_id=1, raw_prompt="top 10 customers", db_context_id=7)
    fake_ctx_row = MagicMock(context_id="sales")

    def fake_invoke(subgraph_input):
        return {
            "generated_sql": "SELECT * FROM customers LIMIT 10",
            "explanation": "regenerated",
            "query_result": [{"id": 1}],
            "execution_error": None,
            "validation_errors": [],
            "cache_hit": False,
        }

    with patch(f"{_ROUTER_MODULE}.get_history_item", return_value=fake_row), \
         patch(f"{_ROUTER_MODULE}.get_context_by_id", return_value=fake_ctx_row), \
         patch(f"{_ROUTER_MODULE}.sql_subgraph") as mock_subgraph, \
         patch(f"{_ROUTER_MODULE}.sync_prompt_cache"), \
         patch(
             f"{_ROUTER_MODULE}.update_prompt_history_after_regenerate",
             side_effect=Exception("db write failed"),
         ):
        mock_subgraph.invoke.side_effect = fake_invoke

        result = regenerate_prompt(history_id=1, current_user=fake_current_user, db=fake_db)

    assert result.generated_sql == "SELECT * FROM customers LIMIT 10"


# ---------------------------------------------------------------------------
# /stream: prompt_history is written only for SQL turns
# ---------------------------------------------------------------------------

from ai_agentic_chatbot.agent.schema import StreamRequest, Message


async def _run_stream_endpoint_and_drain(chunks: list[dict]):
    """Calls server.stream_endpoint directly (bypassing FastAPI's Depends
    machinery, same convention as this repo's other direct-function-call
    tests) with graph.astream faked to yield `chunks`, then fully drains the
    returned StreamingResponse so event_generator's post-astream-loop
    prompt_history write actually executes. Returns the write_prompt_history
    mock for assertions."""
    from ai_agentic_chatbot import server as server_module

    async def fake_astream(inputs, config, stream_mode):
        for chunk in chunks:
            yield chunk

    fake_graph = MagicMock()
    fake_graph.get_state.return_value = MagicMock(values=None)
    fake_graph.astream = fake_astream

    fake_db = MagicMock()
    fake_history_db = MagicMock()
    fake_current_user = MagicMock(id=1, daily_prompt_limit=0)

    stream_request = StreamRequest(
        thread_id="thread-1",
        messages=[Message(role="user", content="hello")],
        context_id=None,
    )

    with patch.object(server_module, "graph", fake_graph), \
         patch.object(server_module, "_resolve_stream_context", return_value="sales"), \
         patch.object(server_module, "create_prompt_log"), \
         patch.object(server_module, "get_session", return_value=fake_history_db), \
         patch.object(server_module, "get_context_by_slug", return_value=MagicMock(id=7)), \
         patch.object(server_module, "write_prompt_history") as mock_write_history:

        response = await server_module.stream_endpoint(
            stream_request, current_user=fake_current_user, db=fake_db
        )
        async for _ in response.body_iterator:
            pass

    return mock_write_history


@pytest.mark.asyncio
async def test_prompt_history_not_written_for_greeting_turn():
    mock_write_history = await _run_stream_endpoint_and_drain([
        {"next_step": "greeting", "messages": [AIMessage(content="Hi there!")]},
    ])
    mock_write_history.assert_not_called()


@pytest.mark.asyncio
async def test_prompt_history_written_for_sql_turn():
    mock_write_history = await _run_stream_endpoint_and_drain([
        {
            "next_step": "end",
            "generated_sql": "SELECT 1",
            "was_cache_hit": False,
            "cache_row_id": None,
            "visualization": {"type": "kpi"},
            "messages": [AIMessage(content="Here's your data:")],
        },
    ])
    mock_write_history.assert_called_once()


@pytest.mark.asyncio
async def test_prompt_history_result_snapshot_includes_analysis_text():
    mock_write_history = await _run_stream_endpoint_and_drain([
        {
            "next_step": "end",
            "generated_sql": "SELECT 1",
            "was_cache_hit": False,
            "cache_row_id": None,
            "visualization": {"type": "kpi"},
            "analysis": "Revenue is up 12% month over month.",
            "messages": [AIMessage(content="Here's your data:")],
        },
    ])
    call_kwargs = mock_write_history.call_args.kwargs
    assert call_kwargs["result_snapshot"] == {
        "type": "kpi",
        "analysis": "Revenue is up 12% month over month.",
    }