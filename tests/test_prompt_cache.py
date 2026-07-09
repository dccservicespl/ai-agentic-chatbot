"""Unit tests for Section 38 Phase 6 — prompt-cache read/write wiring.

Follows tests/test_context_isolation.py's convention: plain functions, no
TestClient, no conftest — dependencies mocked directly with MagicMock/patch.
"""
from unittest.mock import MagicMock, patch

from ai_agentic_chatbot.utils.text_normalize import normalize_prompt
from ai_agentic_chatbot.schema_extractor.schema_hash import compute_schema_hash
from ai_agentic_chatbot.schema_extractor.SchemaModels import (
    ColumnSchema, DatabaseSchema, TableSchema,
)
from ai_agentic_chatbot.agent.subgraphs.sql_query.routes import (
    route_after_cache_lookup, route_after_execution,
)
from ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.cache_lookup import cache_lookup_node
from ai_agentic_chatbot.agent.subgraphs.sql_query.cache_sync import sync_prompt_cache


# ---------------------------------------------------------------------------
# normalize_prompt
# ---------------------------------------------------------------------------

def test_normalize_prompt_lowercases_trims_and_collapses_whitespace():
    assert normalize_prompt("  Top   10   Customers  ") == "top 10 customers"


def test_normalize_prompt_does_not_strip_punctuation_or_handle_synonyms():
    # v1 is deliberately simple — "top 10" and "top ten" must NOT normalize
    # to the same key, punctuation is preserved as-is.
    assert normalize_prompt("Top 10 customers?") == "top 10 customers?"
    assert normalize_prompt("top ten customers") != normalize_prompt("top 10 customers")


# ---------------------------------------------------------------------------
# compute_schema_hash
# ---------------------------------------------------------------------------

def _make_schema(tables: list) -> DatabaseSchema:
    """tables: list of (table_name, [column_names])."""
    return DatabaseSchema(
        database_name="test_db",
        tables=[
            TableSchema(
                schema_name="public",
                table_name=name,
                columns=[
                    ColumnSchema(name=c, data_type="text", nullable=True, default=None)
                    for c in columns
                ],
                primary_keys=["id"],
                foreign_keys=[],
            )
            for name, columns in tables
        ],
    )


def test_compute_schema_hash_is_order_independent():
    schema_a = _make_schema([("orders", ["id", "total"]), ("customers", ["id", "name"])])
    schema_b = _make_schema([("customers", ["id", "name"]), ("orders", ["id", "total"])])
    assert compute_schema_hash(schema_a) == compute_schema_hash(schema_b)


def test_compute_schema_hash_changes_when_a_column_is_added():
    before = _make_schema([("orders", ["id", "total"])])
    after = _make_schema([("orders", ["id", "total", "discount"])])
    assert compute_schema_hash(before) != compute_schema_hash(after)


# ---------------------------------------------------------------------------
# route_after_cache_lookup
# ---------------------------------------------------------------------------

def test_route_after_cache_lookup_hit_goes_straight_to_execute_query():
    assert route_after_cache_lookup({"cache_hit": True}) == "execute_query"


def test_route_after_cache_lookup_miss_falls_through_to_retrieve_schemas():
    assert route_after_cache_lookup({"cache_hit": False}) == "retrieve_schemas"


# ---------------------------------------------------------------------------
# route_after_execution — schema-drift fallback (one-shot, no loop)
# ---------------------------------------------------------------------------

def test_route_after_execution_cache_hit_not_found_falls_back_to_retrieve_schemas():
    state = {
        "execution_error": "relation \"foo\" does not exist",
        "error_category": "not_found",
        "cache_hit": True,
        "cache_fallback_used": True,  # execute_query_node sets this when triggering the fallback
        "generation_attempts": 0,     # never went through generate_sql on the hit path
        "max_retries": 2,
    }
    assert route_after_execution(state) == "retrieve_schemas"


def test_route_after_execution_does_not_refallback_after_generate_sql_has_run():
    # A second failure after the one-shot fallback already routed through
    # retrieve_schemas -> generate_sql once (generation_attempts is now >=1)
    # must use the NORMAL retry path, not trigger the fallback again.
    state = {
        "execution_error": "still broken",
        "error_category": "not_found",
        "cache_hit": True,
        "cache_fallback_used": True,
        "generation_attempts": 1,
        "max_retries": 2,
    }
    assert route_after_execution(state) == "generate_sql"


def test_route_after_execution_ends_after_max_retries_even_for_a_former_cache_hit():
    state = {
        "execution_error": "still broken",
        "error_category": "not_found",
        "cache_hit": True,
        "cache_fallback_used": True,
        "generation_attempts": 2,
        "max_retries": 2,
    }
    assert route_after_execution(state) == "END"


# ---------------------------------------------------------------------------
# cache_lookup_node
# ---------------------------------------------------------------------------

_CACHE_LOOKUP_MODULE = "ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.cache_lookup"


def test_cache_lookup_node_misses_when_schema_hash_is_none():
    fake_db = MagicMock()
    fake_ctx_row = MagicMock(schema_hash=None)

    with patch(f"{_CACHE_LOOKUP_MODULE}.get_session", return_value=fake_db), \
         patch(f"{_CACHE_LOOKUP_MODULE}.get_context_by_slug", return_value=fake_ctx_row):
        result = cache_lookup_node({"user_query": "top 10 customers", "db_context_id": "sales"})

    assert result["cache_hit"] is False
    assert result["schema_hash"] is None
    fake_db.close.assert_called_once()


def test_cache_lookup_hit_skips_retrieve_schemas_and_generate_sql():
    """Rebuilds a FRESH sql_subgraph with retrieve_schemas_node/generate_sql_node/
    execute_query_node patched at the point graph.py imported them — patching the
    leaf modules directly would NOT work here, because create_sql_subgraph()'s
    workflow.add_node(name, some_node) call captures whatever function object
    graph.py's own module-level name currently points to, and that binding
    already happened once at the first `import graph` (module-level
    `sql_subgraph = create_sql_subgraph()`). So: patch graph.py's names, THEN
    call create_sql_subgraph() again to get a fresh graph bound to the mocks.
    """
    from ai_agentic_chatbot.agent.subgraphs.sql_query import graph as sql_graph_module

    fake_db = MagicMock()
    fake_ctx_row = MagicMock(id=1, schema_hash="abc123", context_id="sales")
    fake_cache_row = MagicMock(
        id=42, generated_sql="SELECT 1", explanation="test explanation",
        chart_type="kpi", chart_config={},
    )
    fake_db.execute.return_value.scalar_one_or_none.return_value = fake_cache_row

    with patch(f"{_CACHE_LOOKUP_MODULE}.get_session", return_value=fake_db), \
         patch(f"{_CACHE_LOOKUP_MODULE}.get_context_by_slug", return_value=fake_ctx_row), \
         patch("ai_agentic_chatbot.agent.subgraphs.sql_query.graph.retrieve_schemas_node") as mock_retrieve, \
         patch("ai_agentic_chatbot.agent.subgraphs.sql_query.graph.generate_sql_node") as mock_generate, \
         patch("ai_agentic_chatbot.agent.subgraphs.sql_query.graph.execute_query_node") as mock_execute:
        mock_execute.return_value = {"query_result": [], "execution_error": None}

        fresh_subgraph = sql_graph_module.create_sql_subgraph()
        result = fresh_subgraph.invoke({
            "user_query": "top 10 customers",
            "router_table_hints": [],
            "db_context_id": "sales",
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
            "force_regenerate": False,
        })

    mock_retrieve.assert_not_called()
    mock_generate.assert_not_called()
    mock_execute.assert_called_once()
    assert result["cache_hit"] is True
    assert result["generated_sql"] == "SELECT 1"


# ---------------------------------------------------------------------------
# sync_prompt_cache
# ---------------------------------------------------------------------------

def test_sync_prompt_cache_skips_writing_on_execution_error():
    with patch(
        "ai_agentic_chatbot.agent.subgraphs.sql_query.cache_sync.get_session"
    ) as mock_get_session:
        sync_prompt_cache(
            {
                "execution_error": "boom",
                "normalized_prompt": "top 10 customers",
                "schema_hash": "abc123",
                "generated_sql": "SELECT 1",
            },
            {"type": "table"},
        )

    mock_get_session.assert_not_called()