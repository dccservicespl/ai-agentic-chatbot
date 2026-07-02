"""Integration tests for TODO 21 (technical_doc.md, section 30, Phase 8):
"Write integration tests for context isolation".

Covers, one test method per bullet point:

1. A query from context A never touches context B's tables or vector
   collection (SQL schema isolation via search_path + pgvector collection
   isolation via collection_name).
2. A user with zero assigned contexts gets HTTP 422, not HTTP 500.
3. A user not assigned to a specific requested context gets HTTP 403.
4. `SET LOCAL search_path` resets correctly between requests when a pooled
   connection is reused (no leakage across connection.close()/reconnect).
5. Switching context on an existing thread_id returns HTTP 400.

Tests 1 and 4 require a reachable PostgreSQL instance (the same one the app
uses, resolved via the "postgresql.primary" datasource). They probe
connectivity with a genuine `engine.connect()` attempt and skip gracefully
(not fail) if the database is unreachable in the current environment. Tests
2, 3 and 5 are pure unit tests against `_resolve_stream_context` with all
dependencies mocked — they require no external infrastructure and must
always pass.
"""

import random
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared helpers (not test methods) for Tests 1 and 4.
# ---------------------------------------------------------------------------

def _get_pg_engine_or_skip():
    """Genuinely attempt to obtain a working "postgresql.primary" engine.

    Registers datasources from config.yaml if they aren't already registered
    (module-level registration only happens via the app's lifespan handler,
    which does not run under pytest), then performs a real connect() to
    confirm the database is reachable. Skips the calling test if not.
    """
    from ai_agentic_chatbot.infrastructure.datasource.datasource_init import (
        initialize_datasources,
    )
    from ai_agentic_chatbot.infrastructure.datasource.factory import get_engine

    try:
        initialize_datasources()
    except Exception:
        # Already registered (e.g. a previous test in this session), or
        # registration itself failed — either way, let the connect() attempt
        # below be the authoritative reachability check.
        pass

    try:
        engine = get_engine("postgresql.primary")
        conn = engine.connect()
        conn.close()
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {e}")
        return None

    return engine


class _FakeEmbeddings:
    """Deterministic stand-in for the langchain `Embeddings` interface.

    Avoids any real Azure OpenAI API call. Vectors are derived from a hash of
    the input text so that distinct texts get distinguishable (but stable)
    embeddings without needing to be semantically realistic.
    """

    _DIM = 8

    def _vector_for(self, text_value: str) -> list[float]:
        rng = random.Random(hash(text_value))
        return [rng.uniform(-1.0, 1.0) for _ in range(self._DIM)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text_value: str) -> list[float]:
        return self._vector_for(text_value)


# ---------------------------------------------------------------------------
# Test 1 — SQL table isolation + pgvector collection isolation.
# ---------------------------------------------------------------------------

def test_context_a_query_never_touches_context_b_tables_or_vectors():
    engine = _get_pg_engine_or_skip()

    schema_a = "test_iso_ctx_a"
    schema_b = "test_iso_ctx_b"

    try:
        # --- SQL table isolation -------------------------------------------------
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_a}"))
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_b}"))
            conn.execute(text(f"CREATE TABLE {schema_a}.probe (id INT, label TEXT)"))
            conn.execute(text(f"CREATE TABLE {schema_b}.probe (id INT, label TEXT)"))
            conn.execute(
                text(f"INSERT INTO {schema_a}.probe (id, label) VALUES (1, 'A-only-row')")
            )
            conn.execute(
                text(f"INSERT INTO {schema_b}.probe (id, label) VALUES (1, 'B-only-row')")
            )

        # Mirrors execute_query.py's SET LOCAL pattern exactly, scoped to
        # context A's schema only.
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text(f"SET LOCAL search_path TO {schema_a}, public"))
                result = conn.execute(text("SELECT label FROM probe"))
                rows = [row[0] for row in result.fetchall()]

        assert "A-only-row" in rows
        assert "B-only-row" not in rows

        # --- pgvector collection isolation ----------------------------------------
        collection_a = "test_iso_vectors_a"
        collection_b = "test_iso_vectors_b"

        with patch(
            "ai_agentic_chatbot.infrastructure.vector_store.pgvector_store.get_azure_openai_embedding",
            return_value=_FakeEmbeddings(),
        ):
            from ai_agentic_chatbot.infrastructure.vector_store.pgvector_store import (
                PgVectorSchemaStore,
            )

            store_a = PgVectorSchemaStore(collection_name=collection_a)
            store_b = PgVectorSchemaStore(collection_name=collection_b)

            try:
                store_a.ingest(
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "content": "table_only_in_a stores rows exclusive to context A",
                            "metadata": {"table_name": "table_only_in_a"},
                        }
                    ]
                )
                store_b.ingest(
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "content": "table_only_in_b stores rows exclusive to context B",
                            "metadata": {"table_name": "table_only_in_b"},
                        }
                    ]
                )

                results_a = store_a.search(
                    "table_only_in_a stores rows exclusive to context A", score_threshold=0.0
                )
                tables_a = {name for name, _score in results_a}
                assert "table_only_in_a" in tables_a
                assert "table_only_in_b" not in tables_a

                results_b = store_b.search(
                    "table_only_in_b stores rows exclusive to context B", score_threshold=0.0
                )
                tables_b = {name for name, _score in results_b}
                assert "table_only_in_b" in tables_b
                assert "table_only_in_a" not in tables_b
            finally:
                store_a.reset_collection()
                store_b.reset_collection()
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_a} CASCADE"))
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_b} CASCADE"))


# ---------------------------------------------------------------------------
# Test 2 — zero assigned contexts -> HTTP 422, not 500.
# ---------------------------------------------------------------------------

def test_zero_assigned_contexts_returns_422_not_500():
    from ai_agentic_chatbot.server import _resolve_stream_context

    fake_user = MagicMock(id=1)
    fake_db = MagicMock()

    with patch("ai_agentic_chatbot.server.get_default_context", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            _resolve_stream_context(
                db=fake_db,
                current_user=fake_user,
                requested_context_id=None,
                existing_context_id=None,
            )

    assert exc_info.value.status_code == 422
    assert "No context assigned" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Test 3 — user not assigned to the requested context -> HTTP 403.
# ---------------------------------------------------------------------------

def test_unassigned_context_returns_403():
    from ai_agentic_chatbot.server import _resolve_stream_context

    fake_user = MagicMock(id=1)
    fake_db = MagicMock()
    fake_ctx = MagicMock(id=99, context_id="hr", is_active=True)

    with patch("ai_agentic_chatbot.server.get_context_by_slug", return_value=fake_ctx), \
         patch("ai_agentic_chatbot.server.is_user_assigned_to_context", return_value=False):
        with pytest.raises(HTTPException) as exc_info:
            _resolve_stream_context(
                db=fake_db,
                current_user=fake_user,
                requested_context_id="hr",
                existing_context_id=None,
            )

    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Test 4 — SET LOCAL search_path must not leak across pooled connections.
# ---------------------------------------------------------------------------

def test_search_path_resets_between_pooled_connections():
    engine = _get_pg_engine_or_skip()

    # First "request": set a distinctive, non-default search_path inside an
    # explicit transaction — mirrors execute_query.py's SET LOCAL pattern.
    with engine.connect() as conn1:
        with conn1.begin():
            conn1.execute(text("SET LOCAL search_path TO pg_catalog"))
            result = conn1.execute(text("SHOW search_path"))
            assert result.scalar() == "pg_catalog"
        # transaction committed here — SET LOCAL should have reset already.

    # Second "request" — possibly the same pooled connection (small pool),
    # possibly a different one. Either way, search_path must NOT still be
    # pg_catalog — proving SET LOCAL didn't leak across pool reuse.
    with engine.connect() as conn2:
        result = conn2.execute(text("SHOW search_path"))
        leaked_path = result.scalar()

    assert leaked_path != "pg_catalog"


# ---------------------------------------------------------------------------
# Test 5 — switching context on the same thread_id -> HTTP 400.
# ---------------------------------------------------------------------------

def test_context_switch_on_same_thread_returns_400():
    from ai_agentic_chatbot.server import _resolve_stream_context

    fake_user = MagicMock(id=1)
    fake_db = MagicMock()
    fake_ctx = MagicMock(id=99, context_id="hr", is_active=True)

    with patch("ai_agentic_chatbot.server.get_context_by_slug", return_value=fake_ctx), \
         patch("ai_agentic_chatbot.server.is_user_assigned_to_context", return_value=True):
        with pytest.raises(HTTPException) as exc_info:
            _resolve_stream_context(
                db=fake_db,
                current_user=fake_user,
                requested_context_id="hr",
                existing_context_id="sales",  # thread's existing checkpointed context differs
            )

    assert exc_info.value.status_code == 400
    assert "new thread_id" in exc_info.value.detail
