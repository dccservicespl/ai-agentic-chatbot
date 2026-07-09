"""Regression test: this file previously imported
`ai_agentic_chatbot.infrastructure.db_session.AsyncSessionLocal`, a module
that no longer exists anywhere in the codebase (confirmed via repo-wide grep —
no async SQLAlchemy session/engine is used anywhere). The actual, live DB
access pattern is the synchronous `DataSourceFactory`
(`infrastructure/datasource/factory.py`, `get_engine`/`get_session`), used by
every other module and test in this repo. Rewritten to use that pattern,
mirroring `test_context_isolation.py`'s skip-gracefully-if-unreachable
convention (no `conftest.py` in this repo — each test file is self-contained,
so the helper is duplicated here rather than shared).
"""

import pytest
from sqlalchemy import text


def _get_pg_engine_or_skip():
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


def test_database_connection():
    engine = _get_pg_engine_or_skip()

    with engine.connect() as conn:
        value = conn.execute(text("SELECT 1")).scalar_one()

    assert value == 1