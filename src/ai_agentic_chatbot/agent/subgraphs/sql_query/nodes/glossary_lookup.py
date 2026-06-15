"""Runtime glossary and column-hint lookups for SQL generation prompt enrichment."""

from typing import List
from sqlalchemy import text
from sqlalchemy.engine import Engine
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


def fetch_glossary_hints(user_query: str, engine: Engine) -> str:
    """Query business_glossary for terms that appear in the user query.

    Performs a case-insensitive substring match of each glossary term against
    the full user query text. Returns a formatted prompt block, or an empty
    string if no terms match or the table does not exist yet (pre-P0).
    """
    try:
        sql = text("""
            SELECT term, sql_meaning
            FROM business_glossary
            WHERE :query ILIKE '%' || term || '%'
            ORDER BY LENGTH(term) DESC
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, {"query": user_query}).fetchall()

        if not rows:
            return ""

        lines = ["## BUSINESS GLOSSARY (matched to your query)"]
        for term, sql_meaning in rows:
            lines.append(f'- "{term}" → {sql_meaning}')

        logger.debug(f"Glossary matched {len(rows)} term(s) for query")
        return "\n".join(lines)

    except Exception as e:
        if "does not exist" in str(e).lower() or "relation" in str(e).lower():
            logger.debug("business_glossary table not found — skipping glossary hints (run P0 migration first)")
        else:
            logger.warning(f"Glossary lookup failed: {e}")
        return ""


def fetch_column_hints(table_names: List[str], engine: Engine) -> str:
    """Query schema_metadata for column descriptions of the retrieved tables.

    Returns a formatted prompt block with per-column descriptions, data types,
    sample values, and join-key flags. Returns empty string if no tables
    provided, no rows found, or the table does not exist yet (pre-P0).
    """
    if not table_names:
        return ""

    try:
        sql = text("""
            SELECT table_name, column_name, data_type, description, sample_values, is_join_key
            FROM schema_metadata
            WHERE table_name = ANY(:tables)
              AND column_name IS NOT NULL
            ORDER BY table_name, column_name
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, {"tables": table_names}).fetchall()

        if not rows:
            return ""

        lines = ["## COLUMN HINTS (for retrieved tables)"]
        for table_name, column_name, data_type, description, sample_values, is_join_key in rows:
            join_flag = ", join key" if is_join_key else ""
            type_info = f"[{data_type}{join_flag}]" if data_type else ""
            samples = f" e.g. {sample_values}" if sample_values else ""
            lines.append(f"- {table_name}.{column_name} {type_info}: {description}{samples}")

        logger.debug(f"Column hints fetched for {len(table_names)} table(s): {len(rows)} column(s)")
        return "\n".join(lines)

    except Exception as e:
        if "does not exist" in str(e).lower() or "relation" in str(e).lower():
            logger.debug("schema_metadata table not found — skipping column hints (run P0 migration first)")
        else:
            logger.warning(f"Column hints lookup failed: {e}")
        return ""