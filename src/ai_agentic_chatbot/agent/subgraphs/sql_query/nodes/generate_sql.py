"""SQL generation node with structured LLM output."""

import re
import json
from ai_agentic_chatbot.infrastructure.llm.factory import get_llm
from ai_agentic_chatbot.infrastructure.llm.types import LLMProvider, ModelType
from ai_agentic_chatbot.infrastructure.datasource.factory import get_engine
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import List, Optional
from ai_agentic_chatbot.utils.prompt_loader import get_system_prompt
from ai_agentic_chatbot.agent.subgraphs.sql_query.nodes.glossary_lookup import (
    fetch_glossary_hints,
    fetch_column_hints,
)
from ai_agentic_chatbot.infrastructure.context.context_settings import get_context_registry
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)

# TODO 18 (not yet done): /stream will inject the real db_context_id. Until
# then, every request resolves against this one real, configured context —
# "default" is not a registered context_id in config.yaml.
_FALLBACK_CONTEXT_ID = "sales"


class SQLGeneration(BaseModel):
    """Structured LLM output for SQL generation."""

    query: str = Field(description="Generated SQL query")
    explanation: str = Field(description="Plain English explanation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    tables_used: List[str] = Field(description="Tables referenced in query")
    warnings: Optional[List[str]] = None


def generate_sql_node(state: dict) -> dict:
    """
    Generate SQL query from retrieved schemas.
    """
    logger.info("[Generate SQL] Creating query")

    retrieved_tables = state.get("retrieved_tables", [])

    if not retrieved_tables:
        return {"validation_errors": ["Cannot generate SQL without table schemas"]}

    schema_text = "\n\n".join(
        [
            f"-- Table: {name} (Relevance: {score:.2f})\n{ddl}"
            for name, ddl, score in retrieved_tables
        ]
    )

    user_query = state["user_query"]

    previous_error = state.get("execution_error")
    generation_attempts = state.get("generation_attempts", 0)
    db_context_id = state.get("db_context_id") or _FALLBACK_CONTEXT_ID

    try:
        ctx = get_context_registry().get_context(db_context_id)
        engine = get_engine("postgresql.primary")
        table_names = [name for name, _ddl, _score in retrieved_tables]

        llm = get_llm(LLMProvider.AZURE_OPENAI, ModelType.SMART)
        structured_llm = llm.with_structured_output(SQLGeneration)

        prompt_content = _create_generation_prompt(
            schema_text=schema_text,
            user_query=user_query,
            previous_error=previous_error,
            generation_attempts=generation_attempts,
            glossary_hints=fetch_glossary_hints(user_query, engine, ctx.schema_name),
            column_hints=fetch_column_hints(table_names, engine, ctx.schema_name),
            system_prompt_path=ctx.system_prompt_path,
        )

        prompt = SystemMessage(content=prompt_content)

        try:
            result: SQLGeneration = structured_llm.invoke([prompt])
        except Exception as parse_error:
            # Llama/DeepSeek models sometimes append explanatory text after the JSON
            # block, causing "trailing characters" parse errors. Extract the JSON
            # object manually and parse via Pydantic as a fallback.
            if "trailing" in str(parse_error).lower() or "json" in str(parse_error).lower():
                logger.warning(
                    f"Structured output parse failed, trying JSON extraction fallback: {parse_error}"
                )
                raw = llm.invoke([prompt])
                result = SQLGeneration(**_extract_json_block(raw.content))
            else:
                raise parse_error

        logger.info(f"Generated SQL: {result.query}")
        logger.info(f"Confidence: {result.confidence}")

        return {
            "generated_sql": result.query,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "tables_used": result.tables_used,
            "generation_attempts": generation_attempts + 1,
        }

    except Exception as e:
        logger.error(f"SQL generation failed: {e}", exc_info=True)
        return {
            "validation_errors": [f"Generation error: {str(e)}"],
            "generation_attempts": generation_attempts + 1,
        }


def _create_generation_prompt(
        schema_text: str,
        user_query: str,
        previous_error: Optional[str] = None,
        generation_attempts: int = 0,
        glossary_hints: str = "",
        column_hints: str = "",
        system_prompt_path: Optional[str] = None,
) -> str:
    """Create the SQL generation prompt."""

    system_context = get_system_prompt(system_prompt_path)

    hints_block = "\n\n".join(filter(None, [glossary_hints, column_hints]))
    hints_section = f"\n\n{hints_block}" if hints_block else ""

    base_prompt = f"""{system_context}

---

## RETRIEVED SCHEMA CONTEXT

The following tables/views were retrieved as most relevant to the user's request.
Use them as the authoritative DDL reference — prefer v_sales_summary whenever it appears:

{schema_text}{hints_section}

## USER REQUEST

{user_query}

---

## RESPONSE FORMAT

Return ONLY a JSON object with exactly these fields — no markdown fences, no text before or after:

{{
    "query": "the complete SQL SELECT statement",
    "explanation": "plain English explanation of what the query does",
    "confidence": 0.95,
    "tables_used": ["table_or_view_names_used"],
    "warnings": ["any data caveats or limitations — omit if none"]
}}

CRITICAL: Follow every rule in the SQL GENERATION RULES section above. Return ONLY the JSON object."""

    # Add error feedback if retrying
    if previous_error and generation_attempts > 0:
        base_prompt += f"""

PREVIOUS ATTEMPT FAILED WITH ERROR:
{previous_error}

Generate a CORRECTED query that fixes this error.
Common fixes:
- Check column names and spelling (use exact names from schema)
- Verify JOIN conditions match foreign key relationships
- Ensure proper GROUP BY clauses (include all non-aggregate columns)
- Handle data type conversions properly
- Check table aliases and references
- Verify aggregate function usage
- Ensure WHERE clause syntax is correct

This is attempt #{generation_attempts + 1}. Be extra careful with:
1. Column name accuracy
2. Table relationship correctness
3. SQL syntax validation
"""

    return base_prompt


def _extract_json_block(text: str) -> dict:
    """Extract the first complete JSON object from a string that may contain trailing text.

    Needed because open-weight models (Llama, DeepSeek) sometimes append
    explanatory sentences after the JSON block, breaking standard parsers.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text[:200]}")
    return json.loads(match.group())
