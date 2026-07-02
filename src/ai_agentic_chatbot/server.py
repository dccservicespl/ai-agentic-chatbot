from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Query
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from ai_agentic_chatbot.agent.graph import build_graph
from ai_agentic_chatbot.agent.schema import StreamRequest
from ai_agentic_chatbot.application.ingest_vector_schema import ingest_schema
from ai_agentic_chatbot.infrastructure.vector_store.pgvector_store import PgVectorSchemaStore
from ai_agentic_chatbot.infrastructure.datasource.datasource_init import (
    initialize_datasources,
)
from ai_agentic_chatbot.infrastructure.datasource.factory import (
    get_datasource_factory,
    get_engine,
)
from ai_agentic_chatbot.infrastructure.db_depency import get_db_session
from ai_agentic_chatbot.infrastructure.context.context_settings import (
    DbContextConfig,
    get_context_registry,
)
from ai_agentic_chatbot.logging_config import setup_logging, get_logger
from ai_agentic_chatbot.schema_extractor.SaveSchemaJson import save_schema_temp_file
from ai_agentic_chatbot.schema_extractor.SchemaExtractionConfig import (
    SchemaExtractionConfig,
)
from ai_agentic_chatbot.schema_extractor.SchemaExtractor import SchemaExtractor
from ai_agentic_chatbot.application.transform_schema_to_text import (
    transform_schema_to_text,
    generate_schema_summary,
)
from ai_agentic_chatbot.auth.router import router as auth_router
from ai_agentic_chatbot.auth.dependencies import get_auth_db, get_current_user
from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.auth.repository import count_prompts_today, create_prompt_log
from ai_agentic_chatbot.context.router import router as context_router
from ai_agentic_chatbot.context.repository import (
    get_context_by_slug,
    get_default_context,
    is_user_assigned_to_context,
)

load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_context(context_id: str) -> DbContextConfig:
    try:
        return get_context_registry().get_context(context_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def _resolve_stream_context(
    db: Session,
    current_user: User,
    requested_context_id: Optional[str],
    existing_context_id: Optional[str],
) -> str:
    """Resolve which context_id a /stream request should run against.

    requested_context_id: stream_request.context_id from the client (may be None).
    existing_context_id: db_context_id already stored in this thread_id's
    LangGraph checkpoint, if any (None for a brand-new thread_id).
    """
    if requested_context_id is not None:
        ctx = get_context_by_slug(db, requested_context_id)
        if ctx is None or not ctx.is_active:
            raise HTTPException(status_code=404, detail=f"Unknown context_id: {requested_context_id!r}")
        if not is_user_assigned_to_context(db, current_user.id, ctx.id):
            raise HTTPException(status_code=403, detail="You do not have access to this context")
        if existing_context_id is not None and existing_context_id != ctx.context_id:
            raise HTTPException(status_code=400, detail="Context switch requires a new thread_id")
        return ctx.context_id

    # No context_id supplied — continue an in-progress conversation's existing
    # context rather than re-resolving the user's default on every turn (that
    # would let a later default-context change silently retarget an
    # already-started thread with no error).
    if existing_context_id is not None:
        return existing_context_id

    default_ctx = get_default_context(db, current_user.id)
    if default_ctx is None:
        raise HTTPException(
            status_code=422,
            detail="No context assigned to this user. Contact an administrator.",
        )
    return default_ctx.context_id


@asynccontextmanager
async def lifespan(api: FastAPI):
    """Manage application lifespan events."""
    logger.info("Starting AI Agentic Chatbot application")

    logger.info("Initializing datasources...")
    try:
        factory = initialize_datasources()
        datasources = factory.list_datasources()
        logger.info(f"Datasources initialized successfully: {datasources}")
    except Exception as e:
        logger.error(f"Failed to initialize datasources: {e}", exc_info=True)

    yield

    logger.info("Shutting down application...")
    try:
        get_datasource_factory().close_all_connections()
        logger.info("Datasource connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing datasource connections: {e}", exc_info=True)

    logger.info("Application shutdown complete")


app = FastAPI(
    title="AI Chat Application",
    version="1.0.0",
    description="Agent enabled AI ChatBot application",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(context_router)


@app.get(
    "/health",
    tags=["Health"],
    summary="Liveness check",
    description="Returns `UP` if the server process is running. Use this as a basic liveness probe.",
    responses={200: {"description": "Server is running"}},
)
def health_check():
    return {"status": "UP"}


@app.get(
    "/db-health",
    tags=["Health"],
    summary="Database connectivity check",
    description="Executes a lightweight `SELECT 1` against the PostgreSQL datasource. "
                "Returns `UP` on success or HTTP 503 with the error detail if the database is unreachable.",
    responses={
        200: {"description": "Database is reachable"},
        503: {"description": "Database unreachable — connection or query failed"},
    },
)
def db_health(db: Session = Depends(get_db_session)):
    try:
        db["postgresql"].execute(text("SELECT 1"))
        return {"databases": "UP"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


graph = build_graph()


@app.get(
    "/schemaJson",
    tags=["SchemaExtractor"],
    summary="Extract database schema to JSON",
    description=(
            "Introspects the PostgreSQL database using SQLAlchemy and extracts the structural schema "
            "(tables, columns, primary keys, foreign keys) for the given context's PostgreSQL schema "
            "and table whitelist (both defined in config.yaml under `contexts.<context_id>`). "
            "The result is serialised to `<context.schema_dir>/db_schema.json` and the file path is returned. "
            "\n\n**Run this as Step 1 of the schema setup pipeline** before calling `/schemaText` or `/ingest`."
    ),
    responses={
        200: {"description": "Schema extracted successfully — returns path to the JSON file"},
        404: {"description": "Unknown context_id"},
        503: {"description": "Extraction failed — database unreachable or introspection error"},
    },
)
def schema_json(
    context_id: str = Query(..., description="Context slug from config.yaml's contexts: block"),
    current_user: User = Depends(get_current_user),
):
    ctx = _resolve_context(context_id)
    try:
        db_engine = get_engine("postgresql.primary")
        config = SchemaExtractionConfig(
            include_schemas=[ctx.schema_name],
            include_tables=ctx.include_tables,
        )

        extractor = SchemaExtractor(db_engine, config)
        schema = extractor.extract_database_schema()
        schema_file_path = save_schema_temp_file(schema, PROJECT_ROOT / ctx.schema_dir)

        return {"SchemaPath": schema_file_path}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get(
    "/schemaText",
    tags=["SchemaExtractor"],
    summary="Convert schema JSON to LLM-enriched documentation",
    description=(
            "Reads the schema JSON produced by `/schemaJson` for this context and sends each table to the LLM, "
            "which generates a human-readable `TableSchemaDocumentation` (business purpose, key fields, "
            "relationships, example questions). The output is saved as `<context.schema_dir>/schema_documentation.yaml`. "
            "\n\n**Run this as Step 2 of the schema setup pipeline** after `/schemaJson` and before `/ingest`."
    ),
    responses={
        200: {"description": "Schema converted to text documentation successfully"},
        404: {"description": "Unknown context_id"},
        503: {"description": "Conversion failed — missing schema JSON or LLM error"},
    },
)
def schema_text(
    context_id: str = Query(..., description="Context slug from config.yaml's contexts: block"),
    current_user: User = Depends(get_current_user),
):
    ctx = _resolve_context(context_id)
    try:
        schema_dir = PROJECT_ROOT / ctx.schema_dir
        transform_schema_to_text(schema_dir)
        generate_schema_summary(schema_dir)

        return {"Schema to text conversion completed"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get(
    "/ingest",
    tags=["SchemaExtractor"],
    summary="Ingest schema into vector store",
    description=(
            "Reads this context's LLM-generated schema documentation (`schema_documentation.yaml`), chunks it "
            "per table, embeds each chunk using Azure OpenAI embeddings, and upserts the vectors into this "
            "context's pgvector collection (`context.vector_collection_name`) in PostgreSQL. "
            "After this step the SQL agent can perform semantic table discovery at query time. "
            "\n\n**Run this as Step 3 of the schema setup pipeline** after `/schemaText`. "
            "Re-run whenever the database schema changes."
    ),
    responses={
        200: {"description": "Schema ingested into pgvector successfully"},
        404: {"description": "Unknown context_id"},
        500: {"description": "Ingestion failed — embedding or database error"},
    },
)
def ingest_schema_endpoint(
    context_id: str = Query(..., description="Context slug from config.yaml's contexts: block"),
    force_reset: bool = False,
    current_user: User = Depends(get_current_user),
):
    ctx = _resolve_context(context_id)
    try:
        if force_reset:
            PgVectorSchemaStore(collection_name=ctx.vector_collection_name).reset_collection()
            logger.info(f"force_reset=True: pgvector collection '{ctx.vector_collection_name}' cleared before ingest")

        schema_dir = PROJECT_ROOT / ctx.schema_dir
        ingest_schema(
            schema_path=schema_dir / "schema_documentation.yaml",
            context_id=ctx.context_id,
            collection_name=ctx.vector_collection_name,
        )
        return {"status": "ingested", "force_reset": force_reset, "context_id": ctx.context_id}
    except Exception as exc:
        raise exc


def build_stream_response_data(content: str, accumulated_state: dict) -> dict:
    """Build the SSE payload for one streamed turn.

    sql_query_node is the only graph branch that sets next_step to "end";
    greeting/fallback/clarification never produce a chart. Gating on that
    flag stops a chart from an earlier turn lingering in accumulated_state
    from being echoed back on an unrelated turn.
    """
    went_through_sql_node = accumulated_state.get("next_step") == "end"
    return {
        "content": content,
        "visualization": (
            accumulated_state.get("visualization") if went_through_sql_node else None
        ),
        "analysis": (
            accumulated_state.get("analysis") if went_through_sql_node else None
        ),
    }


@app.post(
    "/stream",
    tags=["Chat"],
    summary="Stream agent response via SSE",
    description=(
            "Main chat endpoint. Accepts a user message and a session `thread_id`, runs the full "
            "LangGraph agent workflow (intent routing → schema retrieval → SQL generation → execution → visualisation), "
            "and streams the response as **Server-Sent Events (SSE)**. "
            "\n\nEach SSE event is a JSON object: `{ \"content\": \"...\", \"visualization\": { ... } }`. "
            "The `visualization` field is `null` for non-SQL responses (greetings, clarifications) and "
            "contains chart config (type, data, axes, summary) for SQL query results. "
            "\n\nPass the same `thread_id` across turns to maintain multi-turn conversation memory."
    ),
    responses={
        200: {"description": "SSE stream — JSON events with content and optional visualization"},
        400: {"description": "Bad request — messages list is empty, or context switch requires a new thread_id"},
        403: {"description": "User is not assigned to the requested context"},
        404: {"description": "Unknown context_id"},
        422: {"description": "No context assigned to this user"},
        500: {"description": "Internal server error during agent execution"},
    },
)
async def stream_endpoint(
    stream_request: StreamRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    """Streams agent responses using Server-Sent Events."""
    try:
        thread_id = stream_request.thread_id
        messages = stream_request.messages

        if not messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = graph.get_state(config)
        existing_context_id = (
            state_snapshot.values.get("db_context_id")
            if state_snapshot and state_snapshot.values
            else None
        )

        db_context_id = _resolve_stream_context(
            db=db,
            current_user=current_user,
            requested_context_id=stream_request.context_id,
            existing_context_id=existing_context_id,
        )

        if current_user.daily_prompt_limit > 0:
            used_today = count_prompts_today(db, current_user.id)
            if used_today >= current_user.daily_prompt_limit:
                raise HTTPException(status_code=429, detail="Daily prompt limit reached")

        create_prompt_log(
            db,
            user_id=current_user.id,
            thread_id=thread_id,
            prompt_text=messages[-1].content,
        )

        inputs = {
            "messages": [HumanMessage(content=messages[-1].content)],
            "db_context_id": db_context_id,
        }

        if state_snapshot and state_snapshot.values:
            existing_messages = state_snapshot.values.get("messages", [])
            print(
                f"[CHECKPOINT DEBUG] Thread {thread_id} has {len(existing_messages)} messages:"
            )
            for i, msg in enumerate(existing_messages):
                print(f"  [{i}] {type(msg).__name__}: {msg.content[:50]}")
        else:
            print(f"[CHECKPOINT DEBUG] Thread {thread_id} has NO previous state")

        async def event_generator():
            try:
                # Track accumulated state for visualization data
                accumulated_state = {}

                async for chunk in graph.astream(
                        inputs, config=config, stream_mode="values"
                ):
                    # Update accumulated state with new chunk data
                    accumulated_state.update(chunk)

                    if "messages" in chunk and chunk["messages"]:
                        last_message = chunk["messages"][-1]

                        if isinstance(last_message, AIMessage):
                            content = last_message.content
                            if content:
                                response_data = build_stream_response_data(
                                    content, accumulated_state
                                )

                                import json

                                yield json.dumps(response_data).encode("utf-8")

            except Exception as e:
                logger.error(f"[STREAM ERROR] {e}")
                yield f"data: Error: {str(e)}\n\n".encode("utf-8")

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
