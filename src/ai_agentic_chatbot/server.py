from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from langchain_core.messages import HumanMessage
from sqlalchemy import text, Engine
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from ai_agentic_chatbot.agent.graph import build_graph
from ai_agentic_chatbot.agent.schema import StreamRequest
from ai_agentic_chatbot.infrastructure.datasource.datasource_init import (
    initialize_datasources,
)
from ai_agentic_chatbot.infrastructure.datasource.factory import get_datasource_factory, get_engine
from ai_agentic_chatbot.infrastructure.db_depency import get_db_session
from ai_agentic_chatbot.logging_config import setup_logging, get_logger
from ai_agentic_chatbot.schema_extractor.SaveSchemaJson import save_schema_temp_file
from ai_agentic_chatbot.schema_extractor.SchemaExtractionConfig import SchemaExtractionConfig
from ai_agentic_chatbot.schema_extractor.SchemaExtractor import SchemaExtractor
from ai_agentic_chatbot.schema_extractor.transform_schema_to_text import transform_schema_to_text

load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)


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


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "UP"}


@app.get("/db-health", tags=["Health"])
def db_health(db: Session = Depends(get_db_session)):
    try:
        db.execute(text("SELECT 1"))
        return {"database": "UP"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/schemaJson", tags=["SchemaExtractor"])
def schema_json(db_engine: Engine = Depends(get_engine)):
    try:
        config = SchemaExtractionConfig(
            include_tables=["orders", "customer", "sales", "product", "inventory"]
        )

        extractor = SchemaExtractor(db_engine, config)
        schema = extractor.extract_database_schema()
        schema_file_path = save_schema_temp_file(schema)

        return {"SchemaPath": schema_file_path}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/schemaText", tags=["SchemaExtractor"])
def schema_text():
    try:
        transform_schema_to_text()

        return {"Schema to text conversion completed"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/stream")
async def stream_endpoint(stream_request: StreamRequest):
    """Streams agent responses using Server-Sent Events."""
    try:
        thread_id = stream_request.thread_id
        messages = stream_request.messages

        if not messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        last_msg = messages[-1].content

        graph = build_graph()

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=last_msg)]}

        async def event_generator():
            try:
                async for event in graph.astream_events(
                        inputs, config=config, version="v2"
                ):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            yield chunk.content.encode("utf-8")

            except Exception as e:
                print(f"[STREAM ERROR] {e}")
                yield b"[Error: Stream interrupted]"

        return StreamingResponse(event_generator(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API ERROR] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
