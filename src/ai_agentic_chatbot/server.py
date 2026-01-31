from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from ai_agentic_chatbot.controller.chat import router
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import uvicorn

from ai_agentic_chatbot.infrastructure.datasource.factory import get_datasource_factory, get_engine
from ai_agentic_chatbot.infrastructure.db_depency import get_db_session
from ai_agentic_chatbot.logging_config import setup_logging, get_logger
from sqlalchemy import text, Engine

from ai_agentic_chatbot.infrastructure.datasource.datasource_init import (
    initialize_datasources,
)
from ai_agentic_chatbot.schema_extractor.SaveSchemaJson import save_schema_temp_file
from ai_agentic_chatbot.schema_extractor.SchemaExtractionConfig import SchemaExtractionConfig
from ai_agentic_chatbot.schema_extractor.SchemaExtractor import SchemaExtractor

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
app.include_router(router)


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
