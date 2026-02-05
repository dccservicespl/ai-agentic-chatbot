from pathlib import Path

from dotenv import load_dotenv
import os

from ai_agentic_chatbot.infrastructure.vector_store.pgvector_store import PgVectorSchemaStore
from ai_agentic_chatbot.schema_extractor.vector_schema_builder import VectorSchemaBuilder
from ai_agentic_chatbot.utils.utils import get_db_connection_string

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_TO_TEXT_PATH = BASE_DIR / "temp" / "schema_documentation.yaml"


def ingest_schema(schema_path: str) -> None:
    # Domain logic
    builder = VectorSchemaBuilder()
    schema = builder.load_schema(schema_path)
    table_chunks = builder.build_all_tables(schema)

    # Infrastructure logic
    store = PgVectorSchemaStore(
        collection_name=os.getenv("VECTOR_COLLECTION_NAME")
    )

    store.ingest(table_chunks)

# if __name__ == "__main__":
#     ingest_schema(
#         schema_path="schema.yaml",
#         pg_conn_str="postgresql+psycopg://user:pass@host:5432/db",
#     )
