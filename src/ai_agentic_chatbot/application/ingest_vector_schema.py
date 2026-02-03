from pathlib import Path

from ai_agentic_chatbot.infrastructure.vector_store.pgvector_store import PgVectorSchemaStore
from ai_agentic_chatbot.schema_extractor.vector_schema_builder import VectorSchemaBuilder
from ai_agentic_chatbot.utils.utils import get_db_connection_string

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_TO_TEXT_PATH = BASE_DIR / "temp" / "schema_documentation.yaml"


def ingest_schema(schema_path: str, pg_conn_str: str) -> None:
    # Domain logic
    builder = VectorSchemaBuilder()
    schema = builder.load_schema(schema_path)
    table_chunks = builder.build_all_tables(schema)

    # Infrastructure logic
    store = PgVectorSchemaStore(
        connection_string=pg_conn_str,
        collection_name="db_schema_vectors"
    )

    store.ingest(table_chunks)


# if __name__ == "__main__":
#     ingest_schema(
#         schema_path=SCHEMA_TO_TEXT_PATH,
#         pg_conn_str=get_db_connection_string(),
#     )

# if __name__ == "__main__":
#     ingest_schema(
#         schema_path="schema.yaml",
#         pg_conn_str="postgresql+psycopg://user:pass@host:5432/db",
#     )
