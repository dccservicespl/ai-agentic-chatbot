from ai_agentic_chatbot.infrastructure.vector_store.pgvector_store import PgVectorSchemaStore
from ai_agentic_chatbot.schema_extractor.vector_schema_builder import VectorSchemaBuilder


def ingest_schema(schema_path: str, context_id: str, collection_name: str) -> None:
    # Domain logic
    builder = VectorSchemaBuilder()
    schema = builder.load_schema(schema_path)
    table_chunks = builder.build_all_tables(schema, context_id=context_id)

    # Infrastructure logic
    store = PgVectorSchemaStore(collection_name=collection_name)

    store.ingest(table_chunks)