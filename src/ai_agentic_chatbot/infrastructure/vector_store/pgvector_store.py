from typing import List, Dict

from langchain_core.documents import Document
from langchain_postgres import PGVector

from ai_agentic_chatbot.infrastructure.datasource import get_engine
from ai_agentic_chatbot.infrastructure.embedding.embedding_connection import get_azure_openai_embedding


class PgVectorSchemaStore:
    """
    Infrastructure service responsible for:
    - Embedding schema text
    - Storing vectors in PostgreSQL (pgvector)
    """

    def __init__(
            self,
            connection_string: str,
            collection_name: str = "db_schema_vectors",
            embedding_model: str = "text-embedding-3-small",
    ):
        # self._embedding = OpenAIEmbeddings(model=embedding_model)
        # self._embedding = get_embedding(provider=LLMProvider.AZURE_OPENAI,model=ModelType.EMBEDDING)

        self._engine = get_engine("postgresql.primary")
        self._embedding = get_azure_openai_embedding()

        self._vectorstore = PGVector(
            connection=self._engine,
            collection_name=collection_name,
            embeddings=self._embedding,
        )

    def ingest(self, table_chunks: List[Dict]) -> None:
        documents = []

        for chunk in table_chunks:
            documents.append(
                Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"],
                )
            )

        self._vectorstore.add_documents(documents)
