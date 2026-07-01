from typing import List, Dict, Tuple

from langchain_core.documents import Document
from langchain_postgres import PGVector

from ai_agentic_chatbot.infrastructure.datasource import get_engine
from ai_agentic_chatbot.infrastructure.embedding.embedding_connection import get_azure_openai_embedding
from ai_agentic_chatbot.logging_config import get_logger

logger = get_logger(__name__)


class PgVectorSchemaStore:
    """
    Infrastructure service responsible for:
    - Embedding schema text
    - Storing vectors in PostgreSQL (pgvector)
    - Searching stored vectors by semantic similarity

    collection_name must be unique per context (see DbContextConfig.vector_collection_name)
    — different contexts share the same underlying langchain_pg_collection /
    langchain_pg_embedding tables (in the public schema) and are isolated only by
    collection name. langchain_postgres has no schema= constructor option to
    physically separate them into their own PostgreSQL schema (checked 0.0.16
    and 0.0.17 — neither supports it).
    """

    def __init__(
            self,
            collection_name: str,
            embedding_model: str = "text-embedding-3-small",
    ):
        self._engine = get_engine("postgresql.primary")
        self._embedding = get_azure_openai_embedding()

        self._vectorstore = PGVector(
            connection=self._engine,
            collection_name=collection_name,
            embeddings=self._embedding,
        )

    def ingest(self, table_chunks: List[Dict]) -> None:
        """Upsert schema chunks into pgvector using stable deterministic IDs.

        Each chunk must carry an "id" field (uuid5 keyed on table_name) so that
        repeated /ingest calls UPDATE existing rows instead of inserting duplicates.
        """
        documents, ids = [], []

        for chunk in table_chunks:
            documents.append(
                Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"],
                )
            )
            ids.append(chunk["id"])

        self._vectorstore.add_documents(documents, ids=ids)
        logger.info(f"Upserted {len(documents)} schema chunks into pgvector")

    def reset_collection(self) -> None:
        """Drop and recreate the collection — use once to clear existing duplicates."""
        self._vectorstore.delete_collection()
        self._vectorstore.create_collection()
        logger.info("pgvector collection reset — all existing schema vectors deleted")

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Search stored schema vectors by semantic similarity to query.

        Uses similarity_search_with_relevance_scores (NOT similarity_search_with_score)
        because PGVector returns cosine *distance* (lower = better) from the raw method.
        The relevance variant converts it to cosine *similarity* (higher = better, 0-1)
        via: relevance = 1.0 - distance. The score_threshold of 0.3 applies directly.

        Costs exactly 1 embedding API call regardless of how many tables are stored.
        Deduplicates results by table_name keeping the highest score per table.

        Args:
            query: The user query to search against stored schema embeddings.
            k: Maximum number of results to return before threshold filtering.
            score_threshold: Minimum relevance score (0-1) to include a result.

        Returns:
            List of (table_name, relevance_score) sorted by score descending.
        """
        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=k)

        # Deduplicate by table_name — keep highest score per table as a safety
        # net against any residual duplicate rows in the collection.
        best: Dict[str, float] = {}
        for doc, score in results:
            if score < score_threshold:
                continue
            table_name = doc.metadata.get("table_name", "")
            if not table_name:
                logger.warning(f"pgvector document missing table_name in metadata: {doc.metadata}")
                continue
            if table_name not in best or score > best[table_name]:
                best[table_name] = score

        matched = sorted(best.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"pgvector search returned {len(matched)} tables above threshold {score_threshold}")
        return matched
