from typing import List

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_postgres import PGVector
from rank_bm25 import BM25Okapi
import re

from ai_agentic_chatbot.infrastructure.vector_store.schema_retrieval_input import SchemaRetrieverInput


class Retriever:
    """
    Hybrid schema retriever:
    - Vector similarity (L2)
    - Text search
    - Metadata filtering
    - BM25 re-ranking
    """

    # Class-level vector store (shared)
    vector_store: PGVector | None = None

    def __init__(self, vector_store: PGVector):
        Retriever.vector_store = vector_store

    def retrieve(
            self,
            context: str,
            collection_name: str,
            *,
            k_vector: int = 20,
            k_final: int = 5,
            metadata_filter: dict | None = None,
    ) -> List[Document]:
        """
        Returns top-k relevant schema documents for the given context.
        """

        if not Retriever.vector_store:
            raise RuntimeError("Vector store is not initialized")

        # 1. Vector similarity search (L2 / cosine)
        vector_docs = Retriever.vector_store.similarity_search(
            query=context,
            k=k_vector,
            filter=metadata_filter,
            collection_name=collection_name,
        )

        if not vector_docs:
            return []

        # 2. Prepare corpus for BM25
        corpus = [doc.page_content for doc in vector_docs]
        tokenized_corpus = [self._tokenize(text) for text in corpus]

        # 3. Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)

        # 4. Score documents against query
        tokenized_query = self._tokenize(context)
        scores = bm25.get_scores(tokenized_query)

        # 5. Rank documents by BM25 score
        ranked_docs = sorted(
            zip(vector_docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 6. Select top-k
        top_docs = [doc for doc, _ in ranked_docs[:k_final]]

        return top_docs

    def as_tool(self) -> StructuredTool:
        """
        Exposes the retriever as a LangChain structured tool.
        """

        return StructuredTool.from_function(
            name="schema_retriever",
            description=(
                "Retrieves the most relevant database schema documents using "
                "hybrid vector similarity search and BM25 re-ranking. "
                "Use this BEFORE generating SQL."
            ),
            func=self.retrieve,
            args_schema=SchemaRetrieverInput,
        )

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"\b\w+\b", text)
