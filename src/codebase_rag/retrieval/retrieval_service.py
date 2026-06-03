"""High-level retrieval service — the public API of the retrieval layer.

Example::

    from codebase_rag.retrieval.retrieval_service import RetrievalService
    from codebase_rag.core.config import Settings

    settings = Settings()
    service = RetrievalService.from_settings(settings)

    service.index_chunks(chunks)
    results = service.retrieve("where is login handled?", top_k=8)
"""

from __future__ import annotations

import logging

from codebase_rag.core.config import Settings
from codebase_rag.domain.code_chunk import CodeChunk
from codebase_rag.domain.retreival_result import RetrievalResult
from codebase_rag.domain.vector_store import IVectorStore, SearchFilters
from codebase_rag.infrastructure.embedding.sentence_transformers_embedder import STEmbedder
from codebase_rag.infrastructure.vectordb.faiss import FaissStore
from codebase_rag.retrieval.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

__all__ = ["RetrievalService"]


class RetrievalService:
    """Orchestrates embedding and vector search for code chunk retrieval.

    This is the only class from the retrieval layer that external modules
    should import. It delegates to EmbeddingService and IVectorStore,
    keeping all infrastructure choices hidden behind those interfaces.

    Attributes:
        _embedding_service: Handles text-to-vector conversion.
        _vector_store:      Handles vector storage and similarity search.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: IVectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalService:
        """Build a production RetrievalService from application settings.

        Constructs the embedder first (vector_size comes from it), then
        builds the vector store, initialises the collection, and returns
        a fully wired RetrievalService ready for use.

        Args:
            settings: Application settings.

        Returns:
            A RetrievalService with an initialised vector store.
        """
        embedder = STEmbedder(
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            device=settings.device,
        )
        embedding_service = EmbeddingService(embedder=embedder)

        vector_store = FaissStore.from_settings(vector_size=embedder.vector_size)
        vector_store.create_collection()

        logger.info(
            "RetrievalService initialised (model=%s, dim=%d).",
            settings.embedding_model_name,
            embedder.vector_size,
        )
        return cls(embedding_service=embedding_service, vector_store=vector_store)

    def index_chunks(self, chunks: list[CodeChunk]) -> None:
        """Embed and store a batch of code chunks in the vector store.

        Args:
            chunks: Code chunks produced by the ingestion layer.
        """
        logger.info("Indexing %d chunk(s)...", len(chunks))
        vectors = self._embedding_service.embed_chunks(chunks)
        self._vector_store.upsert(chunks, vectors)
        logger.info("Indexing complete.")

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        filters: SearchFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant code chunks for a natural-language query.

        Args:
            query:   Natural-language question about the codebase.
            top_k:   Maximum number of chunks to return.
            filters: Optional dict to narrow results. Supported keys:
                    ``language``, ``file_path``, ``repo_name``.

        Returns:
            Results ordered by descending relevance score.
        """
        logger.debug("Retrieving top-%d chunks for query: %r", top_k, query[:80])
        query_vector = self._embedding_service.embed_query(query)
        return self._vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
