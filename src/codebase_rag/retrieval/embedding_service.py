"""Orchestrates text embedding for code chunks in the retrieval pipeline.

This service sits between the domain layer (CodeChunk) and the infrastructure
layer (IEmbeddingPort). It extracts text from chunks and delegates the actual
embedding to whichever IEmbeddingPort implementation is injected at construction
time, keeping this module free of any dependency on sentence-transformers,
OpenAI, or any other embedding library.
"""

from __future__ import annotations

import logging

from codebase_rag.domain.code_chunk import CodeChunk
from codebase_rag.domain.embedding_port import IEmbeddingPort

logger = logging.getLogger(__name__)

__all__ = ["EmbeddingService"]


class EmbeddingService:
    """Produces dense vectors for CodeChunks and natural-language queries.

    Delegates all model-specific logic to the injected IEmbeddingPort,
    keeping this service independent of any particular embedding library.

    Attributes:
        _embedder: Any IEmbeddingPort-compatible implementation.
    """

    def __init__(self, embedder: IEmbeddingPort) -> None:
        self._embedder = embedder

    @property
    def vector_size(self) -> int:
        """Dimensionality of vectors produced by the underlying embedder."""
        return self._embedder.vector_size

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Embed a batch of code chunks into dense vectors.

        Extracts the raw text from each chunk and delegates to
        ``IEmbeddingPort.embed_batch`` for a single efficient forward pass.

        Args:
            chunks: Non-empty list of code chunks to embed.

        Returns:
            A list of float vectors parallel to ``chunks``.
            ``result[i]`` is the embedding for ``chunks[i]``.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("Cannot embed an empty list of chunks.")

        texts = [chunk.text for chunk in chunks]
        logger.info("Embedding %d chunks...", len(texts))

        vectors = self._embedder.embed_batch(texts)

        logger.debug("Embedding complete. vector_size=%d", self._embedder.vector_size)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a natural-language query string for similarity search.

        Args:
            query: Raw natural-language question from the user.

        Returns:
            A single float vector of length ``vector_size``.

        Raises:
            ValueError: If ``query`` is blank or whitespace-only.
        """
        if not query.strip():
            raise ValueError("Query must not be blank.")

        logger.debug("Embedding query: %r", query[:80])
        return self._embedder.embed_one(query)
