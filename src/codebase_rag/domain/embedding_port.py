"""Abstract port defining the interface for text embedding models.

Any class that satisfies this Protocol can serve as an embedder in the
retrieval pipeline, regardless of the underlying provider or library.
Implementors live in ``infrastructure/embeddings/`` and are never imported
directly by the retrieval or domain layers.

Implementors:
    - ``SentenceTransformerEmbedder`` (infrastructure/embeddings/)
    - ``OpenAIEmbedder``             (infrastructure/embeddings/)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["IEmbeddingPort"]


@runtime_checkable
class IEmbeddingPort(Protocol):
    """Structural interface for text embedding model implementations.

    A class satisfies this Protocol if it exposes ``vector_size``,
    ``embed_one``, and ``embed_batch`` with the correct signatures.
    Explicit inheritance from this class is not required.
    """

    @property
    def vector_size(self) -> int:
        """Dimensionality of vectors produced by this embedder."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single raw string into a dense vector.

        Args:
            text: Raw string to embed. Must be non-empty.

        Returns:
            A float vector of length ``vector_size``.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one batched forward pass.

        Prefer this over calling ``embed_one`` in a loop — batch encoding
        is significantly faster on both CPU and GPU.

        Args:
            texts: Non-empty list of raw strings to embed.

        Returns:
            A list of float vectors parallel to ``texts``.
            Each vector has length ``vector_size``.
        """
        ...
