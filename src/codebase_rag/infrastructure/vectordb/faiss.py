"""FAISS implementation of IVectorStore.

Uses an in-memory ``faiss.IndexFlatIP`` index for exact cosine similarity
search over L2-normalised vectors. Chunk metadata is kept in a parallel
``_chunks`` list aligned to FAISS integer positions.

Since FAISS does not natively support metadata or filtering, payload
filters are applied in Python after the vector search (post-filtering).

Vectors passed to ``upsert`` must be L2-normalised (unit length). This is
guaranteed when using ``STEmbedder`` with ``normalize_embeddings=True``.
"""

from __future__ import annotations

import logging

import faiss
import numpy as np

from codebase_rag.core.config import Settings
from codebase_rag.domain.code_chunk import CodeChunk
from codebase_rag.domain.retreival_result import RetrievalResult
from codebase_rag.domain.vector_store import SearchFilters

logger = logging.getLogger(__name__)

__all__ = ["FaissStore"]


class FaissStore:
    """IVectorStore implementation backed by an in-memory FAISS index.

    Vectors are stored in a ``faiss.IndexFlatIP`` index. Chunk metadata
    is maintained in a parallel ``_chunks`` list where position ``i``
    corresponds to the vector at FAISS position ``i``.

    Overwriting an existing chunk requires a full index rebuild since
    FAISS does not support in-place vector updates. For append-only
    workloads (the common MVP case), no rebuild is triggered.

    Attributes:
        vector_size: Dimensionality of vectors stored in the index.
    """

    def __init__(self, vector_size: int) -> None:
        self.vector_size = vector_size
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[CodeChunk] = []

    @classmethod
    def from_settings(cls, vector_size: int) -> FaissStore:
        """Construct a FaissStore with the given vector dimension.

        FAISS is purely in-memory and requires no connection config.
        Unlike ``QdrantStore``, no ``Settings`` object is needed.

        Args:
            vector_size: Dimensionality of vectors to store. Must match
                        the output dimension of the embedder exactly.

        Returns:
            A new ``FaissStore`` instance ready for ``create_collection()``.
        """
        return cls(vector_size=vector_size)

    def create_collection(self) -> None:
        """Initialise the FAISS index and prepare for upsert operations.

        This method is idempotent — if the index already exists it logs a
        warning and returns without modifying existing data. To wipe and
        recreate, call ``delete_collection()`` first.

        Raises:
            ValueError: If ``vector_size`` is not a positive integer.
        """
        if self.vector_size <= 0:
            raise ValueError(
                f"vector_size must be a positive integer, got {self.vector_size}."
            )

        if self._index is not None:
            logger.warning(
                "Collection already exists (%d vectors). "
                "Call delete_collection() first to reset.",
                self._index.ntotal,
            )
            return

        self._index = faiss.IndexFlatIP(self.vector_size)
        self._chunks = []
        logger.info("Created FAISS IndexFlatIP collection (dim=%d).", self.vector_size)

    def upsert(
        self,
        chunks: list[CodeChunk],
        vectors: list[list[float]],
        overwrite: bool = False,
    ) -> None:
        """Add chunks and their vectors to the FAISS index.

        New chunks are always appended. Behaviour for duplicate ``chunk_id``
        values is controlled by ``overwrite``:

        - ``overwrite=False``: duplicates are skipped with a warning.
        - ``overwrite=True``:  duplicates replace the existing entry.
            This triggers a full index rebuild since FAISS does not support
            in-place vector updates.

        Args:
            chunks:    Code chunks carrying metadata. Parallel to ``vectors``.
            vectors:   L2-normalised float vectors. Parallel to ``chunks``.
            overwrite: Whether to replace chunks with a duplicate
                        ``chunk_id``. Defaults to ``False``.

        Raises:
            RuntimeError: If ``create_collection()`` has not been called.
            ValueError:   If ``chunks`` and ``vectors`` differ in length.
        """
        if self._index is None:
            raise RuntimeError(
                "Index is not initialised. Call create_collection() first."
            )

        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have equal length: "
                f"{len(chunks)} != {len(vectors)}"
            )

        # O(1) lookup: chunk_id → position in self._chunks
        existing: dict[str, int] = {c.chunk_id: i for i, c in enumerate(self._chunks)}

        new_chunks: list[CodeChunk] = []
        new_vectors: list[list[float]] = []
        # position → (new chunk, new vector) for entries that need overwriting
        overwrite_map: dict[int, tuple[CodeChunk, list[float]]] = {}

        for chunk, vector in zip(chunks, vectors):
            pos = existing.get(chunk.chunk_id)
            if pos is not None:
                if overwrite:
                    overwrite_map[pos] = (chunk, vector)
                else:
                    logger.warning(
                        "Skipping duplicate chunk_id '%s'. "
                        "Pass overwrite=True to replace it.",
                        chunk.chunk_id,
                    )
            else:
                new_chunks.append(chunk)
                new_vectors.append(vector)

        # --- Overwrite path: mutate metadata + rebuild index ----------------
        if overwrite_map:
            for pos, (chunk, _) in overwrite_map.items():
                self._chunks[pos] = chunk

            # Collect all vectors: use new vector for overwritten positions,
            # reconstruct from FAISS for unchanged positions.
            all_vectors = [
                overwrite_map[i][1] if i in overwrite_map
                else self._index.reconstruct(i).tolist()
                for i in range(len(self._chunks))
            ]

            self._index.reset()
            self._index.add(np.array(all_vectors, dtype=np.float32))
            logger.info("Rebuilt index after %d overwrite(s).", len(overwrite_map))

        # --- Append path: add new chunks to existing index ------------------
        if new_chunks:
            self._index.add(np.array(new_vectors, dtype=np.float32))
            self._chunks.extend(new_chunks)
            logger.info(
                "Appended %d new chunk(s). Total in index: %d.",
                len(new_chunks),
                len(self._chunks),
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[RetrievalResult]:
        """Find the most similar chunks to a query vector.

        Performs exact inner-product search. Since FAISS has no native
        metadata filtering, ``filters`` are applied in Python after the
        vector search. To ensure ``top_k`` results are returned after
        filtering, the search over-fetches ``top_k * 10`` candidates first.

        Args:
            query_vector: L2-normalised query vector. Length must equal
                        ``vector_size``.
            top_k:        Maximum number of results to return.
            filters:      Optional payload filters applied post-search.
                        Supported keys: ``language``, ``file_path``,
                        ``repo_name``.

        Returns:
            Results ordered by descending similarity score. May return
            fewer than ``top_k`` items if the collection has fewer
            matching entries after filtering.

        Raises:
            RuntimeError: If ``create_collection()`` has not been called.
            ValueError:   If ``query_vector`` length does not match
                        ``vector_size``, or ``top_k`` is not positive.
        """
        if self._index is None:
            raise RuntimeError(
                "Index is not initialised. Call create_collection() first."
            )

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"query_vector length {len(query_vector)} does not match "
                f"vector_size {self.vector_size}."
            )

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        # Over-fetch to account for results filtered out post-search
        candidates = min(top_k * 10, self._index.ntotal)
        query_np = np.array([query_vector], dtype=np.float32)
        distances, indices = self._index.search(query_np, candidates)

        results: list[RetrievalResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for empty slots when candidates > ntotal
                continue

            chunk = self._chunks[idx]

            if filters and not self._matches(chunk, filters):
                continue

            results.append(
                RetrievalResult(
                    score=float(dist),
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    text=chunk.text,
                    repo_name=chunk.repo_name,
                )
            )

            if len(results) == top_k:
                break

        return results

    def delete_collection(self) -> None:
        """Wipe the index and all stored chunk metadata.

        After this call, ``create_collection()`` must be called again
        before any ``upsert`` or ``search`` operations.
        """
        self._index = None
        self._chunks = []
        logger.info("Deleted FAISS collection.")

    def _matches(self, chunk: CodeChunk, filters: SearchFilters) -> bool:
        """Return True if the chunk satisfies all filter conditions.

        Iterates over every key-value pair in ``filters`` and checks the
        corresponding attribute on ``chunk``. All conditions must pass
        (logical AND).

        Args:
            chunk:   The code chunk to evaluate.
            filters: A ``SearchFilters`` dict with zero or more conditions.

        Returns:
            ``True`` if the chunk matches all filters, ``False`` otherwise.
        """
        for field, value in filters.items():
            if getattr(chunk, field, None) != value:
                return False
        return True
