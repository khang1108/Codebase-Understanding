"""
A Protocol for Vector Store.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, TypedDict
from codebase_rag.domain.code_chunk import CodeChunk
from codebase_rag.domain.retreival_result import RetrievalResult

__all__ = ["IVectorStore", "SearchFilters"]

class SearchFilters(TypedDict, total=False):
    """
    A dataclass to store some filter config for RetrievalResult.

    You do not need to fill all fields of this class. 

    Attributes:
        language: Apply ``language`` filter for RetrievalResult (e.g. ``python``, ``cpp``).
        file_path: Apply ``file_path`` filter for RetrievalResult. Only return chunks from this file path.
    """
    language: str
    file_path: str

@runtime_checkable
class IVectorStore(Protocol):
    """
    Structural interface for Vector Database implementations.

    A class satisfies this Protocol if it exposes ``create_collection``,
    ``upsert``, ``search``, ``delete_collection`` with correct signatures.
    Explicit inheritance from this class is not required
    """

    def create_collection(self) -> None:
        """
        Ensure the target collection exists, creating it if needed.

        This method is idempotent, feel free to call multiple times.
        Must be called before any ``upsert`` or ``search`` operations.
        """
        ...
    
    def upsert(
            self, 
            chunks: list[CodeChunk],
            vector: list[list[float]],
            overwrite: bool = True
    ) -> None:
        """
        Insert or update chunks with their computed embedding vectors.

        ``chunks`` and ``vector`` must be parallel lists of equal length.
        If there is a chunk with the same ``chunk_id`` already existed and ``overwrite`` is True,
        it is overwritten.
        
        Args:
            chunks (list[CodeChunk]): A list of CodeChunk need to be added.
            vector (list[list[float]]): A list of embedding vectors of CodeChunks.
            overwrite (bool): Whether overwrite chunks with the same ``chunk_id`` or not (Default = False).
        """
        ...
    
    def search(
            self,
            query_vector: list[float],
            filter: SearchFilters,
            top_k: int = 10
    ) -> list[RetrievalResult]:
        """Find the most similar chunks to a query vector.
        
        Args:
            query_vector (list[float]): A embedding vector of a query.
            filter (SearchFilters): Optional payload filters. Only return chunks
                                        that match all specified fields.
            top_k (int): Maximum number of results to return.
        
        Returns:
            Results orders by descending similarity score. Please aware that it maybe
            return fewer than ``top_k`` items if the collection has fewer matching entries.

        Raises:
            RuntimeError: If you haven't created a collection.
        """
        ...

    def delete(self) -> None:
        """Drop the entire collection and all stored vectors"""
        ...