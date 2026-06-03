"""
A dataclass for a result of retrieval service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from codebase_rag.domain.code_chunk import CodeChunk
class RetrievalResult(BaseModel):
    """
    A dataclass that stores result data from retrieval service.

    Attributes:
        content: str 
    """
    score: float = Field(ge=0.0, le=1.0, description="Score for similarity search")
    content: str = Field(description="Text result of retrieval service")
    chunk_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    repo_name: str | None = None

    @classmethod
    def _load_from_codechunk(cls, chunk: CodeChunk, score: float):
        """
        Construct a RetrievalResult from CodeChunk.

        Args:
            chunk (CodeChunk): The input CodeChunk
            score (float): The score of retrieval system
        """
        return cls(
            score=score,
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            language=chunk.language,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            text=chunk.text,
            repo_name=chunk.repo_name,
        )