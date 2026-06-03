"""
A dataclass for a chunk of source code.

This data model is the primary unit of data exchanged between the ingestion layer and the retrieval layer.
"""

from __future__ import annotations
from pydantic import BaseModel, Field

__all__ = ["CodeChunk"]

class CodeChunk(BaseModel):
    """
    CodeChunk là data model thống nhất chứa các thông tin cho một chunk source code bất kỳ. 

    Attributes:
        chunk_id: str
        field_path: str
        language: str
        start_line: int
        end_line: int
        text: str
        repo_name: str
    """
    # Lưu chunk_id theo dạng đường dẫn và dòng sẽ bổ sung đầy đủ thông tin hơn cho một chunk, thay vì là những có số random.
    chunk_id: str = Field(examples=["path/to/your/file/src.py:from_line-end_line"])
    file_path: str 
    language: str # Ngôn ngữ mà đoạn code đó dùng
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str # Nội dung đoạn code
    repo_name: str | None = None # thuộc về repo nào. Có thể lưu hoặc không lưu