from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone
@dataclass
class SourceChunk:
    """3/6/2026 Class dùng để biểu diễn lớp dữ liệu chứa thông tin về một đoạn chunk ở đây nó dùng như interface
        để lưu trữ không phụ thuộc kiểu dữ liệu trả về từ retrival
    """
    content: str
    metadata: Optional[dict] = field(default_factory=dict)
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    file_path: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None
    timestamp: Optional[datetime] = None    
    def to_dict(self) -> dict:
        """3/6/2026 Method dùng để chuyển đổi đối tượng SourceChunk thành một dictionary, giúp dễ dàng lưu trữ hoặc truyền dữ liệu."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "file_path": self.file_path if hasattr(self, "file_path") else None,
            "source": self.source,
            "score": self.score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
@dataclass
class RagAnswer:
    """3/6/2026 class dùng để biểu diễn lớp dữ liệu mà rag sẽ trả về"""
    question: str
    answer: str
    source_chunks: List[SourceChunk] = field(default_factory=list)
    model_used: Optional[str] = None
    created_at: Optional[datetime] = None
    def to_dict(self) -> dict:
        """3/6/2026 Method dùng để chuyển đổi đối tượng RagAnswer thành một dictionary, giúp dễ dàng lưu trữ hoặc truyền dữ liệu."""
        return {
            "question": self.question,
            "answer": self.answer,
            "source_chunks": [chunk.to_dict() for chunk in self.source_chunks],
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }