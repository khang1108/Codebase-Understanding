from domain.models.rag_answer import RagAnswer, SourceChunk

def extract_context_results(results) -> str:
    """3/6/2026 Function dùng để trích xuất kết quả context từ danh sách các chunk đã được truy xuất."""
    if isinstance(results, dict):
        return results.get("text") or results.get("content", "")
    return getattr(results, "content", "") or getattr(results, "text", "")

def build_context(retrieved_chunks: list) -> list:
    """3/6/2026 Function dùng để xây dựng context từ các chunk đã được truy xuất, 
    nó sẽ chuyển đổi các chunk này thành một định dạng chuẩn để sử dụng trong quá trình tạo câu trả lời."""
    context = []
    for chunk in retrieved_chunks:  
        if isinstance(chunk, dict):
            source_chunk = SourceChunk(
                content=extract_context_results(chunk),
                metadata=chunk.get("metadata", {}),
                start_line=chunk.get("start_line"),
                end_line=chunk.get("end_line"),
                file_path=chunk.get("file_path"),
                source=chunk.get("source"),
                score=chunk.get("score"),
                timestamp=chunk.get("timestamp")
            )
        else:
            source_chunk = SourceChunk(
                content=getattr(chunk, "content", ""),
                metadata=getattr(chunk, "metadata", {}),
                start_line=getattr(chunk, "start_line", None),
                end_line=getattr(chunk, "end_line", None),
                file_path=getattr(chunk, "file_path", None),
                source=getattr(chunk, "source", None),
                score=getattr(chunk, "score", None),                
                timestamp=getattr(chunk, "timestamp", None)
            )
        if source_chunk.content:  #không thêm vào context nếu content rỗng
            context.append(source_chunk)
    return context
def chunk_to_dict(chunks: list[SourceChunk]) -> list:
    """3/6/2026 Function dùng để chuyển đổi một danh sách các SourceChunk thành một danh sách các dictionary, 
    giúp dễ dàng lưu trữ hoặc truyền dữ liệu."""
    return [chunk.to_dict() for chunk in chunks]
